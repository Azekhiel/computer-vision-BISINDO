"""
feature_engine.py - BISINDO V3.1 body-frame kinematic preprocessing.

Public output remains 179-D:
  0:144    spatial coordinates in absolute body-frame coordinates
            pose(6x3), left hand(21x3), right hand(21x3)
  144:176  hand joint angles
  176:179  original detection flags [pose_ok, left_hand_ok, right_hand_ok]

V3.1 changes the geometry, not the tensor size. Hands are reconstructed from a
kinematic tree with sequence-locked bone lengths. Occlusion gaps are handled as
short internal track segments only; long gaps are left empty instead of being
copied across the sequence. The renderer must draw V3.1 hands directly; it must
not re-anchor hands to pose wrists.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np

FEATURE_SCHEMA = "bisindo_v3_1_bodyframe_kinematic_segments"
LEGACY_SCHEMA = "legacy_or_unknown"

N_POSE = 18
N_HAND = 63
N_ANGLES = 16
N_FLAGS = 3

N_TOTAL_SPATIAL = N_POSE + N_HAND + N_HAND
N_TOTAL_HYBRID = N_TOTAL_SPATIAL + N_ANGLES * 2
N_TOTAL_WITH_FLAGS = N_TOTAL_HYBRID + N_FLAGS

SLICE_POSE = slice(0, N_POSE)
SLICE_LH = slice(N_POSE, N_POSE + N_HAND)
SLICE_RH = slice(N_POSE + N_HAND, N_TOTAL_SPATIAL)
SLICE_LH_A = slice(N_TOTAL_SPATIAL, N_TOTAL_SPATIAL + N_ANGLES)
SLICE_RH_A = slice(N_TOTAL_SPATIAL + N_ANGLES, N_TOTAL_HYBRID)
SLICE_FLAGS = slice(N_TOTAL_HYBRID, N_TOTAL_WITH_FLAGS)

IDX_POSE, IDX_LH, IDX_RH = 0, 1, 2

# Tree edges used for kinematic reconstruction. The order is parent-before-child.
HAND_TREE_EDGES = np.array(
    [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (0, 9), (9, 10), (10, 11), (11, 12),
        (0, 13), (13, 14), (14, 15), (15, 16),
        (0, 17), (17, 18), (18, 19), (19, 20),
    ],
    dtype=np.int64,
)

HAND_DRAW_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]

HAND_KINEMATIC_CHAINS = [
    (0, 1, 2), (1, 2, 3), (2, 3, 4),
    (0, 5, 6), (5, 6, 7), (6, 7, 8),
    (0, 9, 10), (9, 10, 11), (10, 11, 12),
    (0, 13, 14), (13, 14, 15), (14, 15, 16),
    (0, 17, 18), (17, 18, 19), (18, 19, 20),
    (5, 0, 9),
]


@dataclass(frozen=True)
class PreprocessConfig:
    schema: str = FEATURE_SCHEMA
    smooth_kernel: tuple[float, ...] = (1.0, 2.0, 3.0, 2.0, 1.0)
    imputed_smooth_weight: float = 0.35
    min_shoulder_width: float = 0.03
    shoulder_scale_clip_ratio: float = 0.20
    origin_clip_ratio: float = 0.20
    min_valid_hand_frames: int = 3
    max_occlusion_gap: int = 4
    max_edge_hold: int = 0
    hand_length_floor: float = 0.015
    hand_length_ceiling: float = 0.28
    hand_z_delta_limit: float = 0.35
    hand_total_mad_z: float = 5.5
    hand_bbox_mad_z: float = 5.5
    anchor_speed_min: float = 0.45
    anchor_speed_max: float = 1.25
    anchor_speed_mad_scale: float = 4.0
    hand_skeleton_step_max: float = 2.25
    pose_abs_limit: float = 4.0
    hand_abs_limit: float = 8.0
    angle_min: float = 0.0
    angle_max: float = math.pi


@dataclass
class BodyFrame:
    origin: np.ndarray
    scale: float
    valid: bool


@dataclass
class HandState:
    anchor: np.ndarray
    directions: np.ndarray
    lengths: np.ndarray
    valid: bool


@dataclass
class FrameObservation:
    pose_raw: np.ndarray
    left_hand_raw: np.ndarray
    right_hand_raw: np.ndarray
    mask: np.ndarray
    body_origin_raw: np.ndarray
    shoulder_scale_raw: float


DEFAULT_PREPROCESS_CONFIG = PreprocessConfig()


def joint_angle_xy(A: np.ndarray, B: np.ndarray, C: np.ndarray) -> float:
    BA = A[:2] - B[:2]
    BC = C[:2] - B[:2]
    cross_z = BA[0] * BC[1] - BA[1] * BC[0]
    dot = float(np.dot(BA, BC))
    return float(np.arctan2(abs(cross_z), dot))


def extract_hand_angles_from_points(points: np.ndarray) -> list[float]:
    if points is None or points.shape != (21, 3) or np.allclose(points, 0.0):
        return [0.0] * N_ANGLES
    return [joint_angle_xy(points[a], points[b], points[c]) for a, b, c in HAND_KINEMATIC_CHAINS]


def extract_hand_angles(landmarks) -> list[float]:
    if not landmarks:
        return [0.0] * N_ANGLES
    pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark], dtype=np.float32)
    return extract_hand_angles_from_points(pts)


def _landmarks_to_array(landmarks, count: int) -> np.ndarray:
    if not landmarks:
        return np.zeros((count, 3), dtype=np.float32)
    return np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark[:count]], dtype=np.float32)


def extract_frame_observation(results) -> FrameObservation:
    pose_raw = np.zeros((6, 3), dtype=np.float32)
    left_hand_raw = np.zeros((21, 3), dtype=np.float32)
    right_hand_raw = np.zeros((21, 3), dtype=np.float32)
    mask = np.zeros(N_FLAGS, dtype=bool)
    body_origin = np.zeros(3, dtype=np.float32)
    shoulder_scale = 1.0

    if results.pose_landmarks:
        mask[IDX_POSE] = True
        lm = results.pose_landmarks.landmark
        pose_raw = np.array([[lm[i].x, lm[i].y, lm[i].z] for i in range(11, 17)], dtype=np.float32)
        body_origin = (pose_raw[0] + pose_raw[1]) * 0.5
        shoulder_scale = float(np.linalg.norm(pose_raw[0, :2] - pose_raw[1, :2]))
        if not np.isfinite(shoulder_scale) or shoulder_scale < DEFAULT_PREPROCESS_CONFIG.min_shoulder_width:
            shoulder_scale = 1.0

    if results.left_hand_landmarks:
        mask[IDX_LH] = True
        left_hand_raw = _landmarks_to_array(results.left_hand_landmarks, 21)

    if results.right_hand_landmarks:
        mask[IDX_RH] = True
        right_hand_raw = _landmarks_to_array(results.right_hand_landmarks, 21)

    return FrameObservation(
        pose_raw=np.nan_to_num(pose_raw, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32),
        left_hand_raw=np.nan_to_num(left_hand_raw, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32),
        right_hand_raw=np.nan_to_num(right_hand_raw, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32),
        mask=mask,
        body_origin_raw=np.nan_to_num(body_origin, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32),
        shoulder_scale_raw=float(shoulder_scale),
    )


def observation_signature(observation: FrameObservation) -> np.ndarray:
    return np.concatenate(
        [
            observation.pose_raw.reshape(-1),
            observation.left_hand_raw.reshape(-1),
            observation.right_hand_raw.reshape(-1),
            observation.mask.astype(np.float32),
        ]
    ).astype(np.float32)


def extract_keypoints_relative(results):
    """
    Compatibility wrapper for old callers. New code should pass the returned
    FrameObservation into SequenceBuilder.add_observation().
    """
    obs = extract_frame_observation(results)
    builder = SequenceBuilder()
    builder.add_observation(obs)
    seq, _ = builder.build()
    if seq:
        vector = seq[0][:N_TOTAL_HYBRID].astype(np.float32)
    else:
        vector = np.zeros(N_TOTAL_HYBRID, dtype=np.float32)
    pose_lw = vector[SLICE_POSE].reshape(6, 3)[4] if obs.mask[IDX_POSE] else np.zeros(3, dtype=np.float32)
    pose_rw = vector[SLICE_POSE].reshape(6, 3)[5] if obs.mask[IDX_POSE] else np.zeros(3, dtype=np.float32)
    return vector, obs.mask.copy(), pose_lw.astype(np.float32), pose_rw.astype(np.float32)


def calculate_movement_score(prev_vector, curr_vector) -> float:
    if prev_vector is None or curr_vector is None:
        return 0.0
    prev_arr = np.asarray(prev_vector, dtype=np.float32)
    curr_arr = np.asarray(curr_vector, dtype=np.float32)
    return float(np.linalg.norm(curr_arr[:N_TOTAL_SPATIAL] - prev_arr[:N_TOTAL_SPATIAL]))


class SequenceBuilder:
    def __init__(
        self,
        smooth_window: int | None = None,
        smooth_poly: int | None = None,
        config: PreprocessConfig | None = None,
    ):
        self.config = config or DEFAULT_PREPROCESS_CONFIG
        if smooth_window is not None and smooth_window >= 3:
            self.smooth_kernel = _triangular_kernel(smooth_window)
        else:
            self.smooth_kernel = np.asarray(self.config.smooth_kernel, dtype=np.float32)
        self._observations: list[FrameObservation] = []
        self._legacy_vectors: list[np.ndarray] = []
        self._legacy_masks: list[np.ndarray] = []

    @property
    def _vectors(self):
        return self._observations if self._observations else self._legacy_vectors

    def add_observation(self, observation: FrameObservation):
        self._observations.append(observation)

    def add_frame(self, vector, mask=None, pose_lw=None, pose_rw=None):
        if isinstance(vector, FrameObservation):
            self.add_observation(vector)
            return

        arr = np.asarray(vector, dtype=np.float32).reshape(-1)
        if arr.shape[0] == N_TOTAL_WITH_FLAGS:
            inferred_mask = arr[SLICE_FLAGS] >= 0.5
            arr = arr[:N_TOTAL_HYBRID]
            if mask is None:
                mask = inferred_mask
        if arr.shape[0] != N_TOTAL_HYBRID:
            raise ValueError(f"Expected frame dim {N_TOTAL_HYBRID}, got {arr.shape[0]}")

        if mask is None:
            mask_arr = np.ones(N_FLAGS, dtype=bool)
        else:
            mask_arr = np.asarray(mask, dtype=bool).reshape(-1)
            if mask_arr.shape[0] != N_FLAGS:
                raise ValueError(f"Expected mask dim {N_FLAGS}, got {mask_arr.shape[0]}")

        self._legacy_vectors.append(np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32))
        self._legacy_masks.append(mask_arr)

    def reset(self):
        self._observations.clear()
        self._legacy_vectors.clear()
        self._legacy_masks.clear()

    def build(self):
        if self._observations:
            return self._build_from_observations()
        return self._build_from_legacy_vectors()

    def _build_from_observations(self):
        observations = self._observations
        T = len(observations)
        if T == 0:
            return [], []

        masks = np.stack([o.mask for o in observations]).astype(bool)
        origins, scales = self._stable_body_frames(observations, masks)

        pose = self._build_pose(observations, masks, origins, scales)
        left_hand = self._build_hand(observations, masks[:, IDX_LH], origins, scales, "left")
        right_hand = self._build_hand(observations, masks[:, IDX_RH], origins, scales, "right")

        sanitizer_masks = masks.copy()
        sanitizer_masks[:, IDX_LH] = _hand_frame_nonzero(left_hand)
        sanitizer_masks[:, IDX_RH] = _hand_frame_nonzero(right_hand)

        left_angles = np.array([extract_hand_angles_from_points(left_hand[i]) for i in range(T)], dtype=np.float32)
        right_angles = np.array([extract_hand_angles_from_points(right_hand[i]) for i in range(T)], dtype=np.float32)

        features = np.concatenate(
            [
                pose.reshape(T, N_POSE),
                left_hand.reshape(T, N_HAND),
                right_hand.reshape(T, N_HAND),
                left_angles,
                right_angles,
            ],
            axis=1,
        )
        features = sanitize_sequence(features, masks=sanitizer_masks, preserve_flags=False)
        augmented = np.concatenate([features, masks.astype(np.float32)], axis=1).astype(np.float32)
        scores = self._compute_scores(features)
        return [augmented[i] for i in range(T)], scores

    def _stable_body_frames(self, observations, masks):
        T = len(observations)
        raw_origins = np.stack([o.body_origin_raw for o in observations]).astype(np.float32)
        raw_scales = np.array([o.shoulder_scale_raw for o in observations], dtype=np.float32)
        pose_valid = masks[:, IDX_POSE] & np.isfinite(raw_scales) & (raw_scales >= self.config.min_shoulder_width)

        if pose_valid.any():
            filled_origins = _fill_matrix_gaps(raw_origins, pose_valid)
            filled_scales = _fill_vector_gaps(raw_scales, pose_valid)
            median_scale = float(np.median(raw_scales[pose_valid]))
            scale_lo = median_scale * (1.0 - self.config.shoulder_scale_clip_ratio)
            scale_hi = median_scale * (1.0 + self.config.shoulder_scale_clip_ratio)
            filled_scales = np.clip(filled_scales, scale_lo, scale_hi)
            stable_scales = np.full(T, median_scale, dtype=np.float32)

            origin_med = np.median(raw_origins[pose_valid], axis=0)
            origin_span = median_scale * self.config.origin_clip_ratio
            filled_origins = np.clip(filled_origins, origin_med - origin_span, origin_med + origin_span)
            stable_origins = _positive_smooth(filled_origins, np.ones(T, dtype=bool), self.smooth_kernel)
        else:
            stable_origins = np.zeros((T, 3), dtype=np.float32)
            stable_scales = np.ones(T, dtype=np.float32)

        stable_scales = np.maximum(stable_scales, self.config.min_shoulder_width).astype(np.float32)
        return stable_origins.astype(np.float32), stable_scales.astype(np.float32)

    def _build_pose(self, observations, masks, origins, scales):
        T = len(observations)
        pose_body = np.zeros((T, 6, 3), dtype=np.float32)
        pose_valid = masks[:, IDX_POSE]
        for i, obs in enumerate(observations):
            if pose_valid[i]:
                pose_body[i] = _to_body_frame(obs.pose_raw, origins[i], scales[i])
        if pose_valid.any():
            flat = pose_body.reshape(T, -1)
            flat = _fill_matrix_gaps(flat, pose_valid)
            flat = _positive_smooth(flat, pose_valid, self.smooth_kernel, self.config.imputed_smooth_weight)
            pose_body = flat.reshape(T, 6, 3)
        pose_body = np.clip(pose_body, -self.config.pose_abs_limit, self.config.pose_abs_limit)
        return pose_body.astype(np.float32)

    def _build_hand(self, observations, hand_valid, origins, scales, side: str):
        T = len(observations)
        hand_valid = np.asarray(hand_valid, dtype=bool)
        anchors = np.zeros((T, 3), dtype=np.float32)
        directions = np.zeros((T, len(HAND_TREE_EDGES), 3), dtype=np.float32)
        lengths = np.zeros((T, len(HAND_TREE_EDGES)), dtype=np.float32)
        z_deltas = np.zeros((T, len(HAND_TREE_EDGES)), dtype=np.float32)
        bbox = np.zeros(T, dtype=np.float32)

        for i, obs in enumerate(observations):
            if not hand_valid[i]:
                continue
            raw = obs.left_hand_raw if side == "left" else obs.right_hand_raw
            pts = _to_body_frame(raw, origins[i], scales[i])
            if not np.isfinite(pts).all() or np.allclose(pts, 0.0):
                continue

            anchors[i] = pts[0]
            bbox[i] = float(np.linalg.norm(np.ptp(pts[:, :2], axis=0)))
            for e_idx, (parent, child) in enumerate(HAND_TREE_EDGES):
                vec = pts[child] - pts[parent]
                norm = float(np.linalg.norm(vec[:2]))
                lengths[i, e_idx] = norm
                z_deltas[i, e_idx] = float(np.clip(vec[2], -self.config.hand_z_delta_limit, self.config.hand_z_delta_limit))
                if norm > 1e-6:
                    directions[i, e_idx, :2] = vec[:2] / norm

        quality_valid, vmax = _quality_gate_hand(anchors, lengths, bbox, hand_valid, self.config)
        if int(quality_valid.sum()) < self.config.min_valid_hand_frames:
            return np.zeros((T, 21, 3), dtype=np.float32)

        locked_lengths = np.median(lengths[quality_valid], axis=0)
        locked_lengths = np.clip(locked_lengths, self.config.hand_length_floor, self.config.hand_length_ceiling)
        locked_z = np.median(z_deltas[quality_valid], axis=0)
        locked_z = np.clip(locked_z, -self.config.hand_z_delta_limit, self.config.hand_z_delta_limit)

        anchor_filled, usable, imputed = _fill_short_internal_gaps(
            anchors,
            quality_valid,
            self.config.max_occlusion_gap,
            self.config.max_edge_hold,
        )
        anchor_conf = quality_valid & usable
        anchor_smooth = _positive_smooth_segments(
            anchor_filled,
            usable,
            anchor_conf,
            self.smooth_kernel,
            self.config.imputed_smooth_weight,
        )

        flat_dirs = directions.reshape(T, -1)
        flat_dirs, dir_usable, _ = _fill_short_internal_gaps(
            flat_dirs,
            quality_valid,
            self.config.max_occlusion_gap,
            self.config.max_edge_hold,
        )
        flat_dirs = _positive_smooth_segments(
            flat_dirs,
            dir_usable,
            anchor_conf,
            self.smooth_kernel,
            self.config.imputed_smooth_weight,
        )
        dirs = flat_dirs.reshape(T, len(HAND_TREE_EDGES), 3)
        dirs = _normalize_vectors(dirs)

        frame_valid = usable & dir_usable
        frame_valid = _clamp_anchor_steps(anchor_smooth, frame_valid, vmax, quality_valid)

        reconstructed = np.zeros((T, 21, 3), dtype=np.float32)
        reconstructed[frame_valid, 0, :] = anchor_smooth[frame_valid]
        for e_idx, (parent, child) in enumerate(HAND_TREE_EDGES):
            reconstructed[frame_valid, child, :2] = (
                reconstructed[frame_valid, parent, :2] + dirs[frame_valid, e_idx, :2] * locked_lengths[e_idx]
            )
            reconstructed[frame_valid, child, 2] = reconstructed[frame_valid, parent, 2] + locked_z[e_idx]

        frame_valid = _clamp_skeleton_steps(reconstructed, frame_valid, quality_valid, self.config.hand_skeleton_step_max)
        reconstructed[~frame_valid] = 0.0
        reconstructed = np.clip(reconstructed, -self.config.hand_abs_limit, self.config.hand_abs_limit)
        return reconstructed.astype(np.float32)

    def _build_from_legacy_vectors(self):
        if not self._legacy_vectors:
            return [], []
        seq = np.stack(self._legacy_vectors).astype(np.float32)
        masks = np.stack(self._legacy_masks).astype(bool)
        features = sanitize_sequence(seq, masks=masks, preserve_flags=False)
        augmented = np.concatenate([features, masks.astype(np.float32)], axis=1).astype(np.float32)
        scores = self._compute_scores(features)
        return [augmented[i] for i in range(len(augmented))], scores

    @staticmethod
    def _compute_scores(seq: np.ndarray) -> list[float]:
        scores = [0.0]
        for i in range(1, len(seq)):
            scores.append(float(np.linalg.norm(seq[i, :N_TOTAL_SPATIAL] - seq[i - 1, :N_TOTAL_SPATIAL])))
        return scores


def sanitize_sequence(
    sequence: np.ndarray,
    masks: np.ndarray | None = None,
    config: PreprocessConfig | None = None,
    preserve_flags: bool = True,
    zero_missing_hands: bool = False,
) -> np.ndarray:
    cfg = config or DEFAULT_PREPROCESS_CONFIG
    arr = np.asarray(sequence, dtype=np.float32)
    was_1d = arr.ndim == 1
    if was_1d:
        arr = arr.reshape(1, -1)

    if arr.shape[1] == N_TOTAL_WITH_FLAGS:
        features = arr[:, :N_TOTAL_HYBRID].copy()
        flags = arr[:, SLICE_FLAGS].copy()
        if masks is None:
            masks = flags >= 0.5
    elif arr.shape[1] == N_TOTAL_HYBRID:
        features = arr.copy()
        flags = None
    else:
        raise ValueError(f"Expected {N_TOTAL_HYBRID} or {N_TOTAL_WITH_FLAGS} features, got {arr.shape[1]}")

    if masks is None:
        masks = np.ones((features.shape[0], N_FLAGS), dtype=bool)
    else:
        masks = np.asarray(masks, dtype=bool)
        if masks.ndim == 1:
            masks = masks.reshape(1, -1)
        if masks.shape != (features.shape[0], N_FLAGS):
            raise ValueError(f"Expected masks shape ({features.shape[0]}, {N_FLAGS}), got {masks.shape}")

    T = features.shape[0]
    pose = np.clip(features[:, SLICE_POSE].reshape(T, 6, 3), -cfg.pose_abs_limit, cfg.pose_abs_limit)
    left = _rigidify_hand_block(features[:, SLICE_LH].reshape(T, 21, 3), masks[:, IDX_LH], cfg, zero_missing_hands)
    right = _rigidify_hand_block(features[:, SLICE_RH].reshape(T, 21, 3), masks[:, IDX_RH], cfg, zero_missing_hands)
    left_angles = np.clip(features[:, SLICE_LH_A], cfg.angle_min, cfg.angle_max)
    right_angles = np.clip(features[:, SLICE_RH_A], cfg.angle_min, cfg.angle_max)

    sanitized = np.concatenate(
        [pose.reshape(T, N_POSE), left.reshape(T, N_HAND), right.reshape(T, N_HAND), left_angles, right_angles],
        axis=1,
    ).astype(np.float32)

    if flags is not None:
        if preserve_flags:
            flags = (flags >= 0.5).astype(np.float32)
        sanitized = np.concatenate([sanitized, flags], axis=1)

    return sanitized.reshape(-1).astype(np.float32) if was_1d else sanitized.astype(np.float32)


def is_current_feature_frame(row_or_df) -> bool:
    if hasattr(row_or_df, "columns"):
        if "feature_version" not in row_or_df.columns:
            return False
        return bool((row_or_df["feature_version"] == FEATURE_SCHEMA).any())
    try:
        return str(row_or_df.get("feature_version", LEGACY_SCHEMA)) == FEATURE_SCHEMA
    except AttributeError:
        return False


def filter_current_feature_rows(df):
    if "feature_version" not in df.columns:
        return df.iloc[0:0].copy()
    return df[df["feature_version"] == FEATURE_SCHEMA].copy()


def ensure_feature_dim(sequence: Iterable[np.ndarray], expected_dim: int = N_TOTAL_WITH_FLAGS) -> np.ndarray:
    arr = np.asarray(list(sequence), dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != expected_dim:
        raise ValueError(f"Expected sequence shape (T, {expected_dim}), got {arr.shape}")
    return arr


def _to_body_frame(points: np.ndarray, origin: np.ndarray, scale: float) -> np.ndarray:
    denom = max(float(scale), DEFAULT_PREPROCESS_CONFIG.min_shoulder_width)
    return ((points - origin.reshape(1, 3)) / denom).astype(np.float32)


def _triangular_kernel(window: int) -> np.ndarray:
    win = int(max(3, window))
    if win % 2 == 0:
        win -= 1
    half = win // 2
    values = np.array([half + 1 - abs(i - half) for i in range(win)], dtype=np.float32)
    return values / max(float(values.sum()), 1e-8)


def _fill_vector_gaps(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    return _fill_matrix_gaps(values.reshape(-1, 1), valid).reshape(-1)


def _fill_matrix_gaps(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32).copy()
    valid = np.asarray(valid, dtype=bool)
    T = arr.shape[0]
    valid_idx = np.where(valid)[0]
    if valid_idx.size == 0:
        return np.zeros_like(arr, dtype=np.float32)
    if valid_idx.size == 1:
        out = np.zeros_like(arr, dtype=np.float32)
        out[valid_idx[0]] = arr[valid_idx[0]]
        return out

    out = arr.copy()
    for t in range(T):
        if valid[t]:
            continue
        pos = int(np.searchsorted(valid_idx, t))
        if pos == 0:
            out[t] = arr[valid_idx[0]]
        elif pos == valid_idx.size:
            out[t] = arr[valid_idx[-1]]
        else:
            left = int(valid_idx[pos - 1])
            right = int(valid_idx[pos])
            alpha = (t - left) / float(right - left)
            out[t] = (1.0 - alpha) * arr[left] + alpha * arr[right]
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _positive_smooth(values, valid, kernel, imputed_weight=0.35):
    arr = np.asarray(values, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)
    T = arr.shape[0]
    if T < 3:
        return arr.copy()

    ker = np.asarray(kernel, dtype=np.float32)
    if ker.size % 2 == 0:
        ker = ker[:-1]
    ker = np.maximum(ker, 0.0)
    if float(ker.sum()) <= 0.0:
        return arr.copy()

    half = ker.size // 2
    conf = np.where(valid, 1.0, imputed_weight).astype(np.float32)
    out = arr.copy()
    for t in range(T):
        lo = max(0, t - half)
        hi = min(T, t + half + 1)
        k_lo = half - (t - lo)
        k_hi = k_lo + (hi - lo)
        weights = ker[k_lo:k_hi] * conf[lo:hi]
        denom = float(weights.sum())
        if denom > 1e-8:
            out[t] = (arr[lo:hi] * weights.reshape(-1, *([1] * (arr.ndim - 1)))).sum(axis=0) / denom
    return out.astype(np.float32)


def _normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    safe = np.where(norms > 1e-6, vectors / np.maximum(norms, 1e-6), 0.0)
    return np.nan_to_num(safe, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _hand_frame_nonzero(hand: np.ndarray) -> np.ndarray:
    pts = np.asarray(hand, dtype=np.float32).reshape(-1, 21, 3)
    return np.max(np.linalg.norm(pts[:, :, :2], axis=2), axis=1) > 1e-6


def _robust_sigma(values: np.ndarray) -> float:
    vals = np.asarray(values, dtype=np.float32)
    if vals.size == 0:
        return 0.0
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med)))
    return 1.4826 * mad


def _robust_inlier_mask(
    values: np.ndarray,
    valid: np.ndarray,
    z_limit: float,
    lower: float | None = None,
    upper: float | None = None,
) -> np.ndarray:
    vals = np.asarray(values, dtype=np.float32)
    out = np.asarray(valid, dtype=bool).copy() & np.isfinite(vals)
    if lower is not None:
        out &= vals >= float(lower)
    if upper is not None:
        out &= vals <= float(upper)

    idx = np.where(out)[0]
    if idx.size >= 5:
        sample = vals[idx]
        med = float(np.median(sample))
        sigma = _robust_sigma(sample)
        if sigma > 1e-6:
            out &= np.abs(vals - med) <= float(z_limit) * sigma
    return out


def _adaptive_anchor_vmax(anchors: np.ndarray, valid: np.ndarray, cfg: PreprocessConfig) -> float:
    idx = np.where(valid)[0]
    if idx.size < 2:
        return float(cfg.anchor_speed_min)

    speeds = []
    for left, right in zip(idx[:-1], idx[1:]):
        dt = max(int(right - left), 1)
        speed = float(np.linalg.norm(anchors[right, :2] - anchors[left, :2]) / dt)
        if np.isfinite(speed):
            speeds.append(speed)

    if not speeds:
        return float(cfg.anchor_speed_min)

    speeds_arr = np.asarray(speeds, dtype=np.float32)
    med = float(np.median(speeds_arr))
    sigma = _robust_sigma(speeds_arr)
    vmax = med + cfg.anchor_speed_mad_scale * sigma
    return float(min(cfg.anchor_speed_max, max(cfg.anchor_speed_min, vmax)))


def _gate_anchor_speed(anchors: np.ndarray, valid: np.ndarray, vmax: float) -> np.ndarray:
    gated = np.zeros_like(valid, dtype=bool)
    last_idx: int | None = None
    for idx in np.where(valid)[0]:
        if last_idx is None:
            gated[idx] = True
            last_idx = int(idx)
            continue

        dt = max(int(idx - last_idx), 1)
        speed = float(np.linalg.norm(anchors[idx, :2] - anchors[last_idx, :2]) / dt)
        if np.isfinite(speed) and speed <= vmax:
            gated[idx] = True
            last_idx = int(idx)
    return gated


def _quality_gate_hand(
    anchors: np.ndarray,
    lengths: np.ndarray,
    bbox: np.ndarray,
    detector_valid: np.ndarray,
    cfg: PreprocessConfig,
) -> tuple[np.ndarray, float]:
    total_length = np.sum(lengths, axis=1)
    finite = (
        np.asarray(detector_valid, dtype=bool)
        & np.isfinite(anchors).all(axis=1)
        & np.isfinite(lengths).all(axis=1)
        & np.isfinite(bbox)
    )
    total_floor = cfg.hand_length_floor * len(HAND_TREE_EDGES)
    total_ceiling = cfg.hand_length_ceiling * len(HAND_TREE_EDGES) * 1.35

    valid = _robust_inlier_mask(
        total_length,
        finite,
        cfg.hand_total_mad_z,
        lower=total_floor,
        upper=total_ceiling,
    )
    valid = _robust_inlier_mask(
        bbox,
        valid,
        cfg.hand_bbox_mad_z,
        lower=cfg.hand_length_floor,
        upper=cfg.hand_abs_limit,
    )
    valid &= np.max(np.abs(anchors[:, :2]), axis=1) <= cfg.hand_abs_limit

    vmax = _adaptive_anchor_vmax(anchors, valid, cfg)
    if int(valid.sum()) >= cfg.min_valid_hand_frames:
        valid = _gate_anchor_speed(anchors, valid, vmax)
    return valid.astype(bool), vmax


def _fill_short_internal_gaps(
    values: np.ndarray,
    valid: np.ndarray,
    max_gap: int,
    max_edge_hold: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = np.asarray(values, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)
    out = np.zeros_like(arr, dtype=np.float32)
    usable = np.zeros(valid.shape, dtype=bool)
    imputed = np.zeros(valid.shape, dtype=bool)
    idx = np.where(valid)[0]

    if idx.size == 0:
        return out, usable, imputed

    out[valid] = arr[valid]
    usable[valid] = True

    if max_edge_hold > 0:
        first = int(idx[0])
        last = int(idx[-1])
        lead_start = max(0, first - max_edge_hold)
        if lead_start < first:
            out[lead_start:first] = arr[first]
            usable[lead_start:first] = True
            imputed[lead_start:first] = True
        trail_end = min(len(valid), last + max_edge_hold + 1)
        if last + 1 < trail_end:
            out[last + 1:trail_end] = arr[last]
            usable[last + 1:trail_end] = True
            imputed[last + 1:trail_end] = True

    for left, right in zip(idx[:-1], idx[1:]):
        gap = int(right - left - 1)
        if gap <= 0 or gap > int(max_gap):
            continue
        for t in range(int(left) + 1, int(right)):
            alpha = (t - left) / float(right - left)
            out[t] = (1.0 - alpha) * arr[left] + alpha * arr[right]
            usable[t] = True
            imputed[t] = True

    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32), usable, imputed


def _true_segments(mask: np.ndarray) -> list[tuple[int, int]]:
    mask = np.asarray(mask, dtype=bool)
    segments: list[tuple[int, int]] = []
    start: int | None = None
    for i, value in enumerate(mask):
        if value and start is None:
            start = i
        elif not value and start is not None:
            segments.append((start, i))
            start = None
    if start is not None:
        segments.append((start, len(mask)))
    return segments


def _positive_smooth_with_confidence(values: np.ndarray, confidence: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    conf = np.asarray(confidence, dtype=np.float32).reshape(-1)
    T = arr.shape[0]
    if T < 3:
        return arr.copy()

    ker = np.asarray(kernel, dtype=np.float32)
    if ker.size % 2 == 0:
        ker = ker[:-1]
    ker = np.maximum(ker, 0.0)
    if float(ker.sum()) <= 0.0:
        return arr.copy()

    half = ker.size // 2
    out = arr.copy()
    for t in range(T):
        lo = max(0, t - half)
        hi = min(T, t + half + 1)
        k_lo = half - (t - lo)
        k_hi = k_lo + (hi - lo)
        weights = ker[k_lo:k_hi] * conf[lo:hi]
        denom = float(weights.sum())
        if denom > 1e-8:
            shape = (-1,) + (1,) * (arr.ndim - 1)
            out[t] = (arr[lo:hi] * weights.reshape(shape)).sum(axis=0) / denom
    return out.astype(np.float32)


def _positive_smooth_segments(
    values: np.ndarray,
    segment_mask: np.ndarray,
    original_valid: np.ndarray,
    kernel: np.ndarray,
    imputed_weight: float,
) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    segment_mask = np.asarray(segment_mask, dtype=bool)
    original_valid = np.asarray(original_valid, dtype=bool)
    out = np.zeros_like(arr, dtype=np.float32)

    for start, end in _true_segments(segment_mask):
        conf = np.where(original_valid[start:end], 1.0, float(imputed_weight)).astype(np.float32)
        out[start:end] = _positive_smooth_with_confidence(arr[start:end], conf, kernel)

    return out.astype(np.float32)


def _clamp_anchor_steps(
    anchors: np.ndarray,
    valid: np.ndarray,
    vmax: float,
    original_valid: np.ndarray | None = None,
) -> np.ndarray:
    out = np.asarray(valid, dtype=bool).copy()
    original = np.asarray(original_valid, dtype=bool) if original_valid is not None else out.copy()
    for start, end in _true_segments(out):
        last = start
        for idx in range(start + 1, end):
            speed = float(np.linalg.norm(anchors[idx, :2] - anchors[last, :2]))
            if not np.isfinite(speed) or speed > vmax:
                if original[idx]:
                    last = idx
                else:
                    out[idx] = False
            else:
                last = idx
    return out


def _clamp_skeleton_steps(
    hand: np.ndarray,
    valid: np.ndarray,
    original_valid: np.ndarray | None,
    max_step: float,
) -> np.ndarray:
    out = np.asarray(valid, dtype=bool).copy()
    original = np.asarray(original_valid, dtype=bool) if original_valid is not None else out.copy()
    flat = np.asarray(hand, dtype=np.float32).reshape(hand.shape[0], -1)
    for start, end in _true_segments(out):
        last = start
        for idx in range(start + 1, end):
            step = float(np.linalg.norm(flat[idx] - flat[last]))
            if not np.isfinite(step) or step > float(max_step):
                if original[idx]:
                    last = idx
                else:
                    out[idx] = False
            else:
                last = idx
    return out


def _rigidify_hand_block(hand: np.ndarray, valid: np.ndarray, cfg: PreprocessConfig, zero_missing: bool = False) -> np.ndarray:
    hand = np.nan_to_num(hand.copy(), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    valid = np.asarray(valid, dtype=bool)
    if valid.sum() < cfg.min_valid_hand_frames:
        return np.zeros_like(hand, dtype=np.float32)

    directions = np.zeros((hand.shape[0], len(HAND_TREE_EDGES), 3), dtype=np.float32)
    lengths = np.zeros((hand.shape[0], len(HAND_TREE_EDGES)), dtype=np.float32)
    z_deltas = np.zeros((hand.shape[0], len(HAND_TREE_EDGES)), dtype=np.float32)
    for e_idx, (parent, child) in enumerate(HAND_TREE_EDGES):
        vec = hand[:, child, :] - hand[:, parent, :]
        norm = np.linalg.norm(vec[:, :2], axis=1)
        lengths[:, e_idx] = norm
        z_deltas[:, e_idx] = vec[:, 2]
        directions[:, e_idx, :2] = np.where(norm[:, None] > 1e-6, vec[:, :2] / np.maximum(norm[:, None], 1e-6), 0.0)

    locked_lengths = np.median(lengths[valid], axis=0)
    locked_lengths = np.clip(locked_lengths, cfg.hand_length_floor, cfg.hand_length_ceiling)
    locked_z = np.median(z_deltas[valid], axis=0)
    dirs = _normalize_vectors(directions)

    out = np.zeros_like(hand)
    out[:, 0, :] = hand[:, 0, :]
    for e_idx, (parent, child) in enumerate(HAND_TREE_EDGES):
        out[:, child, :2] = out[:, parent, :2] + dirs[:, e_idx, :2] * locked_lengths[e_idx]
        out[:, child, 2] = out[:, parent, 2] + locked_z[e_idx]
    if zero_missing:
        out[~valid] = 0.0
    return np.clip(out, -cfg.hand_abs_limit, cfg.hand_abs_limit).astype(np.float32)
