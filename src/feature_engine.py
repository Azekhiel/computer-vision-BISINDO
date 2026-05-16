"""
feature_engine.py - BISINDO V3.2 body-frame kinematic preprocessing.

Public output remains 179-D:
  0:144    spatial coordinates in absolute body-frame coordinates
            pose(6x3), left hand(21x3), right hand(21x3)
  144:176  hand joint angles
  176:179  original detection flags [pose_ok, left_hand_ok, right_hand_ok]

V3.2 keeps the V3.1 kinematic reconstruction, but adds detector recovery and
traceable provenance. Holistic remains the primary observation source; fallback
MediaPipe Hands, Lucas-Kanade optical flow, and short constant-velocity
prediction can bridge detector drops without changing the public tensor shape.
The renderer must draw V3.2 hands directly; it must not re-anchor hands to pose
wrists.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
import time
from typing import Iterable

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover - OpenCV is optional for non-live numeric tests.
    cv2 = None

FEATURE_SCHEMA = "bisindo_v3_2_bodyframe_kinematic_recovered"
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
    tracker_max_flow_gap: int = 6
    tracker_max_prediction_gap: int = 2
    tracker_flow_min_success_ratio: float = 0.55
    tracker_flow_max_median_error: float = 32.0
    tracker_duplicate_anchor_radius: float = 0.10
    tracker_min_raw_bbox: float = 0.015
    tracker_max_raw_bbox: float = 1.35
    tracker_min_raw_total_length: float = 0.08
    tracker_max_raw_total_length: float = 7.5
    tracker_holistic_confidence: float = 1.0
    tracker_hands_confidence: float = 0.82
    tracker_flow_confidence: float = 0.68
    tracker_prediction_confidence: float = 0.25


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


@dataclass
class TrackedHandObservation:
    landmarks_raw: np.ndarray
    valid: bool
    source: str = "missing"
    confidence: float = 0.0
    quality_reason: str = "missing"
    roi: tuple[float, float, float, float] | None = None
    gap_age: int = 0
    original_detected: bool = False


@dataclass
class TrackedFrameObservation:
    pose_raw: np.ndarray
    left_hand_raw: np.ndarray
    right_hand_raw: np.ndarray
    mask: np.ndarray
    body_origin_raw: np.ndarray
    shoulder_scale_raw: float
    left_tracking: TrackedHandObservation
    right_tracking: TrackedHandObservation


@dataclass
class _HandTrackState:
    landmarks: np.ndarray | None = None
    prev_gray: np.ndarray | None = None
    velocity_xy: np.ndarray | None = None
    gap_age: int = 999
    source: str = "missing"


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


def _empty_tracked_hand(original_detected: bool = False, reason: str = "missing") -> TrackedHandObservation:
    return TrackedHandObservation(
        landmarks_raw=np.zeros((21, 3), dtype=np.float32),
        valid=False,
        source="missing",
        confidence=0.0,
        quality_reason=reason,
        roi=None,
        gap_age=999,
        original_detected=bool(original_detected),
    )


def _tracked_hand_from_landmarks(
    landmarks: np.ndarray,
    source: str,
    confidence: float,
    quality_reason: str,
    gap_age: int,
    original_detected: bool,
    config: PreprocessConfig | None = None,
) -> TrackedHandObservation:
    cfg = config or DEFAULT_PREPROCESS_CONFIG
    pts = np.nan_to_num(np.asarray(landmarks, dtype=np.float32).reshape(21, 3), nan=0.0, posinf=0.0, neginf=0.0)
    valid = _plausible_raw_hand(pts, cfg)
    if not valid:
        return _empty_tracked_hand(original_detected=original_detected, reason=f"invalid_{source}")
    return TrackedHandObservation(
        landmarks_raw=pts.astype(np.float32),
        valid=True,
        source=source,
        confidence=float(np.clip(confidence, 0.0, 1.0)),
        quality_reason=quality_reason,
        roi=_landmarks_roi(pts),
        gap_age=int(gap_age),
        original_detected=bool(original_detected),
    )


def _wrap_observation_as_tracked(observation: FrameObservation) -> TrackedFrameObservation:
    left = (
        _tracked_hand_from_landmarks(
            observation.left_hand_raw,
            "holistic",
            DEFAULT_PREPROCESS_CONFIG.tracker_holistic_confidence,
            "holistic_detected",
            0,
            True,
        )
        if bool(observation.mask[IDX_LH])
        else _empty_tracked_hand(False)
    )
    right = (
        _tracked_hand_from_landmarks(
            observation.right_hand_raw,
            "holistic",
            DEFAULT_PREPROCESS_CONFIG.tracker_holistic_confidence,
            "holistic_detected",
            0,
            True,
        )
        if bool(observation.mask[IDX_RH])
        else _empty_tracked_hand(False)
    )
    return TrackedFrameObservation(
        pose_raw=observation.pose_raw,
        left_hand_raw=left.landmarks_raw,
        right_hand_raw=right.landmarks_raw,
        mask=observation.mask.copy(),
        body_origin_raw=observation.body_origin_raw,
        shoulder_scale_raw=float(observation.shoulder_scale_raw),
        left_tracking=left,
        right_tracking=right,
    )


def extract_tracked_frame_observation(
    frame_bgr,
    holistic_results,
    hands_results=None,
    tracker: "HandRecoveryTracker | None" = None,
    config: PreprocessConfig | None = None,
) -> TrackedFrameObservation:
    """
    Extract one frame with detector provenance.

    The returned public mask remains the original Holistic detection mask. If
    fallback Hands, optical flow, or prediction recovers a hand, the recovered
    landmarks are used for geometry while the original detector flag stays 0.
    """
    base = extract_frame_observation(holistic_results)
    if tracker is None:
        return _wrap_observation_as_tracked(base)
    return tracker.update(frame_bgr, base, hands_results=hands_results, config=config)


def observation_signature(observation: FrameObservation | TrackedFrameObservation) -> np.ndarray:
    return np.concatenate(
        [
            observation.pose_raw.reshape(-1),
            observation.left_hand_raw.reshape(-1),
            observation.right_hand_raw.reshape(-1),
            observation.mask.astype(np.float32),
        ]
    ).astype(np.float32)


class HandRecoveryTracker:
    """
    Short-gap hand recovery for detector drops.

    Recovery is intentionally conservative. Holistic and fallback Hands reset a
    track. LK optical flow bridges short misses. Constant-velocity prediction is
    limited to a couple of frames and tagged separately so downstream audits can
    distinguish it from true detections.
    """

    def __init__(self, config: PreprocessConfig | None = None):
        self.config = config or DEFAULT_PREPROCESS_CONFIG
        self._state = {"left": _HandTrackState(), "right": _HandTrackState()}

    def reset(self):
        self._state = {"left": _HandTrackState(), "right": _HandTrackState()}

    def needs_fallback(self, observation: FrameObservation) -> bool:
        left_missing = not bool(observation.mask[IDX_LH])
        right_missing = not bool(observation.mask[IDX_RH])
        left_bad = bool(observation.mask[IDX_LH]) and not _plausible_raw_hand(observation.left_hand_raw, self.config)
        right_bad = bool(observation.mask[IDX_RH]) and not _plausible_raw_hand(observation.right_hand_raw, self.config)
        return left_missing or right_missing or left_bad or right_bad

    def update(
        self,
        frame_bgr,
        observation: FrameObservation,
        hands_results=None,
        config: PreprocessConfig | None = None,
    ) -> TrackedFrameObservation:
        cfg = config or self.config
        gray = _frame_to_gray(frame_bgr)
        fallback_candidates = _parse_hands_candidates(hands_results, cfg)
        assigned = self._assign_fallback_candidates(fallback_candidates, observation)

        left = self._recover_side("left", observation, assigned.get("left"), gray, cfg)
        right = self._recover_side("right", observation, assigned.get("right"), gray, cfg)

        return TrackedFrameObservation(
            pose_raw=observation.pose_raw,
            left_hand_raw=left.landmarks_raw,
            right_hand_raw=right.landmarks_raw,
            mask=observation.mask.copy(),
            body_origin_raw=observation.body_origin_raw,
            shoulder_scale_raw=float(observation.shoulder_scale_raw),
            left_tracking=left,
            right_tracking=right,
        )

    def _assign_fallback_candidates(self, candidates: list[dict], observation: FrameObservation) -> dict[str, np.ndarray]:
        assigned: dict[str, np.ndarray] = {}
        if not candidates:
            return assigned

        used: set[int] = set()
        side_order = ["left", "right"]
        side_order.sort(key=lambda name: 0 if self._state[name].landmarks is not None else 1)
        for side in side_order:
            idx = IDX_LH if side == "left" else IDX_RH
            raw = observation.left_hand_raw if side == "left" else observation.right_hand_raw
            if bool(observation.mask[idx]) and _plausible_raw_hand(raw, self.config):
                continue

            target = self._target_anchor(side, observation)
            best_i = None
            best_cost = float("inf")
            for i, candidate in enumerate(candidates):
                if i in used:
                    continue
                pts = candidate["landmarks"]
                anchor = pts[0, :2]
                other_side = "right" if side == "left" else "left"
                other_idx = IDX_RH if other_side == "right" else IDX_LH
                other_raw = observation.right_hand_raw if other_side == "right" else observation.left_hand_raw
                if bool(observation.mask[other_idx]) and _plausible_raw_hand(other_raw, self.config):
                    other_dist = float(np.linalg.norm(anchor - other_raw[0, :2]))
                    if other_dist < float(self.config.tracker_duplicate_anchor_radius):
                        continue
                cost = float(np.linalg.norm(anchor - target))
                label = str(candidate.get("label", "")).lower()
                if label == side:
                    cost -= 0.08
                elif label in ("left", "right"):
                    cost += 0.08
                cost -= 0.03 * float(candidate.get("score", 0.0))
                if cost < best_cost:
                    best_cost = cost
                    best_i = i

            if best_i is not None:
                used.add(best_i)
                assigned[side] = candidates[best_i]["landmarks"]
        return assigned

    def _target_anchor(self, side: str, observation: FrameObservation) -> np.ndarray:
        state = self._state[side]
        if state.landmarks is not None:
            return np.clip(state.landmarks[0, :2], 0.0, 1.0).astype(np.float32)
        if bool(observation.mask[IDX_POSE]):
            wrist_idx = 4 if side == "left" else 5
            return np.clip(observation.pose_raw[wrist_idx, :2], 0.0, 1.0).astype(np.float32)
        return np.array([0.5, 0.5], dtype=np.float32)

    def _recover_side(
        self,
        side: str,
        observation: FrameObservation,
        fallback_landmarks: np.ndarray | None,
        gray: np.ndarray | None,
        cfg: PreprocessConfig,
    ) -> TrackedHandObservation:
        idx = IDX_LH if side == "left" else IDX_RH
        raw = observation.left_hand_raw if side == "left" else observation.right_hand_raw
        state = self._state[side]

        if bool(observation.mask[idx]) and _plausible_raw_hand(raw, cfg):
            tracked = _tracked_hand_from_landmarks(
                raw,
                "holistic",
                cfg.tracker_holistic_confidence,
                "holistic_detected",
                0,
                True,
                cfg,
            )
            self._accept_state(side, tracked, gray, reset_gap=True)
            return tracked

        if fallback_landmarks is not None and _plausible_raw_hand(fallback_landmarks, cfg):
            tracked = _tracked_hand_from_landmarks(
                fallback_landmarks,
                "hands",
                cfg.tracker_hands_confidence,
                "fallback_hands_detected",
                0,
                bool(observation.mask[idx]),
                cfg,
            )
            self._accept_state(side, tracked, gray, reset_gap=True)
            return tracked

        flow = self._flow_recover(state, gray, cfg)
        if flow is not None:
            gap = min(int(state.gap_age) + 1, 999)
            conf = max(0.20, cfg.tracker_flow_confidence - 0.08 * max(gap - 1, 0))
            tracked = _tracked_hand_from_landmarks(
                flow,
                "flow",
                conf,
                "lk_recovered",
                gap,
                bool(observation.mask[idx]),
                cfg,
            )
            self._accept_state(side, tracked, gray, reset_gap=False)
            return tracked

        prediction = self._predict_state(state, cfg)
        if prediction is not None:
            gap = min(int(state.gap_age) + 1, 999)
            tracked = _tracked_hand_from_landmarks(
                prediction,
                "prediction",
                cfg.tracker_prediction_confidence,
                "constant_velocity_bridge",
                gap,
                bool(observation.mask[idx]),
                cfg,
            )
            self._accept_state(side, tracked, gray, reset_gap=False)
            return tracked

        state.gap_age = min(int(state.gap_age) + 1, 999)
        return _empty_tracked_hand(original_detected=bool(observation.mask[idx]), reason="detector_missing")

    def _flow_recover(
        self,
        state: _HandTrackState,
        gray: np.ndarray | None,
        cfg: PreprocessConfig,
    ) -> np.ndarray | None:
        if cv2 is None or gray is None or state.landmarks is None or state.prev_gray is None:
            return None
        if state.gap_age >= int(cfg.tracker_max_flow_gap):
            return None

        h, w = gray.shape[:2]
        prev_pts = state.landmarks[:, :2].copy()
        prev_pts[:, 0] *= float(w)
        prev_pts[:, 1] *= float(h)
        prev_pts = prev_pts.reshape(-1, 1, 2).astype(np.float32)

        try:
            next_pts, status, err = cv2.calcOpticalFlowPyrLK(
                state.prev_gray,
                gray,
                prev_pts,
                None,
                winSize=(21, 21),
                maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
            )
        except Exception:
            return None

        if next_pts is None or status is None:
            return None

        ok = status.reshape(-1).astype(bool)
        if float(ok.mean()) < float(cfg.tracker_flow_min_success_ratio):
            return None
        if err is not None:
            good_err = err.reshape(-1)[ok]
            if good_err.size and float(np.median(good_err)) > float(cfg.tracker_flow_max_median_error):
                return None

        next_xy = next_pts.reshape(-1, 2)
        out = state.landmarks.copy()
        out[ok, 0] = next_xy[ok, 0] / max(float(w), 1.0)
        out[ok, 1] = next_xy[ok, 1] / max(float(h), 1.0)
        out[:, :2] = np.clip(out[:, :2], -0.25, 1.25)
        return out.astype(np.float32)

    def _predict_state(self, state: _HandTrackState, cfg: PreprocessConfig) -> np.ndarray | None:
        if state.landmarks is None or state.velocity_xy is None:
            return None
        if state.gap_age >= int(cfg.tracker_max_prediction_gap):
            return None
        out = state.landmarks.copy()
        out[:, :2] = np.clip(out[:, :2] + state.velocity_xy, -0.15, 1.15)
        return out.astype(np.float32)

    def _accept_state(self, side: str, tracked: TrackedHandObservation, gray: np.ndarray | None, reset_gap: bool):
        state = self._state[side]
        if state.landmarks is not None:
            state.velocity_xy = tracked.landmarks_raw[:, :2] - state.landmarks[:, :2]
        else:
            state.velocity_xy = np.zeros((21, 2), dtype=np.float32)
        state.landmarks = tracked.landmarks_raw.copy()
        if gray is not None:
            state.prev_gray = gray.copy()
        state.gap_age = 0 if reset_gap else int(tracked.gap_age)
        state.source = tracked.source


class TrackingRegistryWriter:
    """Append-only JSONL writer for generated GIF/extraction traceability."""

    def __init__(self, log_path: str):
        self.log_path = log_path

    def write(self, record: dict):
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        payload = dict(record)
        payload.setdefault("created_at", time.strftime("%Y-%m-%dT%H:%M:%S%z"))
        payload.setdefault("feature_schema", FEATURE_SCHEMA)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


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
        self._observations: list[FrameObservation | TrackedFrameObservation] = []
        self._legacy_vectors: list[np.ndarray] = []
        self._legacy_masks: list[np.ndarray] = []
        self._last_metadata: list[dict] = []

    @property
    def _vectors(self):
        return self._observations if self._observations else self._legacy_vectors

    @property
    def last_build_metadata(self) -> list[dict]:
        return list(self._last_metadata)

    def add_observation(self, observation: FrameObservation | TrackedFrameObservation):
        self._observations.append(observation)

    def add_frame(self, vector, mask=None, pose_lw=None, pose_rw=None):
        if isinstance(vector, (FrameObservation, TrackedFrameObservation)):
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
        self._last_metadata.clear()

    def build(self):
        if self._observations:
            return self._build_from_observations()
        return self._build_from_legacy_vectors()

    def _build_from_observations(self):
        observations = self._observations
        T = len(observations)
        if T == 0:
            return [], []

        original_masks = np.stack([o.mask for o in observations]).astype(bool)
        processing_masks, tracking_conf = self._processing_masks_and_confidence(observations, original_masks)
        origins, scales = self._stable_body_frames(observations, original_masks)

        pose = self._build_pose(observations, original_masks, origins, scales)
        left_hand = self._build_hand(
            observations,
            processing_masks[:, IDX_LH],
            origins,
            scales,
            "left",
            confidence=tracking_conf[:, IDX_LH],
        )
        right_hand = self._build_hand(
            observations,
            processing_masks[:, IDX_RH],
            origins,
            scales,
            "right",
            confidence=tracking_conf[:, IDX_RH],
        )

        sanitizer_masks = original_masks.copy()
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
        augmented = np.concatenate([features, original_masks.astype(np.float32)], axis=1).astype(np.float32)
        scores = self._compute_scores(features)
        self._last_metadata = self._make_tracking_metadata(observations, original_masks, sanitizer_masks)
        return [augmented[i] for i in range(T)], scores

    def _processing_masks_and_confidence(self, observations, original_masks):
        masks = original_masks.copy()
        confidence = original_masks.astype(np.float32)
        for i, obs in enumerate(observations):
            if isinstance(obs, TrackedFrameObservation):
                masks[i, IDX_LH] = bool(obs.left_tracking.valid)
                masks[i, IDX_RH] = bool(obs.right_tracking.valid)
                confidence[i, IDX_LH] = float(obs.left_tracking.confidence) if obs.left_tracking.valid else 0.0
                confidence[i, IDX_RH] = float(obs.right_tracking.confidence) if obs.right_tracking.valid else 0.0
        return masks.astype(bool), np.clip(confidence, 0.0, 1.0).astype(np.float32)

    def _make_tracking_metadata(self, observations, original_masks, sanitizer_masks):
        metadata: list[dict] = []
        for i, obs in enumerate(observations):
            if isinstance(obs, TrackedFrameObservation):
                left = obs.left_tracking
                right = obs.right_tracking
            else:
                left = (
                    _tracked_hand_from_landmarks(
                        obs.left_hand_raw,
                        "holistic",
                        self.config.tracker_holistic_confidence,
                        "legacy_holistic",
                        0,
                        True,
                        self.config,
                    )
                    if original_masks[i, IDX_LH]
                    else _empty_tracked_hand(False)
                )
                right = (
                    _tracked_hand_from_landmarks(
                        obs.right_hand_raw,
                        "holistic",
                        self.config.tracker_holistic_confidence,
                        "legacy_holistic",
                        0,
                        True,
                        self.config,
                    )
                    if original_masks[i, IDX_RH]
                    else _empty_tracked_hand(False)
                )
            metadata.append(
                {
                    "pose_detected": bool(original_masks[i, IDX_POSE]),
                    "left": _tracking_metadata_dict(left, bool(sanitizer_masks[i, IDX_LH])),
                    "right": _tracking_metadata_dict(right, bool(sanitizer_masks[i, IDX_RH])),
                }
            )
        return metadata

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

    def _build_hand(self, observations, hand_valid, origins, scales, side: str, confidence: np.ndarray | None = None):
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

        source_confidence = (
            np.asarray(confidence, dtype=np.float32).reshape(T)
            if confidence is not None
            else np.asarray(hand_valid, dtype=np.float32).reshape(T)
        )
        quality_valid, vmax = _quality_gate_hand(anchors, lengths, bbox, hand_valid, self.config)
        quality_valid &= source_confidence > 0.0
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
        anchor_conf = np.where(quality_valid & usable, source_confidence, 0.0).astype(np.float32)
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
        self._last_metadata = [
            {
                "pose_detected": bool(masks[i, IDX_POSE]),
                "left": {
                    "source": "holistic" if masks[i, IDX_LH] else "missing",
                    "confidence": 1.0 if masks[i, IDX_LH] else 0.0,
                    "gap_age": 0 if masks[i, IDX_LH] else 999,
                    "quality_reason": "legacy_vector",
                    "original_detected": bool(masks[i, IDX_LH]),
                    "rendered": bool(masks[i, IDX_LH]),
                    "roi": None,
                },
                "right": {
                    "source": "holistic" if masks[i, IDX_RH] else "missing",
                    "confidence": 1.0 if masks[i, IDX_RH] else 0.0,
                    "gap_age": 0 if masks[i, IDX_RH] else 999,
                    "quality_reason": "legacy_vector",
                    "original_detected": bool(masks[i, IDX_RH]),
                    "rendered": bool(masks[i, IDX_RH]),
                    "roi": None,
                },
            }
            for i in range(len(augmented))
        ]
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


def flatten_tracking_metadata(metadata: dict | None) -> dict:
    if not metadata:
        metadata = {}
    left = metadata.get("left", {}) if isinstance(metadata, dict) else {}
    right = metadata.get("right", {}) if isinstance(metadata, dict) else {}
    return {
        "tracking_left_source": str(left.get("source", "unknown")),
        "tracking_right_source": str(right.get("source", "unknown")),
        "tracking_left_confidence": float(left.get("confidence", 0.0)),
        "tracking_right_confidence": float(right.get("confidence", 0.0)),
        "tracking_left_gap_age": int(left.get("gap_age", 999)),
        "tracking_right_gap_age": int(right.get("gap_age", 999)),
        "tracking_metadata": json.dumps(metadata, ensure_ascii=False, sort_keys=True),
    }


def _tracking_metadata_dict(hand: TrackedHandObservation, rendered: bool) -> dict:
    return {
        "source": str(hand.source),
        "confidence": float(hand.confidence),
        "gap_age": int(hand.gap_age),
        "quality_reason": str(hand.quality_reason),
        "original_detected": bool(hand.original_detected),
        "rendered": bool(rendered),
        "roi": list(hand.roi) if hand.roi is not None else None,
    }


def _frame_to_gray(frame_bgr):
    if cv2 is None or frame_bgr is None:
        return None
    try:
        if frame_bgr.ndim == 2:
            return frame_bgr.copy()
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    except Exception:
        return None


def _landmarks_roi(points: np.ndarray) -> tuple[float, float, float, float] | None:
    pts = np.asarray(points, dtype=np.float32).reshape(21, 3)
    if not np.isfinite(pts).all() or np.allclose(pts, 0.0):
        return None
    lo = np.min(pts[:, :2], axis=0)
    hi = np.max(pts[:, :2], axis=0)
    return (float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1]))


def _raw_hand_total_length(points: np.ndarray) -> float:
    pts = np.asarray(points, dtype=np.float32).reshape(21, 3)
    total = 0.0
    for parent, child in HAND_TREE_EDGES:
        total += float(np.linalg.norm(pts[child, :2] - pts[parent, :2]))
    return total


def _plausible_raw_hand(points: np.ndarray, cfg: PreprocessConfig) -> bool:
    pts = np.asarray(points, dtype=np.float32).reshape(21, 3)
    if not np.isfinite(pts).all() or np.allclose(pts, 0.0):
        return False
    bbox = float(np.linalg.norm(np.ptp(pts[:, :2], axis=0)))
    total = _raw_hand_total_length(pts)
    return (
        cfg.tracker_min_raw_bbox <= bbox <= cfg.tracker_max_raw_bbox
        and cfg.tracker_min_raw_total_length <= total <= cfg.tracker_max_raw_total_length
    )


def _parse_hands_candidates(hands_results, cfg: PreprocessConfig) -> list[dict]:
    if hands_results is None or not getattr(hands_results, "multi_hand_landmarks", None):
        return []

    handedness = getattr(hands_results, "multi_handedness", None) or []
    candidates: list[dict] = []
    for i, landmarks in enumerate(hands_results.multi_hand_landmarks):
        pts = _landmarks_to_array(landmarks, 21)
        if not _plausible_raw_hand(pts, cfg):
            continue
        label = ""
        score = 0.0
        if i < len(handedness) and getattr(handedness[i], "classification", None):
            cls = handedness[i].classification[0]
            label = str(getattr(cls, "label", "")).lower()
            score = float(getattr(cls, "score", 0.0))
        candidates.append({"landmarks": pts, "label": label, "score": score})
    return candidates


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
    original_weight = np.asarray(original_valid, dtype=np.float32).reshape(-1)
    out = np.zeros_like(arr, dtype=np.float32)

    for start, end in _true_segments(segment_mask):
        weight_slice = original_weight[start:end]
        conf = np.where(
            weight_slice > 0.0,
            np.clip(weight_slice, float(imputed_weight), 1.0),
            float(imputed_weight),
        ).astype(np.float32)
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
