"""
feature_engine.py - Hybrid Feature Extraction (179-D)
=====================================================

Output contract remains unchanged:
  [0:18]    Pose spatial (6 landmarks x 3)
  [18:81]   Left hand spatial (21 landmarks x 3, wrist-relative)
  [81:144]  Right hand spatial (21 landmarks x 3, wrist-relative)
  [144:160] Left hand joint angles (16)
  [160:176] Right hand joint angles (16)
  [176:179] Original detection flags [pose_ok, lh_ok, rh_ok]

The important production change is the temporal preprocessor. Missing
landmarks are filled before smoothing, but the smoother is now a positive
kernel instead of Savitzky-Golay. A positive kernel is a convex combination:

    y_t = sum_i w_i x_i / sum_i w_i,  w_i >= 0

That means each coordinate stays inside the local coordinate envelope. SG
polynomial filters can have negative effective weights, so near padded or
interpolated boundaries they can overshoot and create the "mleyot" hand
stretching artifact. This module also clips features to conservative
shoulder-normalized anatomical bounds after fill and after smoothing.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np

# ---------------------------------------------------------------------------
# Feature dimensions
# ---------------------------------------------------------------------------
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

HAND_KINEMATIC_CHAINS = [
    (0, 1, 2), (1, 2, 3), (2, 3, 4),
    (0, 5, 6), (5, 6, 7), (6, 7, 8),
    (0, 9, 10), (9, 10, 11), (10, 11, 12),
    (0, 13, 14), (13, 14, 15), (14, 15, 16),
    (0, 17, 18), (17, 18, 19), (18, 19, 20),
    (5, 0, 9),
]

_PART_SLICES = {
    IDX_POSE: (SLICE_POSE, None),
    IDX_LH: (SLICE_LH, SLICE_LH_A),
    IDX_RH: (SLICE_RH, SLICE_RH_A),
}


@dataclass(frozen=True)
class PreprocessConfig:
    """Versioned preprocessing controls shared by ingestion and live inference."""

    version: str = "bounded_positive_kernel_v2"
    smooth_kernel: tuple[float, ...] = (1.0, 2.0, 3.0, 2.0, 1.0)
    imputed_smooth_weight: float = 0.35
    pose_abs_limit: float = 4.0
    hand_abs_limit: float = 2.0
    hand_radius_limit: float = 1.65
    robust_quantile: float = 0.01
    robust_margin_ratio: float = 0.25
    robust_min_margin: float = 0.05
    min_valid_for_envelope: int = 3
    min_shoulder_width: float = 0.02


DEFAULT_PREPROCESS_CONFIG = PreprocessConfig()


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def joint_angle_xy(A: np.ndarray, B: np.ndarray, C: np.ndarray) -> float:
    """2-D joint angle at B between rays B->A and B->C, in radians."""
    BA = A[:2] - B[:2]
    BC = C[:2] - B[:2]
    cross_z = BA[0] * BC[1] - BA[1] * BC[0]
    dot = float(np.dot(BA, BC))
    return float(np.arctan2(abs(cross_z), dot))


def extract_hand_angles(landmarks) -> list[float]:
    """Return 16 hand joint angles, or zeros when the hand is absent."""
    if not landmarks:
        return [0.0] * N_ANGLES
    pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark], dtype=np.float32)
    return [joint_angle_xy(pts[a], pts[b], pts[c]) for a, b, c in HAND_KINEMATIC_CHAINS]


def _norm_hand_landmarks(hand_lms, shoulder_width: float) -> list[float]:
    """Normalize 21 hand landmarks relative to the wrist and shoulder width."""
    w = hand_lms.landmark[0]
    wrist = np.array([w.x, w.y, w.z], dtype=np.float32)
    denom = max(float(shoulder_width), DEFAULT_PREPROCESS_CONFIG.min_shoulder_width)
    coords: list[float] = []
    for lm in hand_lms.landmark:
        coords.extend([
            (lm.x - wrist[0]) / denom,
            (lm.y - wrist[1]) / denom,
            (lm.z - wrist[2]) / denom,
        ])
    return coords


def extract_keypoints_relative(results):
    """
    Extract a 176-D feature vector and the 3 original detection flags from a
    MediaPipe Holistic result.

    Returns
    -------
    vector  : np.ndarray shape (176,)
    mask    : np.ndarray shape (3,), bool [pose_ok, lh_ok, rh_ok]
    pose_lw : np.ndarray shape (3,), left wrist relative to shoulders
    pose_rw : np.ndarray shape (3,), right wrist relative to shoulders
    """
    pose_coords: list[float] = []
    lh_coords: list[float] = []
    rh_coords: list[float] = []

    p_ok = lh_ok = rh_ok = False
    shoulder_width = 1.0
    mid_shoulder = np.zeros(3, dtype=np.float32)
    pose_lw = np.zeros(3, dtype=np.float32)
    pose_rw = np.zeros(3, dtype=np.float32)

    if results.pose_landmarks:
        p_ok = True
        lm = results.pose_landmarks.landmark

        p_ls = np.array([lm[11].x, lm[11].y, lm[11].z], dtype=np.float32)
        p_rs = np.array([lm[12].x, lm[12].y, lm[12].z], dtype=np.float32)
        mid_shoulder = (p_ls + p_rs) * 0.5

        dist = float(np.linalg.norm(p_ls - p_rs))
        if dist >= DEFAULT_PREPROCESS_CONFIG.min_shoulder_width:
            shoulder_width = dist

        denom = max(shoulder_width, DEFAULT_PREPROCESS_CONFIG.min_shoulder_width)
        pose_lw = (np.array([lm[15].x, lm[15].y, lm[15].z]) - mid_shoulder) / denom
        pose_rw = (np.array([lm[16].x, lm[16].y, lm[16].z]) - mid_shoulder) / denom

        for idx in range(11, 17):
            pose_coords.extend([
                (lm[idx].x - mid_shoulder[0]) / denom,
                (lm[idx].y - mid_shoulder[1]) / denom,
                (lm[idx].z - mid_shoulder[2]) / denom,
            ])
    else:
        pose_coords = [0.0] * N_POSE

    if results.left_hand_landmarks:
        lh_ok = True
        lh_coords = _norm_hand_landmarks(results.left_hand_landmarks, shoulder_width)
    else:
        lh_coords = [0.0] * N_HAND

    if results.right_hand_landmarks:
        rh_ok = True
        rh_coords = _norm_hand_landmarks(results.right_hand_landmarks, shoulder_width)
    else:
        rh_coords = [0.0] * N_HAND

    lh_angles = extract_hand_angles(results.left_hand_landmarks)
    rh_angles = extract_hand_angles(results.right_hand_landmarks)

    vector = np.array(pose_coords + lh_coords + rh_coords + lh_angles + rh_angles, dtype=np.float32)
    vector = np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)
    mask = np.array([p_ok, lh_ok, rh_ok], dtype=bool)
    return vector, mask, pose_lw.astype(np.float32), pose_rw.astype(np.float32)


def calculate_movement_score(prev_vector, curr_vector) -> float:
    """Movement score from spatial features only."""
    if prev_vector is None or curr_vector is None:
        return 0.0
    prev_arr = np.asarray(prev_vector, dtype=np.float32)
    curr_arr = np.asarray(curr_vector, dtype=np.float32)
    return float(np.linalg.norm(curr_arr[:N_TOTAL_SPATIAL] - prev_arr[:N_TOTAL_SPATIAL]))


# ---------------------------------------------------------------------------
# Public sanitizer used by augmentation and tests
# ---------------------------------------------------------------------------
def sanitize_sequence(
    sequence: np.ndarray,
    masks: np.ndarray | None = None,
    config: PreprocessConfig | None = None,
    preserve_flags: bool = True,
) -> np.ndarray:
    """
    Sanitize a 176-D or 179-D sequence without changing its rank.

    This is intentionally conservative and side-effect free. It does not fill
    gaps; SequenceBuilder owns temporal gap filling. Augmentation uses this to
    keep synthetic features finite, bounded, and flag-safe.
    """
    cfg = config or DEFAULT_PREPROCESS_CONFIG
    arr = np.asarray(sequence, dtype=np.float32)
    if arr.ndim == 1:
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

    features = _clip_anatomy(features, masks, cfg)

    if flags is None:
        return features.astype(np.float32)

    if preserve_flags:
        flags = (flags >= 0.5).astype(np.float32)
    return np.concatenate([features, flags], axis=1).astype(np.float32)


# ---------------------------------------------------------------------------
# SequenceBuilder
# ---------------------------------------------------------------------------
class SequenceBuilder:
    """
    Collect raw per-frame features and build a bounded 179-D sequence.

    Processing order:
      1. finite sanitize
      2. per-part bounded linear gap fill
      3. anatomy/range clipping
      4. mask-aware positive-kernel smoothing
      5. anatomy/range clipping again
      6. append original unsmoothed detection flags
    """

    def __init__(
        self,
        smooth_window: int | None = None,
        smooth_poly: int | None = None,
        config: PreprocessConfig | None = None,
    ):
        self.config = config or DEFAULT_PREPROCESS_CONFIG
        # Compatibility with the old constructor. smooth_poly is intentionally
        # ignored because polynomial smoothing is the source of the overshoot.
        if smooth_window is not None and smooth_window >= 3:
            self.smooth_kernel = _triangular_kernel(smooth_window)
        else:
            self.smooth_kernel = np.asarray(self.config.smooth_kernel, dtype=np.float32)

        self._vectors: list[np.ndarray] = []
        self._masks: list[np.ndarray] = []
        self._pose_lw: list[np.ndarray | None] = []
        self._pose_rw: list[np.ndarray | None] = []

    def add_frame(self, vector, mask=None, pose_lw=None, pose_rw=None):
        arr = np.asarray(vector, dtype=np.float32)
        if arr.ndim != 1:
            arr = arr.reshape(-1)

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

        self._vectors.append(np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32))
        self._masks.append(mask_arr)
        self._pose_lw.append(pose_lw)
        self._pose_rw.append(pose_rw)

    def reset(self):
        self._vectors.clear()
        self._masks.clear()
        self._pose_lw.clear()
        self._pose_rw.clear()

    def build(self):
        """
        Returns
        -------
        sequence : list[np.ndarray], each shape (179,)
        scores   : list[float], spatial movement score per frame
        """
        if not self._vectors:
            return [], []

        seq = np.stack(self._vectors).astype(np.float32)
        msk = np.stack(self._masks).astype(bool)
        flags = msk.astype(np.float32)

        seq = np.nan_to_num(seq, nan=0.0, posinf=0.0, neginf=0.0)
        seq = self._fill_all_gaps(seq, msk)
        seq = _clip_anatomy(seq, msk, self.config)
        seq = self._smooth(seq, msk)
        seq = _clip_anatomy(seq, msk, self.config)

        augmented = np.concatenate([seq, flags], axis=1).astype(np.float32)
        scores = self._compute_scores(seq)
        return [augmented[i] for i in range(len(augmented))], scores

    def _fill_all_gaps(self, seq: np.ndarray, msk: np.ndarray) -> np.ndarray:
        """Fill missing body-part slices with bounded C0 linear interpolation."""
        result = seq.copy()

        for part_idx in range(N_FLAGS):
            spatial_sl, angle_sl = _PART_SLICES[part_idx]
            detected = msk[:, part_idx]
            valid_frames = np.where(detected)[0]

            if valid_frames.size == 0:
                result[:, spatial_sl] = 0.0
                if angle_sl is not None:
                    result[:, angle_sl] = 0.0
                continue

            gap_frames = np.where(~detected)[0]
            for t_raw in gap_frames:
                t = int(t_raw)
                pos = int(np.searchsorted(valid_frames, t))

                if pos == 0:
                    src = int(valid_frames[0])
                    _copy_part(result, t, src, spatial_sl, angle_sl)
                elif pos == valid_frames.size:
                    src = int(valid_frames[-1])
                    _copy_part(result, t, src, spatial_sl, angle_sl)
                else:
                    t0 = int(valid_frames[pos - 1])
                    t1 = int(valid_frames[pos])
                    alpha = (t - t0) / float(t1 - t0)
                    _interp_part(result, t, t0, t1, alpha, spatial_sl, angle_sl)

        return result.astype(np.float32)

    def _smooth(self, seq: np.ndarray, msk: np.ndarray) -> np.ndarray:
        """Mask-aware positive-kernel smoothing. No negative weights, no ringing."""
        T = seq.shape[0]
        if T < 3:
            return seq.astype(np.float32)

        kernel = np.asarray(self.smooth_kernel, dtype=np.float32)
        if kernel.size < 3:
            return seq.astype(np.float32)
        if kernel.size % 2 == 0:
            kernel = kernel[:-1]
        kernel = np.maximum(kernel, 0.0)
        if float(kernel.sum()) <= 0.0:
            return seq.astype(np.float32)

        half = kernel.size // 2
        out = seq.copy()

        for part_idx in range(N_FLAGS):
            spatial_sl, angle_sl = _PART_SLICES[part_idx]
            detected = msk[:, part_idx]
            if not detected.any():
                continue

            confidence = np.where(detected, 1.0, self.config.imputed_smooth_weight).astype(np.float32)
            slices = [spatial_sl] if angle_sl is None else [spatial_sl, angle_sl]

            for sl in slices:
                values = seq[:, sl]
                for t in range(T):
                    lo = max(0, t - half)
                    hi = min(T, t + half + 1)
                    k_lo = half - (t - lo)
                    k_hi = k_lo + (hi - lo)
                    weights = kernel[k_lo:k_hi] * confidence[lo:hi]
                    denom = float(weights.sum())
                    if denom > 1e-8:
                        out[t, sl] = (values[lo:hi] * weights[:, None]).sum(axis=0) / denom

        return out.astype(np.float32)

    @staticmethod
    def _compute_scores(seq: np.ndarray) -> list[float]:
        scores = [0.0]
        for i in range(1, len(seq)):
            scores.append(float(np.linalg.norm(seq[i, :N_TOTAL_SPATIAL] - seq[i - 1, :N_TOTAL_SPATIAL])))
        return scores


# ---------------------------------------------------------------------------
# Internal bounded math
# ---------------------------------------------------------------------------
def _triangular_kernel(window: int) -> np.ndarray:
    win = int(max(3, window))
    if win % 2 == 0:
        win -= 1
    half = win // 2
    values = np.array([half + 1 - abs(i - half) for i in range(win)], dtype=np.float32)
    return values / values.sum()


def _copy_part(result: np.ndarray, dst: int, src: int, spatial_sl: slice, angle_sl: slice | None):
    result[dst, spatial_sl] = result[src, spatial_sl]
    if angle_sl is not None:
        result[dst, angle_sl] = result[src, angle_sl]


def _interp_part(
    result: np.ndarray,
    dst: int,
    left: int,
    right: int,
    alpha: float,
    spatial_sl: slice,
    angle_sl: slice | None,
):
    beta = 1.0 - alpha
    result[dst, spatial_sl] = beta * result[left, spatial_sl] + alpha * result[right, spatial_sl]
    if angle_sl is not None:
        result[dst, angle_sl] = beta * result[left, angle_sl] + alpha * result[right, angle_sl]


def _clip_anatomy(seq: np.ndarray, masks: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    out = np.nan_to_num(seq.copy(), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    out[:, SLICE_POSE] = np.clip(out[:, SLICE_POSE], -cfg.pose_abs_limit, cfg.pose_abs_limit)
    out[:, SLICE_LH] = _clip_hand_block(out[:, SLICE_LH], masks[:, IDX_LH], cfg)
    out[:, SLICE_RH] = _clip_hand_block(out[:, SLICE_RH], masks[:, IDX_RH], cfg)
    out[:, SLICE_LH_A] = np.clip(out[:, SLICE_LH_A], 0.0, math.pi)
    out[:, SLICE_RH_A] = np.clip(out[:, SLICE_RH_A], 0.0, math.pi)
    return out


def _clip_hand_block(hand_flat: np.ndarray, detected: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    hand = np.clip(hand_flat.copy(), -cfg.hand_abs_limit, cfg.hand_abs_limit).reshape(-1, 21, 3)

    if detected.any() and int(detected.sum()) >= cfg.min_valid_for_envelope:
        valid = hand[detected].reshape(-1, N_HAND)
        q = float(np.clip(cfg.robust_quantile, 0.0, 0.2))
        lo = np.quantile(valid, q, axis=0)
        hi = np.quantile(valid, 1.0 - q, axis=0)
        span = np.maximum(hi - lo, 0.0)
        margin = np.maximum(span * cfg.robust_margin_ratio, cfg.robust_min_margin)
        low = np.maximum(lo - margin, -cfg.hand_abs_limit)
        high = np.minimum(hi + margin, cfg.hand_abs_limit)
        hand = np.clip(hand.reshape(-1, N_HAND), low, high).reshape(-1, 21, 3)

    # Hand coordinates are wrist-relative. Keep landmark 0 anchored and bound
    # all fingertips by radial distance from the wrist.
    hand[:, 0, :] = 0.0
    norms = np.linalg.norm(hand, axis=2, keepdims=True)
    scale = np.minimum(1.0, cfg.hand_radius_limit / np.maximum(norms, 1e-6))
    hand = hand * scale
    hand[:, 0, :] = 0.0
    return hand.reshape(-1, N_HAND).astype(np.float32)


def ensure_feature_dim(sequence: Iterable[np.ndarray], expected_dim: int = N_TOTAL_WITH_FLAGS) -> np.ndarray:
    """Small helper for callers/tests that need a strict feature matrix."""
    arr = np.asarray(list(sequence), dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != expected_dim:
        raise ValueError(f"Expected sequence shape (T, {expected_dim}), got {arr.shape}")
    return arr
