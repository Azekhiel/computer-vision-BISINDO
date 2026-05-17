"""
feature_engine.py - BISINDO V4.2 arm-gated tracklet-visible body-frame preprocessing.

Public output remains 179-D:
  0:144    spatial coordinates in absolute body-frame coordinates
            pose(6x3), left hand(21x3), right hand(21x3)
  144:176  hand joint angles
  176:179  accepted visibility flags [pose_ok, left_hand_visible, right_hand_visible]

V4.2 keeps the rigid kinematic reconstruction, but changes the semantics of the
public flags: hands are visible only when a confirmed detector/tracklet segment
is accepted by the tracker or by short in-segment interpolation. Original
Holistic/fallback provenance is stored in metadata, not in the public tensor.
Prediction may guide ROI search but must never become model-visible geometry.
Near-identical two-hand hypotheses are resolved before features are emitted.
Pose elbow/wrist endpoints are gated by accepted hand visibility so a pose wrist
cannot create fake non-signing arm motion in the model tensor or preview.
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

FEATURE_SCHEMA = "bisindo_v4_2_tracklet_arm_gated"
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
    anchor_speed_min: float = 0.18
    anchor_speed_max: float = 0.40
    anchor_speed_mad_scale: float = 4.0
    hand_skeleton_step_max: float = 2.25
    pose_abs_limit: float = 4.0
    hand_abs_limit: float = 8.0
    angle_min: float = 0.0
    angle_max: float = math.pi
    frame_enhance_enabled: bool = True
    frame_enhance_clahe_clip_limit: float = 2.0
    frame_enhance_tile_grid_size: int = 8
    frame_enhance_target_luma: float = 112.0
    frame_enhance_gamma_min: float = 0.70
    frame_enhance_gamma_max: float = 1.35
    frame_enhance_saturation_gain: float = 1.08
    tracker_max_flow_gap: int = 3
    tracker_max_prediction_gap: int = 0
    tracker_flow_min_success_ratio: float = 0.55
    tracker_flow_max_median_error: float = 10.0
    tracker_duplicate_anchor_radius: float = 0.18
    tracker_max_assignment_cost: float = 0.22
    tracker_max_bootstrap_cost: float = 0.18
    tracker_side_margin: float = 0.035
    tracker_min_confirmed_hits: int = 2
    tracker_lost_after: int = 8
    tracker_min_raw_bbox: float = 0.015
    tracker_max_raw_bbox: float = 1.35
    tracker_min_raw_total_length: float = 0.08
    tracker_max_raw_total_length: float = 7.5
    tracker_holistic_confidence: float = 1.0
    tracker_hands_confidence: float = 0.82
    tracker_flow_confidence: float = 0.82
    tracker_prediction_confidence: float = 0.25
    hand_conflict_mean_landmark_radius: float = 0.075
    hand_conflict_extreme_roi_iou: float = 0.78
    hand_conflict_extreme_anchor_radius: float = 0.10
    hand_min_accepted_segment: int = 3
    offline_assignment_max_cost: float = 0.58
    offline_pose_prior_weight: float = 0.38
    offline_velocity_weight: float = 0.90
    offline_handedness_bonus: float = 0.10
    offline_min_track_hits: int = 2
    offline_max_track_gap: int = 6
    offline_contact_impute_enabled: bool = True
    offline_contact_impute_max_anchor_distance: float = 0.18
    offline_contact_impute_max_wrist_separation: float = 0.24
    offline_contact_impute_confidence: float = 0.86
    pose_arm_max_elbow_ratio: float = 1.95


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
    accepted: bool = False
    track_id: int = -1


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
    candidates: list[dict] | None = None


@dataclass
class _HandTrackState:
    landmarks: np.ndarray | None = None
    prev_gray: np.ndarray | None = None
    velocity_xy: np.ndarray | None = None
    gap_age: int = 999
    source: str = "missing"
    track_id: int = -1
    hits: int = 0
    status: str = "lost"


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
        accepted=False,
        track_id=-1,
    )


def _tracked_hand_from_landmarks(
    landmarks: np.ndarray,
    source: str,
    confidence: float,
    quality_reason: str,
    gap_age: int,
    original_detected: bool,
    config: PreprocessConfig | None = None,
    accepted: bool | None = None,
    track_id: int = -1,
) -> TrackedHandObservation:
    cfg = config or DEFAULT_PREPROCESS_CONFIG
    pts = np.nan_to_num(np.asarray(landmarks, dtype=np.float32).reshape(21, 3), nan=0.0, posinf=0.0, neginf=0.0)
    plausible = _plausible_raw_hand(pts, cfg)
    if not plausible:
        return _empty_tracked_hand(original_detected=original_detected, reason=f"invalid_{source}")
    is_accepted = plausible if accepted is None else bool(accepted)
    return TrackedHandObservation(
        landmarks_raw=pts.astype(np.float32),
        valid=is_accepted,
        source=source,
        confidence=float(np.clip(confidence, 0.0, 1.0)),
        quality_reason=quality_reason,
        roi=_landmarks_roi(pts),
        gap_age=int(gap_age),
        original_detected=bool(original_detected),
        accepted=is_accepted,
        track_id=int(track_id),
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
        candidates=None,
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

    The returned mask stores original detector availability. The public model
    flags are created later by SequenceBuilder from accepted tracklet visibility.
    Fallback Hands and LK flow can support a confirmed tracklet, but prediction
    never becomes model-visible geometry.
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
    Conservative hand tracklet recovery for detector drops.

    Holistic detections are accepted immediately. Fallback Hands can confirm or
    bootstrap a tracklet, but a new fallback-only track is tentative until it is
    observed repeatedly. LK optical flow bridges only confirmed tracks. Constant
    velocity prediction is retained only as an internal ROI prior and is never
    returned as model-visible geometry.
    """

    def __init__(self, config: PreprocessConfig | None = None):
        self.config = config or DEFAULT_PREPROCESS_CONFIG
        self._state = {"left": _HandTrackState(), "right": _HandTrackState()}
        self._next_track_id = 1

    def reset(self):
        self._state = {"left": _HandTrackState(), "right": _HandTrackState()}
        self._next_track_id = 1

    def needs_fallback(self, observation: FrameObservation) -> bool:
        left_missing = not bool(observation.mask[IDX_LH])
        right_missing = not bool(observation.mask[IDX_RH])
        left_bad = bool(observation.mask[IDX_LH]) and not _plausible_raw_hand(observation.left_hand_raw, self.config)
        right_bad = bool(observation.mask[IDX_RH]) and not _plausible_raw_hand(observation.right_hand_raw, self.config)
        return left_missing or right_missing or left_bad or right_bad

    def fallback_candidates(self, frame_rgb, hands_solution, observation: FrameObservation) -> list[dict]:
        if hands_solution is None or frame_rgb is None or not self.needs_fallback(observation):
            return []
        if cv2 is None:
            try:
                return _parse_hands_candidates(hands_solution.process(frame_rgb), self.config)
            except Exception:
                return []

        h, w = frame_rgb.shape[:2]
        candidates: list[dict] = []
        sides = ("left", "right")
        shoulder_px = max(float(observation.shoulder_scale_raw) * max(w, h), 80.0)

        for side in sides:
            idx = IDX_LH if side == "left" else IDX_RH
            raw = observation.left_hand_raw if side == "left" else observation.right_hand_raw
            if bool(observation.mask[idx]) and _plausible_raw_hand(raw, self.config):
                continue

            center = self._target_anchor(side, observation)
            cx = int(np.clip(center[0], 0.0, 1.0) * w)
            cy = int(np.clip(center[1], 0.0, 1.0) * h)
            crop_size = int(np.clip(shoulder_px * 3.2, 128, max(w, h)))
            half = crop_size // 2
            x0 = max(0, cx - half)
            y0 = max(0, cy - half)
            x1 = min(w, cx + half)
            y1 = min(h, cy + half)
            if x1 - x0 < 64 or y1 - y0 < 64:
                continue

            crop = frame_rgb[y0:y1, x0:x1]
            try:
                crop_results = hands_solution.process(crop)
            except Exception:
                continue
            for candidate in _parse_hands_candidates(crop_results, self.config):
                pts = candidate["landmarks"].copy()
                pts[:, 0] = (x0 + pts[:, 0] * (x1 - x0)) / max(float(w), 1.0)
                pts[:, 1] = (y0 + pts[:, 1] * (y1 - y0)) / max(float(h), 1.0)
                candidate = dict(candidate)
                candidate["landmarks"] = pts.astype(np.float32)
                candidate["roi_source"] = f"{side}_roi"
                candidates.append(candidate)

        if not candidates and all(self._state[side].status == "lost" for side in sides):
            try:
                candidates = _parse_hands_candidates(hands_solution.process(frame_rgb), self.config)
                for candidate in candidates:
                    candidate["roi_source"] = "full_frame_bootstrap"
            except Exception:
                candidates = []
        return candidates

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
            candidates=fallback_candidates,
        )

    def _assign_fallback_candidates(self, candidates: list[dict], observation: FrameObservation) -> dict[str, dict]:
        assigned: dict[str, dict] = {}
        if not candidates:
            return assigned

        used: set[int] = set()
        side_order = ["left", "right"]
        side_order.sort(key=lambda name: 0 if self._state[name].status == "confirmed" else 1)
        for side in side_order:
            idx = IDX_LH if side == "left" else IDX_RH
            raw = observation.left_hand_raw if side == "left" else observation.right_hand_raw
            if bool(observation.mask[idx]) and _plausible_raw_hand(raw, self.config):
                continue

            best_i = None
            best_cost = float("inf")
            for i, candidate in enumerate(candidates):
                if i in used:
                    continue
                cost = self._candidate_assignment_cost(side, candidate, observation)
                if cost < best_cost:
                    best_cost = cost
                    best_i = i

            if best_i is not None:
                state = self._state[side]
                bootstrapping = state.status == "lost" or state.landmarks is None
                max_cost = self.config.tracker_max_bootstrap_cost if bootstrapping else self.config.tracker_max_assignment_cost
                if best_cost <= float(max_cost):
                    used.add(best_i)
                    assigned[side] = candidates[best_i]
        return assigned

    def _candidate_assignment_cost(self, side: str, candidate: dict, observation: FrameObservation) -> float:
        pts = candidate["landmarks"]
        anchor = np.asarray(pts[0, :2], dtype=np.float32)
        target = self._target_anchor(side, observation)
        cost = float(np.linalg.norm(anchor - target))

        other_side = "right" if side == "left" else "left"
        other_idx = IDX_RH if other_side == "right" else IDX_LH
        other_raw = observation.right_hand_raw if other_side == "right" else observation.left_hand_raw
        if bool(observation.mask[other_idx]) and _plausible_raw_hand(other_raw, self.config):
            other_dist = float(np.linalg.norm(anchor - other_raw[0, :2]))
            if other_dist < float(self.config.tracker_duplicate_anchor_radius):
                return float("inf")

        state = self._state[side]
        bootstrapping = state.status == "lost" or state.landmarks is None
        if bootstrapping and not bool(observation.mask[IDX_POSE]):
            return float("inf")
        if bootstrapping and bool(observation.mask[IDX_POSE]):
            side_anchor = self._pose_wrist_anchor(side, observation)
            other_anchor = self._pose_wrist_anchor(other_side, observation)
            side_dist = float(np.linalg.norm(anchor - side_anchor))
            other_dist = float(np.linalg.norm(anchor - other_anchor))
            if side_dist + float(self.config.tracker_side_margin) >= other_dist:
                return float("inf")
            cost = side_dist

        label = str(candidate.get("label", "")).lower()
        if label == side:
            cost -= 0.06
        elif label in ("left", "right"):
            cost += 0.04
        cost -= 0.02 * float(candidate.get("score", 0.0))
        return max(0.0, float(cost))

    def _target_anchor(self, side: str, observation: FrameObservation) -> np.ndarray:
        state = self._state[side]
        if state.landmarks is not None and state.status != "lost" and state.gap_age <= self.config.tracker_lost_after:
            return np.clip(state.landmarks[0, :2], 0.0, 1.0).astype(np.float32)
        return self._pose_wrist_anchor(side, observation)

    def _pose_wrist_anchor(self, side: str, observation: FrameObservation) -> np.ndarray:
        if bool(observation.mask[IDX_POSE]):
            wrist_idx = 4 if side == "left" else 5
            return np.clip(observation.pose_raw[wrist_idx, :2], 0.0, 1.0).astype(np.float32)
        return np.array([0.5, 0.5], dtype=np.float32)

    def _ensure_track_id(self, side: str) -> int:
        state = self._state[side]
        if state.track_id < 0:
            state.track_id = int(self._next_track_id)
            self._next_track_id += 1
        return int(state.track_id)

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
            track_id = self._ensure_track_id(side)
            tracked = _tracked_hand_from_landmarks(
                raw,
                "holistic",
                cfg.tracker_holistic_confidence,
                "holistic_detected",
                0,
                True,
                cfg,
                accepted=True,
                track_id=track_id,
            )
            self._accept_state(side, tracked, gray, reset_gap=True, detector_hit=True)
            return tracked

        fallback_landmarks = fallback_landmarks.get("landmarks") if isinstance(fallback_landmarks, dict) else fallback_landmarks
        if fallback_landmarks is not None and _plausible_raw_hand(fallback_landmarks, cfg):
            track_id = self._ensure_track_id(side)
            next_hits = (state.hits + 1) if state.status != "lost" else 1
            accepted = state.status == "confirmed" or next_hits >= int(cfg.tracker_min_confirmed_hits)
            tracked = _tracked_hand_from_landmarks(
                fallback_landmarks,
                "hands" if accepted else "hands_tentative",
                cfg.tracker_hands_confidence if accepted else 0.0,
                "fallback_hands_detected" if accepted else "fallback_hands_tentative",
                0,
                bool(observation.mask[idx]),
                cfg,
                accepted=accepted,
                track_id=track_id,
            )
            self._accept_state(side, tracked, gray, reset_gap=True, detector_hit=True)
            return tracked

        flow = self._flow_recover(state, gray, cfg)
        if flow is not None:
            gap = min(int(state.gap_age) + 1, 999)
            conf = max(0.35, cfg.tracker_flow_confidence - 0.10 * max(gap - 1, 0))
            tracked = _tracked_hand_from_landmarks(
                flow,
                "flow",
                conf,
                "lk_recovered",
                gap,
                bool(observation.mask[idx]),
                cfg,
                accepted=True,
                track_id=state.track_id,
            )
            self._accept_state(side, tracked, gray, reset_gap=False, detector_hit=False)
            return tracked

        self._mark_missing(side)
        return _empty_tracked_hand(original_detected=bool(observation.mask[idx]), reason="detector_missing")

    def _flow_recover(
        self,
        state: _HandTrackState,
        gray: np.ndarray | None,
        cfg: PreprocessConfig,
    ) -> np.ndarray | None:
        if cv2 is None or gray is None or state.landmarks is None or state.prev_gray is None:
            return None
        if state.status != "confirmed":
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

    def _accept_state(
        self,
        side: str,
        tracked: TrackedHandObservation,
        gray: np.ndarray | None,
        reset_gap: bool,
        detector_hit: bool,
    ):
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
        state.track_id = int(tracked.track_id)
        if detector_hit:
            state.hits = min(int(state.hits) + 1, int(self.config.tracker_min_confirmed_hits))
        if tracked.accepted or state.hits >= int(self.config.tracker_min_confirmed_hits):
            state.status = "confirmed"
        elif detector_hit:
            state.status = "tentative"

    def _mark_missing(self, side: str):
        state = self._state[side]
        state.gap_age = min(int(state.gap_age) + 1, 999)
        if state.gap_age > int(self.config.tracker_lost_after):
            state.landmarks = None
            state.prev_gray = None
            state.velocity_xy = None
            state.track_id = -1
            state.hits = 0
            state.status = "lost"
            state.source = "missing"


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


@dataclass
class _OfflineSideState:
    landmarks: np.ndarray | None = None
    velocity: np.ndarray | None = None
    prev_gray: np.ndarray | None = None
    track_id: int = -1
    hits: int = 0
    gap: int = 999


class OfflineTrackletSolver:
    """
    Quality-first hand assignment for dataset extraction.

    The online tracker remains causal and lightweight. This solver is used by
    offline ingestion where we can spend more CPU to compare all Holistic and
    standalone Hands candidates before emitting the public 179-D tensor.
    """

    def __init__(self, config: PreprocessConfig | None = None):
        self.config = config or DEFAULT_PREPROCESS_CONFIG
        self._states = {"left": _OfflineSideState(track_id=101), "right": _OfflineSideState(track_id=201)}

    def solve(
        self,
        observations: list[FrameObservation],
        candidate_frames: list[list[dict]] | None = None,
        gray_frames: list[np.ndarray | None] | None = None,
    ) -> list[TrackedFrameObservation]:
        if candidate_frames is None:
            candidate_frames = [[] for _ in observations]
        if gray_frames is None:
            gray_frames = [None for _ in observations]
        tracked: list[TrackedFrameObservation] = []
        for frame_idx, observation in enumerate(observations):
            gray = gray_frames[frame_idx] if frame_idx < len(gray_frames) else None
            candidates = self._frame_candidates(observation, candidate_frames[frame_idx], frame_idx, gray)
            assigned = self._assign_frame_candidates(observation, candidates)
            left = self._make_side_observation("left", observation, assigned.get("left"), gray)
            right = self._make_side_observation("right", observation, assigned.get("right"), gray)
            tracked.append(
                TrackedFrameObservation(
                    pose_raw=observation.pose_raw,
                    left_hand_raw=left.landmarks_raw,
                    right_hand_raw=right.landmarks_raw,
                    mask=observation.mask.copy(),
                    body_origin_raw=observation.body_origin_raw,
                    shoulder_scale_raw=float(observation.shoulder_scale_raw),
                    left_tracking=left,
                    right_tracking=right,
                    candidates=_candidate_metadata_list(candidates),
                )
            )
        return tracked

    def _frame_candidates(
        self,
        observation: FrameObservation,
        external_candidates: list[dict],
        frame_idx: int,
        gray: np.ndarray | None = None,
    ) -> list[dict]:
        candidates: list[dict] = []
        if bool(observation.mask[IDX_LH]) and _plausible_raw_hand(observation.left_hand_raw, self.config):
            candidates.append(
                _candidate_from_points(
                    observation.left_hand_raw,
                    "holistic",
                    "left",
                    self.config.tracker_holistic_confidence,
                    frame_idx,
                    original_detected=True,
                    variant="holistic",
                )
            )
        if bool(observation.mask[IDX_RH]) and _plausible_raw_hand(observation.right_hand_raw, self.config):
            candidates.append(
                _candidate_from_points(
                    observation.right_hand_raw,
                    "holistic",
                    "right",
                    self.config.tracker_holistic_confidence,
                    frame_idx,
                    original_detected=True,
                    variant="holistic",
                )
            )
        for item in external_candidates or []:
            if not isinstance(item, dict) or "landmarks" not in item:
                continue
            pts = np.asarray(item["landmarks"], dtype=np.float32).reshape(21, 3)
            if not _plausible_raw_hand(pts, self.config):
                continue
            candidates.append(
                _candidate_from_points(
                    pts,
                    str(item.get("source", "hands")),
                    str(item.get("side_hint", item.get("label", ""))).lower(),
                    float(item.get("score", self.config.tracker_hands_confidence)),
                    frame_idx,
                    original_detected=bool(item.get("original_detected", False)),
                    variant=str(item.get("variant", item.get("roi_source", "hands"))),
                )
            )
        candidates.extend(self._flow_candidates(gray, frame_idx))
        return _dedupe_candidates(candidates, self.config)

    def _flow_candidates(self, gray: np.ndarray | None, frame_idx: int) -> list[dict]:
        if cv2 is None or gray is None:
            return []
        candidates: list[dict] = []
        for side, state in self._states.items():
            if state.landmarks is None or state.prev_gray is None:
                continue
            if state.hits < int(self.config.offline_min_track_hits):
                continue
            flow_gap = int(state.gap) + 1
            if flow_gap > int(self.config.tracker_max_flow_gap):
                continue

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
                continue

            if next_pts is None or status is None:
                continue
            ok = status.reshape(-1).astype(bool)
            if float(ok.mean()) < float(self.config.tracker_flow_min_success_ratio):
                continue
            if err is not None:
                good_err = err.reshape(-1)[ok]
                if good_err.size and float(np.median(good_err)) > float(self.config.tracker_flow_max_median_error):
                    continue

            next_xy = next_pts.reshape(-1, 2)
            pts = state.landmarks.copy()
            pts[ok, 0] = next_xy[ok, 0] / max(float(w), 1.0)
            pts[ok, 1] = next_xy[ok, 1] / max(float(h), 1.0)
            pts[:, :2] = np.clip(pts[:, :2], -0.20, 1.20)
            if not _plausible_raw_hand(pts, self.config):
                continue
            candidate = _candidate_from_points(
                pts,
                "flow",
                side,
                max(0.45, float(self.config.tracker_flow_confidence) - 0.08 * max(flow_gap - 1, 0)),
                frame_idx,
                original_detected=False,
                variant="offline_lk",
            )
            candidate["flow_gap"] = flow_gap
            candidates.append(candidate)
        return candidates

    def _assign_frame_candidates(self, observation: FrameObservation, candidates: list[dict]) -> dict[str, dict]:
        assignments: dict[str, dict] = {}
        used: set[int] = set()
        options: list[tuple[float, str, int]] = []
        for side in ("left", "right"):
            for idx, candidate in enumerate(candidates):
                cost = self._assignment_cost(side, candidate, observation)
                if np.isfinite(cost) and cost <= float(self.config.offline_assignment_max_cost):
                    options.append((cost, side, idx))
        options.sort(key=lambda item: item[0])
        for _, side, idx in options:
            if side in assignments or idx in used:
                continue
            other_side = "right" if side == "left" else "left"
            if other_side in assignments:
                other = assignments[other_side]
                if _mean_landmark_distance(other["landmarks"], candidates[idx]["landmarks"]) <= self.config.hand_conflict_mean_landmark_radius:
                    continue
            assignments[side] = candidates[idx]
            used.add(idx)
        self._add_contact_imputed_assignment(assignments, observation)
        return assignments

    def _add_contact_imputed_assignment(self, assignments: dict[str, dict], observation: FrameObservation):
        if not bool(self.config.offline_contact_impute_enabled):
            return
        for side in ("left", "right"):
            other_side = "right" if side == "left" else "left"
            if side in assignments or other_side not in assignments:
                continue
            side_wrist = _pose_wrist_anchor_raw(side, observation)
            other_wrist = _pose_wrist_anchor_raw(other_side, observation)
            if side_wrist is None or other_wrist is None:
                continue
            other = assignments[other_side]
            other_pts = np.asarray(other["landmarks"], dtype=np.float32).reshape(21, 3)
            anchor = other_pts[0, :2]
            anchor_dist = float(np.linalg.norm(side_wrist - anchor))
            wrist_sep = float(np.linalg.norm(side_wrist - other_wrist))
            if anchor_dist > float(self.config.offline_contact_impute_max_anchor_distance):
                continue
            if wrist_sep > float(self.config.offline_contact_impute_max_wrist_separation):
                continue
            pts = other_pts.copy()
            delta = side_wrist - pts[0, :2]
            pts[:, :2] += delta.reshape(1, 2)
            assignments[side] = _candidate_from_points(
                pts,
                "contact_imputed",
                side,
                self.config.offline_contact_impute_confidence,
                int(other.get("frame_idx", -1)),
                original_detected=False,
                variant=f"contact_from_{other_side}",
            )

    def _assignment_cost(self, side: str, candidate: dict, observation: FrameObservation) -> float:
        pts = np.asarray(candidate["landmarks"], dtype=np.float32).reshape(21, 3)
        anchor = pts[0, :2]
        state = self._states[side]
        if state.landmarks is not None and state.gap <= int(self.config.offline_max_track_gap):
            predicted = state.landmarks[0, :2]
            if state.velocity is not None:
                predicted = predicted + state.velocity[0]
            velocity_cost = float(np.linalg.norm(anchor - predicted))
        else:
            velocity_cost = 0.0

        pose_anchor = _pose_wrist_anchor_raw(side, observation)
        pose_cost = float(np.linalg.norm(anchor - pose_anchor)) if pose_anchor is not None else 0.35
        hint = str(candidate.get("side_hint", "")).lower()
        hint_bonus = float(self.config.offline_handedness_bonus) if hint == side else 0.0
        source_bonus = 0.08 if candidate.get("source") == "holistic" else 0.0
        confidence_bonus = 0.06 * float(candidate.get("score", 0.0))

        if state.landmarks is None or state.gap > int(self.config.offline_max_track_gap):
            cost = pose_cost
        else:
            cost = self.config.offline_velocity_weight * velocity_cost + self.config.offline_pose_prior_weight * pose_cost
        return float(max(0.0, cost - hint_bonus - source_bonus - confidence_bonus))

    def _make_side_observation(
        self,
        side: str,
        observation: FrameObservation,
        candidate: dict | None,
        gray: np.ndarray | None = None,
    ) -> TrackedHandObservation:
        idx = IDX_LH if side == "left" else IDX_RH
        state = self._states[side]
        if candidate is None:
            state.gap = min(int(state.gap) + 1, 999)
            if state.gap > int(self.config.offline_max_track_gap):
                state.landmarks = None
                state.velocity = None
                state.hits = 0
                state.prev_gray = None
            return _empty_tracked_hand(original_detected=bool(observation.mask[idx]), reason="offline_unassigned")

        pts = np.asarray(candidate["landmarks"], dtype=np.float32).reshape(21, 3)
        if state.landmarks is not None:
            state.velocity = pts[:, :2] - state.landmarks[:, :2]
        else:
            state.velocity = np.zeros((21, 2), dtype=np.float32)
        state.landmarks = pts.copy()
        source = str(candidate.get("source", "hands"))
        if source == "flow":
            state.gap = int(candidate.get("flow_gap", int(state.gap) + 1))
        else:
            state.gap = 0
            state.hits += 1
        if gray is not None:
            state.prev_gray = gray.copy()
        flow_confirmed = (
            source == "flow"
            and state.hits >= int(self.config.offline_min_track_hits)
            and state.gap <= int(self.config.tracker_max_flow_gap)
        )
        accepted = source in ("holistic", "contact_imputed") or state.hits >= int(self.config.offline_min_track_hits) or flow_confirmed
        return _tracked_hand_from_landmarks(
            pts,
            source,
            float(candidate.get("score", 0.0)),
            "offline_assigned" if accepted else "offline_tentative",
            0,
            bool(candidate.get("original_detected", bool(observation.mask[idx]))),
            self.config,
            accepted=accepted,
            track_id=int(state.track_id),
        )


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

        accepted_masks = original_masks.copy()
        accepted_masks[:, IDX_LH] = _hand_frame_nonzero(left_hand)
        accepted_masks[:, IDX_RH] = _hand_frame_nonzero(right_hand)
        self._resolve_hand_visibility_conflicts(
            left_hand,
            right_hand,
            accepted_masks,
            observations,
            origins,
            scales,
        )
        self._gate_pose_arms(pose, left_hand, right_hand, accepted_masks)

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
        features = sanitize_sequence(features, masks=accepted_masks, preserve_flags=False, zero_missing_hands=True)
        augmented = np.concatenate([features, accepted_masks.astype(np.float32)], axis=1).astype(np.float32)
        scores = self._compute_scores(features)
        self._last_metadata = self._make_tracking_metadata(observations, original_masks, accepted_masks)
        return [augmented[i] for i in range(T)], scores

    def _resolve_hand_visibility_conflicts(self, left_hand, right_hand, accepted_masks, observations, origins, scales):
        """
        Enforce the V4.2 visibility contract before angles/features are emitted.

        Close hands are valid in BISINDO contact signs. We therefore reject only
        near-identical skeleton hypotheses, not merely close wrist anchors.
        """
        T = len(observations)
        left_mask = accepted_masks[:, IDX_LH].astype(bool)
        right_mask = accepted_masks[:, IDX_RH].astype(bool)

        for i in range(T):
            if not (left_mask[i] and right_mask[i]):
                continue

            left_anchor = left_hand[i, 0, :2]
            right_anchor = right_hand[i, 0, :2]
            anchor_dist = float(np.linalg.norm(left_anchor - right_anchor))
            left_meta = _observation_hand_metadata(observations[i], "left")
            right_meta = _observation_hand_metadata(observations[i], "right")
            roi_iou = _roi_iou(left_meta.roi if left_meta else None, right_meta.roi if right_meta else None)
            mean_dist = _mean_landmark_distance(left_hand[i], right_hand[i])
            severe_duplicate = mean_dist <= float(self.config.hand_conflict_mean_landmark_radius)
            roi_duplicate = (
                roi_iou >= float(self.config.hand_conflict_extreme_roi_iou)
                and anchor_dist <= float(self.config.hand_conflict_extreme_anchor_radius)
            )

            if not (severe_duplicate or roi_duplicate):
                continue

            left_score = _hand_conflict_score(observations[i], "left", left_hand[i], origins[i], scales[i], self.config)
            right_score = _hand_conflict_score(observations[i], "right", right_hand[i], origins[i], scales[i], self.config)
            if right_score > left_score:
                left_hand[i] = 0.0
                left_mask[i] = False
            else:
                right_hand[i] = 0.0
                right_mask[i] = False

        _remove_short_accepted_segments(left_hand, left_mask, observations, "left", self.config)
        _remove_short_accepted_segments(right_hand, right_mask, observations, "right", self.config)

        accepted_masks[:, IDX_LH] = left_mask
        accepted_masks[:, IDX_RH] = right_mask

    def _gate_pose_arms(self, pose, left_hand, right_hand, accepted_masks):
        left_mask = accepted_masks[:, IDX_LH].astype(bool)
        right_mask = accepted_masks[:, IDX_RH].astype(bool)
        _gate_single_pose_arm(pose, left_hand, left_mask, 0, 2, 4, self.config)
        _gate_single_pose_arm(pose, right_hand, right_mask, 1, 3, 5, self.config)

    def _processing_masks_and_confidence(self, observations, original_masks):
        masks = original_masks.copy()
        confidence = original_masks.astype(np.float32)
        for i, obs in enumerate(observations):
            if isinstance(obs, TrackedFrameObservation):
                masks[i, IDX_LH] = bool(obs.left_tracking.accepted)
                masks[i, IDX_RH] = bool(obs.right_tracking.accepted)
                confidence[i, IDX_LH] = float(obs.left_tracking.confidence) if obs.left_tracking.accepted else 0.0
                confidence[i, IDX_RH] = float(obs.right_tracking.confidence) if obs.right_tracking.accepted else 0.0
        return masks.astype(bool), np.clip(confidence, 0.0, 1.0).astype(np.float32)

    def _make_tracking_metadata(self, observations, original_masks, accepted_masks):
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
                    "left": _tracking_metadata_dict(left, bool(accepted_masks[i, IDX_LH])),
                    "right": _tracking_metadata_dict(right, bool(accepted_masks[i, IDX_RH])),
                    "candidates": _candidate_metadata_list(obs.candidates) if isinstance(obs, TrackedFrameObservation) else [],
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
        quality_valid, vmax = _quality_gate_hand(
            anchors,
            lengths,
            bbox,
            hand_valid,
            self.config,
            protected_valid=source_confidence >= 0.95,
        )
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
        trusted_detector_valid = quality_valid & (source_confidence >= 0.80)
        frame_valid = _clamp_anchor_steps(anchor_smooth, frame_valid, vmax, trusted_detector_valid)

        reconstructed = np.zeros((T, 21, 3), dtype=np.float32)
        reconstructed[frame_valid, 0, :] = anchor_smooth[frame_valid]
        for e_idx, (parent, child) in enumerate(HAND_TREE_EDGES):
            reconstructed[frame_valid, child, :2] = (
                reconstructed[frame_valid, parent, :2] + dirs[frame_valid, e_idx, :2] * locked_lengths[e_idx]
            )
            reconstructed[frame_valid, child, 2] = reconstructed[frame_valid, parent, 2] + locked_z[e_idx]

        frame_valid = _clamp_skeleton_steps(reconstructed, frame_valid, trusted_detector_valid, self.config.hand_skeleton_step_max)
        reconstructed[~frame_valid] = 0.0
        reconstructed = np.clip(reconstructed, -self.config.hand_abs_limit, self.config.hand_abs_limit)
        return reconstructed.astype(np.float32)

    def _build_from_legacy_vectors(self):
        if not self._legacy_vectors:
            return [], []
        seq = np.stack(self._legacy_vectors).astype(np.float32)
        masks = np.stack(self._legacy_masks).astype(bool)
        features = sanitize_sequence(seq, masks=masks, preserve_flags=False, zero_missing_hands=True)
        accepted_masks = masks.copy()
        accepted_masks[:, IDX_LH] = _hand_frame_nonzero(features[:, SLICE_LH].reshape(len(features), 21, 3))
        accepted_masks[:, IDX_RH] = _hand_frame_nonzero(features[:, SLICE_RH].reshape(len(features), 21, 3))
        augmented = np.concatenate([features, accepted_masks.astype(np.float32)], axis=1).astype(np.float32)
        scores = self._compute_scores(features)
        self._last_metadata = [
            {
                "pose_detected": bool(masks[i, IDX_POSE]),
                "left": {
                    "source": "holistic" if masks[i, IDX_LH] else "missing",
                    "confidence": 1.0 if accepted_masks[i, IDX_LH] else 0.0,
                    "gap_age": 0 if masks[i, IDX_LH] else 999,
                    "quality_reason": "legacy_vector",
                    "original_detected": bool(masks[i, IDX_LH]),
                    "accepted": bool(accepted_masks[i, IDX_LH]),
                    "track_id": -1,
                    "rendered": bool(accepted_masks[i, IDX_LH]),
                    "roi": None,
                },
                "right": {
                    "source": "holistic" if masks[i, IDX_RH] else "missing",
                    "confidence": 1.0 if accepted_masks[i, IDX_RH] else 0.0,
                    "gap_age": 0 if masks[i, IDX_RH] else 999,
                    "quality_reason": "legacy_vector",
                    "original_detected": bool(masks[i, IDX_RH]),
                    "accepted": bool(accepted_masks[i, IDX_RH]),
                    "track_id": -1,
                    "rendered": bool(accepted_masks[i, IDX_RH]),
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
    zero_missing_hands: bool = True,
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
    left_angles[~masks[:, IDX_LH]] = 0.0
    right_angles[~masks[:, IDX_RH]] = 0.0

    sanitized = np.concatenate(
        [pose.reshape(T, N_POSE), left.reshape(T, N_HAND), right.reshape(T, N_HAND), left_angles, right_angles],
        axis=1,
    ).astype(np.float32)

    if flags is not None:
        if preserve_flags:
            flags = (flags >= 0.5).astype(np.float32)
            flags[:, IDX_LH] = _hand_frame_nonzero(left).astype(np.float32)
            flags[:, IDX_RH] = _hand_frame_nonzero(right).astype(np.float32)
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
        "tracking_left_accepted": bool(left.get("accepted", left.get("rendered", False))),
        "tracking_right_accepted": bool(right.get("accepted", right.get("rendered", False))),
        "tracking_left_original_detected": bool(left.get("original_detected", False)),
        "tracking_right_original_detected": bool(right.get("original_detected", False)),
        "tracking_left_track_id": int(left.get("track_id", -1)),
        "tracking_right_track_id": int(right.get("track_id", -1)),
        "tracking_metadata": json.dumps(metadata, ensure_ascii=False, sort_keys=True),
    }


def _tracking_metadata_dict(hand: TrackedHandObservation, rendered: bool) -> dict:
    source = str(hand.source)
    confidence = float(hand.confidence)
    reason = str(hand.quality_reason)
    gap_age = int(hand.gap_age)
    original = bool(hand.original_detected)
    track_id = int(hand.track_id)
    roi = list(hand.roi) if hand.roi is not None else None
    if rendered and source == "missing":
        source = "imputed"
        confidence = 0.35
        reason = "short_internal_gap_fill"
        gap_age = 1
    return {
        "source": source,
        "confidence": confidence,
        "gap_age": gap_age,
        "quality_reason": reason,
        "original_detected": original,
        "accepted": bool(rendered),
        "track_id": track_id,
        "rendered": bool(rendered),
        "roi": roi,
    }


def _observation_hand_metadata(
    observation: FrameObservation | TrackedFrameObservation,
    side: str,
) -> TrackedHandObservation | None:
    if isinstance(observation, TrackedFrameObservation):
        return observation.left_tracking if side == "left" else observation.right_tracking
    idx = IDX_LH if side == "left" else IDX_RH
    raw = observation.left_hand_raw if side == "left" else observation.right_hand_raw
    if bool(observation.mask[idx]):
        return _tracked_hand_from_landmarks(
            raw,
            "holistic",
            DEFAULT_PREPROCESS_CONFIG.tracker_holistic_confidence,
            "legacy_holistic",
            0,
            True,
        )
    return _empty_tracked_hand(False)


def _source_priority(source: str) -> float:
    return {
        "holistic": 5.0,
        "hands": 4.0,
        "hands_enhanced": 4.0,
        "hands_raw": 3.5,
        "contact_imputed": 2.6,
        "flow": 2.0,
        "hands_tentative": 1.0,
        "imputed": 1.0,
        "prediction": 0.0,
        "missing": 0.0,
    }.get(str(source), 0.5)


def _pose_wrist_distance_body(
    observation: FrameObservation | TrackedFrameObservation,
    side: str,
    hand_anchor: np.ndarray,
    origin: np.ndarray,
    scale: float,
) -> float:
    if not bool(observation.mask[IDX_POSE]):
        return 3.0
    wrist_idx = 4 if side == "left" else 5
    pose_body = _to_body_frame(observation.pose_raw, origin, scale)
    return float(np.linalg.norm(np.asarray(hand_anchor[:2], dtype=np.float32) - pose_body[wrist_idx, :2]))


def _hand_conflict_score(
    observation: FrameObservation | TrackedFrameObservation,
    side: str,
    hand_points: np.ndarray,
    origin: np.ndarray,
    scale: float,
    cfg: PreprocessConfig,
) -> float:
    metadata = _observation_hand_metadata(observation, side)
    if metadata is None:
        return -10.0
    source = str(metadata.source)
    pose_dist = _pose_wrist_distance_body(observation, side, hand_points[0], origin, scale)
    score = _source_priority(source)
    score += 1.0 if bool(metadata.original_detected) else 0.0
    score += float(np.clip(metadata.confidence, 0.0, 1.0))
    score -= min(pose_dist, 3.0) * 0.85
    if not bool(metadata.accepted):
        score -= 1.5
    return float(score)


def _roi_iou(a, b) -> float:
    if not a or not b or len(a) != 4 or len(b) != 4:
        return 0.0
    ax0, ay0, ax1, ay1 = [float(x) for x in a]
    bx0, by0, bx1, by1 = [float(x) for x in b]
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    denom = area_a + area_b - inter
    if denom <= 1e-8:
        return 0.0
    return float(inter / denom)


def _remove_short_accepted_segments(
    hand: np.ndarray,
    mask: np.ndarray,
    observations: list[FrameObservation | TrackedFrameObservation],
    side: str,
    cfg: PreprocessConfig,
):
    min_len = int(max(1, cfg.hand_min_accepted_segment))
    if min_len <= 1 or len(mask) == 0:
        return
    start = None
    for i in range(len(mask) + 1):
        active = bool(mask[i]) if i < len(mask) else False
        if active and start is None:
            start = i
        elif start is not None and not active:
            end = i
            if end - start < min_len:
                has_original = False
                for j in range(start, end):
                    meta = _observation_hand_metadata(observations[j], side)
                    if meta is not None and bool(meta.original_detected):
                        has_original = True
                        break
                if not has_original:
                    hand[start:end] = 0.0
                    mask[start:end] = False
            start = None


def _gate_single_pose_arm(
    pose: np.ndarray,
    hand: np.ndarray,
    mask: np.ndarray,
    shoulder_idx: int,
    elbow_idx: int,
    wrist_idx: int,
    cfg: PreprocessConfig,
):
    for i in range(len(pose)):
        if not bool(mask[i]):
            pose[i, elbow_idx] = 0.0
            pose[i, wrist_idx] = 0.0
            continue
        wrist = hand[i, 0].copy()
        shoulder = pose[i, shoulder_idx].copy()
        elbow = pose[i, elbow_idx].copy()
        span = float(np.linalg.norm(wrist[:2] - shoulder[:2]))
        bad_elbow = (
            not np.isfinite(elbow).all()
            or np.allclose(elbow, 0.0)
            or float(np.linalg.norm(elbow[:2] - shoulder[:2])) > max(0.25, span * cfg.pose_arm_max_elbow_ratio)
            or float(np.linalg.norm(elbow[:2] - wrist[:2])) > max(0.25, span * cfg.pose_arm_max_elbow_ratio)
        )
        pose[i, wrist_idx] = wrist
        if bad_elbow:
            pose[i, elbow_idx] = shoulder * 0.52 + wrist * 0.48


def _candidate_from_points(
    points: np.ndarray,
    source: str,
    side_hint: str,
    score: float,
    frame_idx: int,
    original_detected: bool,
    variant: str,
) -> dict:
    pts = np.asarray(points, dtype=np.float32).reshape(21, 3)
    return {
        "landmarks": pts,
        "source": str(source),
        "side_hint": str(side_hint).lower(),
        "score": float(np.clip(score, 0.0, 1.0)),
        "frame_idx": int(frame_idx),
        "original_detected": bool(original_detected),
        "variant": str(variant),
        "roi": _landmarks_roi(pts),
        "anchor": pts[0, :2].astype(float).tolist(),
    }


def _candidate_metadata_list(candidates: list[dict] | None) -> list[dict]:
    out = []
    for idx, candidate in enumerate(candidates or []):
        if not isinstance(candidate, dict):
            continue
        pts = np.asarray(candidate.get("landmarks", np.zeros((21, 3))), dtype=np.float32).reshape(21, 3)
        roi = candidate.get("roi") or _landmarks_roi(pts)
        out.append(
            {
                "idx": int(idx),
                "source": str(candidate.get("source", "unknown")),
                "variant": str(candidate.get("variant", candidate.get("roi_source", ""))),
                "side_hint": str(candidate.get("side_hint", candidate.get("label", ""))).lower(),
                "score": float(candidate.get("score", 0.0)),
                "frame_idx": int(candidate.get("frame_idx", -1)),
                "original_detected": bool(candidate.get("original_detected", False)),
                "roi": list(roi) if roi is not None else None,
                "anchor": pts[0, :2].astype(float).tolist(),
            }
        )
    return out


def _mean_landmark_distance(a: np.ndarray, b: np.ndarray) -> float:
    pa = np.asarray(a, dtype=np.float32).reshape(21, 3)
    pb = np.asarray(b, dtype=np.float32).reshape(21, 3)
    return float(np.mean(np.linalg.norm(pa[:, :2] - pb[:, :2], axis=1)))


def _candidate_rank(candidate: dict) -> float:
    source = str(candidate.get("source", "unknown"))
    source_bonus = {"holistic": 3.0, "hands_enhanced": 2.0, "hands_raw": 1.6, "hands": 1.4}.get(source, 0.5)
    return source_bonus + float(candidate.get("score", 0.0))


def _dedupe_candidates(candidates: list[dict], cfg: PreprocessConfig) -> list[dict]:
    kept: list[dict] = []
    for candidate in sorted(candidates, key=_candidate_rank, reverse=True):
        duplicate = False
        for existing in kept:
            mean_dist = _mean_landmark_distance(candidate["landmarks"], existing["landmarks"])
            iou = _roi_iou(candidate.get("roi"), existing.get("roi"))
            anchor_dist = float(np.linalg.norm(candidate["landmarks"][0, :2] - existing["landmarks"][0, :2]))
            if mean_dist <= cfg.hand_conflict_mean_landmark_radius:
                duplicate = True
                break
            if iou >= cfg.hand_conflict_extreme_roi_iou and anchor_dist <= cfg.hand_conflict_extreme_anchor_radius:
                duplicate = True
                break
        if not duplicate:
            kept.append(candidate)
    return kept


def _pose_wrist_anchor_raw(side: str, observation: FrameObservation) -> np.ndarray | None:
    if not bool(observation.mask[IDX_POSE]):
        return None
    wrist_idx = 4 if side == "left" else 5
    return np.clip(observation.pose_raw[wrist_idx, :2], 0.0, 1.0).astype(np.float32)


def detect_full_frame_hand_candidates(
    frame_rgb,
    hands_solution,
    config: PreprocessConfig | None = None,
    source: str = "hands",
    variant: str = "full_frame",
    frame_idx: int = -1,
) -> list[dict]:
    cfg = config or DEFAULT_PREPROCESS_CONFIG
    if hands_solution is None or frame_rgb is None:
        return []
    try:
        results = hands_solution.process(frame_rgb)
    except Exception:
        return []
    candidates = []
    for candidate in _parse_hands_candidates(results, cfg):
        item = _candidate_from_points(
            candidate["landmarks"],
            source,
            str(candidate.get("label", "")).lower(),
            float(candidate.get("score", cfg.tracker_hands_confidence)),
            frame_idx,
            original_detected=False,
            variant=variant,
        )
        candidates.append(item)
    return candidates


def enhance_frame_for_tracking(frame_bgr, config: PreprocessConfig | None = None):
    """
    Lightweight detector-domain enhancement for MediaPipe input.

    This is deliberately classical CV rather than a neural low-light model:
    gray-world white balance, CLAHE on luminance, bounded gamma correction, and
    a small saturation lift. It is deterministic, cheap on Jetson, and keeps the
    camera stream geometry untouched.
    """
    cfg = config or DEFAULT_PREPROCESS_CONFIG
    if cv2 is None or frame_bgr is None or not bool(cfg.frame_enhance_enabled):
        return frame_bgr
    try:
        frame = np.asarray(frame_bgr)
        if frame.ndim != 3 or frame.shape[2] != 3:
            return frame_bgr
        out = frame.astype(np.float32)

        channel_means = out.reshape(-1, 3).mean(axis=0)
        gray_mean = float(channel_means.mean())
        scales = gray_mean / np.maximum(channel_means, 1.0)
        scales = np.clip(scales, 0.82, 1.18)
        out = np.clip(out * scales.reshape(1, 1, 3), 0.0, 255.0).astype(np.uint8)

        lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
        tile = int(max(2, cfg.frame_enhance_tile_grid_size))
        clahe = cv2.createCLAHE(
            clipLimit=float(cfg.frame_enhance_clahe_clip_limit),
            tileGridSize=(tile, tile),
        )
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        luma = float(cv2.cvtColor(out, cv2.COLOR_BGR2GRAY).mean())
        if luma > 1.0:
            target = float(np.clip(cfg.frame_enhance_target_luma, 70.0, 180.0)) / 255.0
            current = float(np.clip(luma / 255.0, 1e-3, 0.999))
            gamma = math.log(target) / math.log(current)
            gamma = float(np.clip(gamma, cfg.frame_enhance_gamma_min, cfg.frame_enhance_gamma_max))
            lut = np.array([np.clip((i / 255.0) ** gamma * 255.0, 0, 255) for i in range(256)], dtype=np.uint8)
            out = cv2.LUT(out, lut)

        sat_gain = float(cfg.frame_enhance_saturation_gain)
        if abs(sat_gain - 1.0) > 1e-3:
            hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat_gain, 0.0, 255.0)
            out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        return out
    except Exception:
        return frame_bgr


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
    if isinstance(hands_results, list):
        out = []
        for candidate in hands_results:
            if not isinstance(candidate, dict) or "landmarks" not in candidate:
                continue
            pts = np.asarray(candidate["landmarks"], dtype=np.float32).reshape(21, 3)
            if _plausible_raw_hand(pts, cfg):
                item = dict(candidate)
                item["landmarks"] = pts
                out.append(item)
        return out
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


def _gate_anchor_speed(
    anchors: np.ndarray,
    valid: np.ndarray,
    vmax: float,
    protected_valid: np.ndarray | None = None,
) -> np.ndarray:
    gated = np.zeros_like(valid, dtype=bool)
    last_idx: int | None = None
    protected = np.asarray(protected_valid, dtype=bool) if protected_valid is not None else np.zeros_like(valid, dtype=bool)
    for idx in np.where(valid)[0]:
        if last_idx is None:
            gated[idx] = True
            last_idx = int(idx)
            continue

        dt = max(int(idx - last_idx), 1)
        speed = float(np.linalg.norm(anchors[idx, :2] - anchors[last_idx, :2]) / dt)
        if (np.isfinite(speed) and speed <= vmax) or protected[idx]:
            gated[idx] = True
            last_idx = int(idx)
    return gated


def _quality_gate_hand(
    anchors: np.ndarray,
    lengths: np.ndarray,
    bbox: np.ndarray,
    detector_valid: np.ndarray,
    cfg: PreprocessConfig,
    protected_valid: np.ndarray | None = None,
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
        protected = np.asarray(protected_valid, dtype=bool) if protected_valid is not None else None
        valid = _gate_anchor_speed(anchors, valid, vmax, protected)
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
