"""
feature_extractor_robust_v4.py
===========================
Robust live-friendly pose + two-hand tracker for Jetson / CPU.

V4 design goal:
- Arm-locked hand identity: a detected hand can only become physical left/right if
  it is physically attached to the matching arm anchor.
- Anti-jitter stabilization: static hands should not create fake movement.
- Arm anchor smoothing: pose wrist/elbow jitter is filtered before being used as
  the hand assignment anchor.
- Optional hand-to-arm wrist snap: keeps the visual/feature hand skeleton attached
  to the lengan without large jumps.
- Keeps compatibility 179-dim feature vector for old training pipelines.
- Adds palm/shoulder-relative feature modes for BISINDO: palm and palm_angles.

Dependencies:
    pip install mediapipe opencv-python numpy
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import time

import cv2
import numpy as np
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands

# Old feature layout compatibility: 6 pose points, 2 hands, 16 angles per hand, 3 flags.
POSE_KP_IDX = [11, 12, 13, 14, 23, 24]  # L/R shoulder, elbow, hip
POSE_ARM_IDX = {
    "left": {"shoulder": 11, "elbow": 13, "wrist": 15},
    "right": {"shoulder": 12, "elbow": 14, "wrist": 16},
}

HAND_ANGLE_TRIPLETS = [
    (5, 0, 1), (0, 1, 2), (1, 2, 3), (2, 3, 4),          # thumb
    (0, 5, 6), (5, 6, 7), (6, 7, 8),                     # index
    (0, 9, 10), (9, 10, 11), (10, 11, 12),               # middle
    (0, 13, 14), (13, 14, 15), (14, 15, 16),             # ring
    (0, 17, 18), (17, 18, 19), (18, 19, 20),             # pinky
]

HAND_CONNECTIONS = tuple(mp_hands.HAND_CONNECTIONS)


def _safe_norm(x: np.ndarray, eps: float = 1e-6) -> float:
    return float(max(np.linalg.norm(x), eps))


def _dist2(a: np.ndarray, b: np.ndarray) -> float:
    return _safe_norm(a[:2] - b[:2])


def _ema_vec(prev: np.ndarray, cur: np.ndarray, alpha: float) -> np.ndarray:
    return ((1.0 - alpha) * prev + alpha * cur).astype(np.float32)


def _deadband_vec(prev: np.ndarray, cur: np.ndarray, deadband: float) -> np.ndarray:
    """Freeze tiny 2D movements. This removes MediaPipe micro-jitter."""
    out = cur.astype(np.float32).copy()
    move_xy = np.linalg.norm(out[:, :2] - prev[:, :2], axis=1)
    tiny = move_xy < deadband
    if np.any(tiny):
        out[tiny, :2] = prev[tiny, :2]
        z_tiny = np.abs(out[tiny, 2] - prev[tiny, 2]) < deadband * 1.5
        # Apply only to the tiny subset.
        idx = np.where(tiny)[0]
        out[idx[z_tiny], 2] = prev[idx[z_tiny], 2]
    return out


def _angle_at_b(pa: np.ndarray, pb: np.ndarray, pc: np.ndarray) -> float:
    ba, bc = pa - pb, pc - pb
    denom = _safe_norm(ba) * _safe_norm(bc)
    cos_val = float(np.dot(ba, bc) / denom)
    return float(np.arccos(np.clip(cos_val, -1.0, 1.0)))


def _hand_angles(lm: np.ndarray) -> np.ndarray:
    angles = np.zeros(16, dtype=np.float32)
    for j, (a, b, c) in enumerate(HAND_ANGLE_TRIPLETS):
        angles[j] = _angle_at_b(lm[a], lm[b], lm[c])
    return angles


def _hand_scale(lm: np.ndarray) -> float:
    # Stable local hand scale: wrist -> middle MCP and index MCP -> pinky MCP.
    return max(_safe_norm(lm[0] - lm[9]), _safe_norm(lm[5] - lm[17]), 1e-4)


def _relative_hand(lm: np.ndarray) -> np.ndarray:
    return ((lm - lm[0]) / _hand_scale(lm)).astype(np.float32)


def _np_landmarks(lm_list) -> np.ndarray:
    arr = np.zeros((21, 3), dtype=np.float32)
    for i, p in enumerate(lm_list.landmark):
        arr[i] = (p.x, p.y, p.z)
    return arr


def _clip01(v: float) -> float:
    return float(np.clip(v, 0.0, 1.0))


def _visibility(lm) -> float:
    return float(getattr(lm, "visibility", 0.0) or 0.0)


@dataclass
class ArmAnchor:
    """Pose-derived physical arm anchor for one side of the body."""

    side: str
    shoulder: np.ndarray
    elbow: np.ndarray
    wrist: np.ndarray
    shoulder_vis: float
    elbow_vis: float
    wrist_vis: float
    age: int = 0

    @property
    def wrist_ok(self) -> bool:
        # Held anchors are allowed for a few frames but with reduced visibility.
        return self.wrist_vis >= 0.24

    @property
    def elbow_ok(self) -> bool:
        return self.elbow_vis >= 0.22

    @property
    def shoulder_ok(self) -> bool:
        return self.shoulder_vis >= 0.20

    @property
    def forearm_len(self) -> float:
        return _safe_norm(self.wrist[:2] - self.elbow[:2])

    @property
    def upperarm_len(self) -> float:
        return _safe_norm(self.elbow[:2] - self.shoulder[:2])

    @property
    def scale(self) -> float:
        # 0.055 keeps gate usable when pose arm is very short/noisy.
        return max(self.forearm_len, 0.65 * self.upperarm_len, 0.055)

    def gate_radius(self, max_gate: float) -> float:
        # Normalized image-space radius. Smaller = fewer ghost swaps.
        return float(np.clip(max(0.052, 0.72 * self.scale), 0.052, max_gate))


@dataclass
class ArmState:
    side: str
    hold_frames: int = 8
    shoulder: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    elbow: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    wrist: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    shoulder_vis: float = 0.0
    elbow_vis: float = 0.0
    wrist_vis: float = 0.0
    valid: bool = False
    age: int = 10_000
    jitter_ema: float = 0.0

    def update(self, raw: ArmAnchor, pose_deadband: float = 0.0045) -> ArmAnchor:
        raw_pts = np.stack([raw.shoulder, raw.elbow, raw.wrist]).astype(np.float32)
        if not self.valid or self.age > self.hold_frames:
            smoothed = raw_pts
            move = 0.0
        else:
            prev = np.stack([self.shoulder, self.elbow, self.wrist]).astype(np.float32)
            move_per_pt = np.linalg.norm(raw_pts[:, :2] - prev[:, :2], axis=1)
            raw_pts2 = _deadband_vec(prev, raw_pts, pose_deadband)
            # Dynamic alpha: fast for actual arm motion, slow for tiny static jitter.
            median_move = float(np.median(move_per_pt))
            if median_move < pose_deadband:
                alpha = 0.06
            elif median_move > 0.035:
                alpha = 0.62
            else:
                alpha = 0.28
            smoothed = _ema_vec(prev, raw_pts2, alpha)
            move = median_move

        self.shoulder, self.elbow, self.wrist = smoothed[0], smoothed[1], smoothed[2]
        self.shoulder_vis = float(raw.shoulder_vis)
        self.elbow_vis = float(raw.elbow_vis)
        self.wrist_vis = float(raw.wrist_vis)
        self.valid = True
        self.age = 0
        self.jitter_ema = 0.85 * self.jitter_ema + 0.15 * move
        return self.anchor()

    def miss(self) -> Optional[ArmAnchor]:
        self.age += 1
        if self.valid and self.age <= self.hold_frames:
            # Keep the previous arm anchor briefly, but reduce trust each missed frame.
            decay = 0.70
            self.shoulder_vis *= decay
            self.elbow_vis *= decay
            self.wrist_vis *= decay
            return self.anchor()
        self.valid = False
        self.shoulder_vis = self.elbow_vis = self.wrist_vis = 0.0
        return None

    def anchor(self) -> ArmAnchor:
        return ArmAnchor(
            side=self.side,
            shoulder=self.shoulder.astype(np.float32),
            elbow=self.elbow.astype(np.float32),
            wrist=self.wrist.astype(np.float32),
            shoulder_vis=float(self.shoulder_vis),
            elbow_vis=float(self.elbow_vis),
            wrist_vis=float(self.wrist_vis),
            age=int(self.age),
        )


@dataclass
class HandDetection:
    landmarks: np.ndarray
    score: float = 1.0
    mp_label: str = ""

    @property
    def wrist(self) -> np.ndarray:
        return self.landmarks[0]

    @property
    def center(self) -> np.ndarray:
        return np.mean(self.landmarks[[0, 5, 9, 13, 17]], axis=0)


@dataclass
class HandState:
    name: str
    hold_frames: int = 8
    landmarks: np.ndarray = field(default_factory=lambda: np.zeros((21, 3), dtype=np.float32))
    detected: bool = False         # true only for current frame detection
    valid: bool = False            # true for detected OR short held state
    age: int = 10_000              # frames since last detection
    quality: float = 0.0           # decays during occlusion
    last_update_t: float = 0.0
    locked_to_arm: bool = False
    arm_link_score: float = 0.0
    motion_ema: float = 0.0
    snap_offset: float = 0.0

    def update(
        self,
        det: HandDetection,
        alpha_slow: float = 0.22,
        alpha_fast: float = 0.68,
        arm_link_score: float = 1.0,
        hand_deadband: float = 0.0065,
        anchor: Optional[ArmAnchor] = None,
        snap_strength: float = 0.35,
        snap_max: float = 0.030,
    ) -> None:
        new_lm = det.landmarks.astype(np.float32).copy()

        # Optional: gently pull the entire hand skeleton so hand wrist stays attached
        # to the smoothed pose wrist. This fixes "hand lepas dari lengan" in overlay
        # and also reduces absolute-position jitter. The shift is clamped so pose
        # errors cannot drag the hand too far.
        self.snap_offset = 0.0
        if anchor is not None and anchor.wrist_ok and snap_strength > 0.0:
            delta = (anchor.wrist - new_lm[0]).astype(np.float32)
            delta[2] *= 0.25  # z from pose and hand are not directly comparable.
            dist_xy = _safe_norm(delta[:2])
            if dist_xy > 1e-6:
                use_delta = delta * float(snap_strength)
                use_dist = _safe_norm(use_delta[:2])
                if use_dist > snap_max:
                    use_delta[:2] *= snap_max / use_dist
                    use_delta[2] = np.clip(use_delta[2], -snap_max, snap_max)
                new_lm += use_delta
                self.snap_offset = _safe_norm(use_delta[:2])

        if not self.valid or self.age > self.hold_frames:
            self.landmarks = new_lm
            motion = 0.0
        else:
            prev = self.landmarks
            point_motion = np.linalg.norm(new_lm[:, :2] - prev[:, :2], axis=1)
            motion = float(np.median(point_motion))
            wrist_jump = _safe_norm(new_lm[0, :2] - prev[0, :2])

            # Hard freeze tiny landmark jitter. This is the main fix for
            # "tangan diem tapi kebaca gerak".
            new_lm = _deadband_vec(prev, new_lm, hand_deadband)

            if motion < hand_deadband:
                alpha = 0.04
            elif wrist_jump > 0.050 or motion > 0.022:
                alpha = alpha_fast
            else:
                alpha = alpha_slow

            self.landmarks = ((1.0 - alpha) * prev + alpha * new_lm).astype(np.float32)

        self.motion_ema = 0.84 * self.motion_ema + 0.16 * motion
        self.detected = True
        self.valid = True
        self.age = 0
        self.locked_to_arm = True
        self.arm_link_score = _clip01(arm_link_score)
        self.quality = _clip01((0.15 + 0.85 * det.score) * (0.35 + 0.65 * self.arm_link_score))
        self.last_update_t = time.perf_counter()

    def miss(self) -> None:
        self.detected = False
        self.locked_to_arm = False
        self.arm_link_score = 0.0
        self.snap_offset = 0.0
        self.age += 1
        if self.age <= self.hold_frames:
            # Keep last pose briefly but lower confidence.
            self.valid = True
            self.quality = _clip01(self.quality * 0.55)
            self.motion_ema *= 0.70
        else:
            self.valid = False
            self.quality = 0.0
            self.motion_ema = 0.0


@dataclass
class RobustFrameResult:
    pose_landmarks: Optional[object]
    pose_selected: np.ndarray
    pose_ok: float
    arm_anchors: Dict[str, ArmAnchor]
    hands: Dict[str, HandState]
    raw_detections: List[HandDetection]
    self_handshake_score: float
    infer_ms: float
    arm_gate: float = 0.13

    @property
    def left_hand(self) -> HandState:
        return self.hands["left"]

    @property
    def right_hand(self) -> HandState:
        return self.hands["right"]


class RobustHolisticTracker:
    """Pose + Hands with arm-locked identity assignment.

    Important rule:
        A hand candidate is not allowed to become physical left/right unless it
        is attached to that side's arm anchor. Identity is based on body side,
        not just MediaPipe's handedness label.
    """

    def __init__(
        self,
        model_complexity: int = 0,
        det_conf: float = 0.55,
        track_conf: float = 0.55,
        pose_every: int = 1,
        hold_frames: int = 6,
        arm_hold_frames: int = 8,
        mirror_handedness: bool = True,
        max_num_hands: int = 2,
        arm_locked: bool = True,
        arm_gate: float = 0.13,
        allow_unanchored_when_pose_missing: bool = False,
        hand_deadband: float = 0.0065,
        pose_deadband: float = 0.0045,
        snap_strength: float = 0.35,
        snap_max: float = 0.030,
    ) -> None:
        self.pose_every = max(1, int(pose_every))
        self.frame_i = 0
        self.pose_result = None
        self.mirror_handedness = bool(mirror_handedness)
        self.arm_locked = bool(arm_locked)
        self.arm_gate = float(arm_gate)
        self.allow_unanchored_when_pose_missing = bool(allow_unanchored_when_pose_missing)
        self.hand_deadband = float(hand_deadband)
        self.pose_deadband = float(pose_deadband)
        self.snap_strength = float(snap_strength)
        self.snap_max = float(snap_max)

        self.pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=int(model_complexity),
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=float(det_conf),
            min_tracking_confidence=float(track_conf),
        )
        self.hands_detector = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=int(max_num_hands),
            model_complexity=int(model_complexity),
            min_detection_confidence=float(det_conf),
            min_tracking_confidence=float(track_conf),
        )
        self.hands = {
            "left": HandState("left", hold_frames=hold_frames),
            "right": HandState("right", hold_frames=hold_frames),
        }
        self.arm_states = {
            "left": ArmState("left", hold_frames=arm_hold_frames),
            "right": ArmState("right", hold_frames=arm_hold_frames),
        }

    def close(self) -> None:
        self.pose.close()
        self.hands_detector.close()

    def reset(self) -> None:
        for st in self.hands.values():
            st.detected = False
            st.valid = False
            st.age = 10_000
            st.quality = 0.0
            st.locked_to_arm = False
            st.arm_link_score = 0.0
            st.motion_ema = 0.0
            st.snap_offset = 0.0
        for st in self.arm_states.values():
            st.valid = False
            st.age = 10_000
            st.shoulder_vis = st.elbow_vis = st.wrist_vis = 0.0
            st.jitter_ema = 0.0

    def _raw_arm_anchors(self) -> Dict[str, ArmAnchor]:
        anchors: Dict[str, ArmAnchor] = {}
        if not self.pose_result or not self.pose_result.pose_landmarks:
            return anchors
        lm = self.pose_result.pose_landmarks.landmark
        for side, ids in POSE_ARM_IDX.items():
            sh, el, wr = lm[ids["shoulder"]], lm[ids["elbow"]], lm[ids["wrist"]]
            anchors[side] = ArmAnchor(
                side=side,
                shoulder=np.array([sh.x, sh.y, sh.z], dtype=np.float32),
                elbow=np.array([el.x, el.y, el.z], dtype=np.float32),
                wrist=np.array([wr.x, wr.y, wr.z], dtype=np.float32),
                shoulder_vis=_visibility(sh),
                elbow_vis=_visibility(el),
                wrist_vis=_visibility(wr),
            )
        return anchors

    def _pose_selected_and_anchors(self) -> Tuple[np.ndarray, float, Dict[str, ArmAnchor]]:
        selected = np.zeros((6, 3), dtype=np.float32)
        pose_ok = 0.0
        raw_anchors = self._raw_arm_anchors()
        anchors: Dict[str, ArmAnchor] = {}

        for side, st in self.arm_states.items():
            raw = raw_anchors.get(side)
            if raw is not None and (raw.wrist_vis >= 0.18 or raw.elbow_vis >= 0.18):
                anchors[side] = st.update(raw, pose_deadband=self.pose_deadband)
            else:
                held = st.miss()
                if held is not None:
                    anchors[side] = held

        # Fill compat pose selected. Shoulders/elbows use smoothed arms when available;
        # hips remain from raw pose because they are less important for hand identity.
        if self.pose_result and self.pose_result.pose_landmarks:
            lm = self.pose_result.pose_landmarks.landmark
            vis = []
            for i, idx in enumerate(POSE_KP_IDX):
                selected[i] = (lm[idx].x, lm[idx].y, lm[idx].z)
                vis.append(_visibility(lm[idx]))

            if "left" in anchors:
                selected[0] = anchors["left"].shoulder
                selected[2] = anchors["left"].elbow
                vis[0] = max(vis[0], anchors["left"].shoulder_vis)
                vis[2] = max(vis[2], anchors["left"].elbow_vis)
            if "right" in anchors:
                selected[1] = anchors["right"].shoulder
                selected[3] = anchors["right"].elbow
                vis[1] = max(vis[1], anchors["right"].shoulder_vis)
                vis[3] = max(vis[3], anchors["right"].elbow_vis)

            pose_ok = 1.0 if np.mean(vis[:4]) > 0.28 else 0.0
        elif anchors:
            # Pose itself was missing this frame, but held arms are still usable.
            vals = []
            if "left" in anchors:
                selected[0] = anchors["left"].shoulder
                selected[2] = anchors["left"].elbow
                vals += [anchors["left"].shoulder_vis, anchors["left"].elbow_vis]
            if "right" in anchors:
                selected[1] = anchors["right"].shoulder
                selected[3] = anchors["right"].elbow
                vals += [anchors["right"].shoulder_vis, anchors["right"].elbow_vis]
            pose_ok = 1.0 if vals and np.mean(vals) > 0.20 else 0.0

        return selected, pose_ok, anchors

    def _detect_hands(self, rgb: np.ndarray) -> List[HandDetection]:
        raw = self.hands_detector.process(rgb)
        detections: List[HandDetection] = []
        if not raw.multi_hand_landmarks:
            return detections

        handed = raw.multi_handedness or []
        for i, lm_list in enumerate(raw.multi_hand_landmarks):
            label, score = "", 1.0
            if i < len(handed) and handed[i].classification:
                cls = handed[i].classification[0]
                label = cls.label or ""
                score = float(cls.score or 0.0)
            detections.append(HandDetection(_np_landmarks(lm_list), score=score, mp_label=label))
        detections.sort(key=lambda d: d.score, reverse=True)
        return detections[:2]

    def _handedness_cost(self, det: HandDetection, physical_label: str) -> float:
        # Tiny weight only. Arm anchor dominates because handedness labels can be
        # inverted on webcam/selfie feeds.
        if not det.mp_label:
            return 0.0
        expected = "Left" if physical_label == "left" else "Right"
        if self.mirror_handedness:
            expected = "Right" if physical_label == "left" else "Left"
        return 0.0 if det.mp_label == expected else 0.035

    def _point_to_segment_dist(self, p: np.ndarray, a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
        """2D distance from p to segment a-b and the segment parameter t."""
        ab = b[:2] - a[:2]
        denom = float(np.dot(ab, ab))
        if denom < 1e-8:
            return _dist2(p, a), 0.0
        t = float(np.clip(np.dot(p[:2] - a[:2], ab) / denom, 0.0, 1.0))
        proj = a[:2] + t * ab
        return _safe_norm(p[:2] - proj), t

    def _arm_link(self, det: HandDetection, physical_label: str, anchors: Dict[str, ArmAnchor]) -> Tuple[bool, float, float]:
        """Return (gate_ok, normalized_distance, link_score)."""
        anchor = anchors.get(physical_label)
        if not self.arm_locked:
            return True, 0.0, 1.0

        if anchor is None or not anchor.wrist_ok:
            if self.allow_unanchored_when_pose_missing:
                return True, 0.0, 0.35
            return False, 99.0, 0.0

        gate = anchor.gate_radius(self.arm_gate)
        wrist_dist = _safe_norm(det.wrist[:2] - anchor.wrist[:2])

        # A hand is attached if its hand wrist is near the pose wrist.
        wrist_ok = wrist_dist <= gate

        # Secondary check: close to forearm segment and near the wrist half of the arm.
        # This helps when pose wrist is a bit noisy but elbow-wrist line is still sane.
        seg_dist, t = self._point_to_segment_dist(det.wrist, anchor.elbow, anchor.wrist)
        seg_ok = (
            anchor.elbow_ok
            and seg_dist <= max(0.030, gate * 0.62)
            and t >= 0.70
            and wrist_dist <= gate * 1.35
        )

        if not (wrist_ok or seg_ok):
            return False, wrist_dist / max(gate, 1e-6), 0.0

        # Normalize based on the better of wrist attachment and forearm attachment.
        norm_wrist = wrist_dist / max(gate, 1e-6)
        norm_seg = seg_dist / max(max(0.030, gate * 0.62), 1e-6) + max(0.0, 0.78 - t) * 0.35
        norm_d = min(norm_wrist, norm_seg)

        # Penalize weird geometry: wrist behind elbow relative to forearm direction.
        forearm = anchor.wrist[:2] - anchor.elbow[:2]
        to_hand = det.wrist[:2] - anchor.elbow[:2]
        if anchor.elbow_ok and _safe_norm(forearm) > 1e-4:
            cosv = float(np.dot(forearm, to_hand) / (_safe_norm(forearm) * _safe_norm(to_hand)))
            if cosv < -0.05:
                norm_d += min(0.35, abs(cosv) * 0.25)

        score = _clip01(1.0 - norm_d)
        return True, norm_d, score

    def _assignment_cost(self, det: HandDetection, physical_label: str, anchors: Dict[str, ArmAnchor]) -> Tuple[float, bool, float]:
        gate_ok, arm_norm_d, link_score = self._arm_link(det, physical_label, anchors)
        if not gate_ok:
            return 1e6, False, 0.0

        # Arm distance is the strongest prior. Temporal/shape only break ties.
        cost = 4.2 * arm_norm_d
        st = self.hands[physical_label]
        if st.valid:
            cost += 0.35 * _safe_norm(det.wrist[:2] - st.landmarks[0, :2])
            prev_rel = _relative_hand(st.landmarks)
            new_rel = _relative_hand(det.landmarks)
            cost += 0.06 * float(np.mean(np.linalg.norm(prev_rel[:, :2] - new_rel[:, :2], axis=1)))
        else:
            cost += 0.12

        cost += self._handedness_cost(det, physical_label)
        cost += 0.04 * (1.0 - _clip01(det.score))
        return float(cost), True, link_score

    def _assign(self, detections: List[HandDetection], anchors: Dict[str, ArmAnchor]) -> Dict[str, Tuple[HandDetection, float]]:
        if not detections:
            return {}

        labels = ["left", "right"]
        scored: Dict[Tuple[int, str], Tuple[float, bool, float]] = {}
        for i, d in enumerate(detections):
            for lab in labels:
                scored[(i, lab)] = self._assignment_cost(d, lab, anchors)

        if len(detections) == 1:
            d = detections[0]
            valid = [(lab, *scored[(0, lab)]) for lab in labels if scored[(0, lab)][1]]
            if not valid:
                return {}
            lab, cost, _ok, link = min(valid, key=lambda x: x[1])
            return {lab: (d, link)}

        d0, d1 = detections[:2]
        assignments: List[Tuple[float, Dict[str, Tuple[HandDetection, float]]]] = []

        c0, ok0, l0 = scored[(0, "left")]
        c1, ok1, l1 = scored[(1, "right")]
        if ok0 and ok1:
            assignments.append((c0 + c1, {"left": (d0, l0), "right": (d1, l1)}))

        c0, ok0, l0 = scored[(0, "right")]
        c1, ok1, l1 = scored[(1, "left")]
        if ok0 and ok1:
            assignments.append((c0 + c1, {"right": (d0, l0), "left": (d1, l1)}))

        if assignments:
            assignments.sort(key=lambda x: x[0])
            return assignments[0][1]

        # If only one detection can be physically attached, update only that side.
        singles: List[Tuple[float, str, HandDetection, float]] = []
        for i, d in enumerate((d0, d1)):
            for lab in labels:
                c, ok, link = scored[(i, lab)]
                if ok:
                    singles.append((c, lab, d, link))
        if not singles:
            return {}
        singles.sort(key=lambda x: x[0])
        _c, lab, det, link = singles[0]
        return {lab: (det, link)}

    def _self_handshake_score(self) -> float:
        lh, rh = self.hands["left"], self.hands["right"]
        if not (lh.valid and rh.valid):
            return 0.0
        wrist_d = _safe_norm(lh.landmarks[0, :2] - rh.landmarks[0, :2])
        palm_d = _safe_norm(lh.landmarks[9, :2] - rh.landmarks[9, :2])
        close = np.exp(-18.0 * min(wrist_d, palm_d))
        arm_lock = 0.5 + 0.5 * min(lh.arm_link_score, rh.arm_link_score)
        return float(np.clip(close * min(lh.quality, rh.quality) * arm_lock, 0.0, 1.0))

    def process(self, frame_bgr: np.ndarray) -> RobustFrameResult:
        t0 = time.perf_counter()
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False

        # In arm-locked mode, pose anchors are critical. pose_every=1 is most stable.
        if self.frame_i % self.pose_every == 0 or self.pose_result is None:
            self.pose_result = self.pose.process(rgb)
        detections = self._detect_hands(rgb)

        rgb.flags.writeable = True
        pose_selected, pose_ok, anchors = self._pose_selected_and_anchors()
        assigned = self._assign(detections, anchors)

        for name, st in self.hands.items():
            if name in assigned:
                det, link_score = assigned[name]
                st.update(
                    det,
                    arm_link_score=link_score,
                    hand_deadband=self.hand_deadband,
                    anchor=anchors.get(name),
                    snap_strength=self.snap_strength,
                    snap_max=self.snap_max,
                )
            else:
                st.miss()

        self.frame_i += 1
        infer_ms = (time.perf_counter() - t0) * 1000.0
        return RobustFrameResult(
            pose_landmarks=self.pose_result.pose_landmarks if self.pose_result else None,
            pose_selected=pose_selected,
            pose_ok=pose_ok,
            arm_anchors=anchors,
            hands=self.hands,
            raw_detections=detections,
            self_handshake_score=self._self_handshake_score(),
            infer_ms=infer_ms,
            arm_gate=self.arm_gate,
        )


def make_tracker(
    model_complexity: int = 0,
    det_conf: float = 0.55,
    track_conf: float = 0.55,
    pose_every: int = 1,
    hold_frames: int = 6,
    arm_hold_frames: int = 8,
    mirror_handedness: bool = True,
    arm_locked: bool = True,
    arm_gate: float = 0.13,
    allow_unanchored_when_pose_missing: bool = False,
    hand_deadband: float = 0.0065,
    pose_deadband: float = 0.0045,
    snap_strength: float = 0.35,
    snap_max: float = 0.030,
) -> RobustHolisticTracker:
    return RobustHolisticTracker(
        model_complexity=model_complexity,
        det_conf=det_conf,
        track_conf=track_conf,
        pose_every=pose_every,
        hold_frames=hold_frames,
        arm_hold_frames=arm_hold_frames,
        mirror_handedness=mirror_handedness,
        arm_locked=arm_locked,
        arm_gate=arm_gate,
        allow_unanchored_when_pose_missing=allow_unanchored_when_pose_missing,
        hand_deadband=hand_deadband,
        pose_deadband=pose_deadband,
        snap_strength=snap_strength,
        snap_max=snap_max,
    )


def _palm_center(lm: np.ndarray) -> np.ndarray:
    """Use stable palm landmarks only: wrist + MCPs."""
    return np.mean(lm[[0, 5, 9, 13, 17]], axis=0).astype(np.float32)


def _shoulder_frame(result: RobustFrameResult) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float, bool]:
    """Return left shoulder, right shoulder, midpoint, scale, shoulder angle, ok."""
    left = None
    right = None
    la = result.arm_anchors.get("left")
    ra = result.arm_anchors.get("right")
    if la is not None and la.shoulder_ok:
        left = la.shoulder
    if ra is not None and ra.shoulder_ok:
        right = ra.shoulder

    # Fallback from compat selected pose: index 0=L shoulder, 1=R shoulder.
    if left is None and result.pose_selected.shape[0] >= 2 and np.any(result.pose_selected[0]):
        left = result.pose_selected[0]
    if right is None and result.pose_selected.shape[0] >= 2 and np.any(result.pose_selected[1]):
        right = result.pose_selected[1]

    if left is None:
        left = np.zeros(3, dtype=np.float32)
    if right is None:
        right = np.zeros(3, dtype=np.float32)

    left = left.astype(np.float32)
    right = right.astype(np.float32)
    mid = ((left + right) * 0.5).astype(np.float32)
    vec = right[:2] - left[:2]
    scale = max(_safe_norm(vec), 0.10)  # normalized shoulder width; fallback prevents explosion.
    angle = float(np.arctan2(vec[1], vec[0])) if scale > 1e-5 else 0.0
    ok = bool(_safe_norm(vec) > 0.03)
    return left, right, mid, float(scale), angle, ok


def _build_compat_features(result: RobustFrameResult) -> np.ndarray:
    feat = np.zeros(179, dtype=np.float32)
    feat[0:18] = result.pose_selected.reshape(-1)

    for label, start, angle_start, flag_idx in (
        ("left", 18, 144, 177),
        ("right", 81, 160, 178),
    ):
        st = result.hands[label]
        if st.valid:
            feat[start:start + 63] = st.landmarks.reshape(-1)
            feat[angle_start:angle_start + 16] = _hand_angles(st.landmarks)
            feat[flag_idx] = st.quality
        else:
            feat[flag_idx] = 0.0

    feat[176] = result.pose_ok
    return feat


def _build_palm_features(result: RobustFrameResult, include_angles: bool = False) -> np.ndarray:
    """Small, BISINDO-friendly features.

    Feature coordinate system:
    - Shoulders define the body anchor.
    - Palm/wrist positions are relative to shoulder midpoint and own shoulder.
    - Values are normalized by shoulder width, so moving closer/farther to camera
      or shifting the body creates less fake motion.

    palm        -> 52 dims
    palm_angles -> 84 dims = palm + 16 left finger angles + 16 right finger angles
    """
    ls, rs, mid, scale, shoulder_angle, shoulder_ok = _shoulder_frame(result)
    inv_scale = 1.0 / max(scale, 1e-6)

    out: List[float] = []

    # 12 global/body anchor features.
    shoulder_vec = (rs - ls) * inv_scale
    out.extend(ls.tolist())                                     # 0..2 absolute normalized L shoulder
    out.extend(rs.tolist())                                     # 3..5 absolute normalized R shoulder
    out.extend(mid.tolist())                                    # 6..8 shoulder midpoint
    out.extend([scale, float(np.sin(shoulder_angle)), float(np.cos(shoulder_angle))])  # 9..11

    palms: Dict[str, np.ndarray] = {}
    wrists: Dict[str, np.ndarray] = {}

    # 16 features per hand. Only palm/wrist position, no full finger coordinates.
    for label, own_shoulder in (("left", ls), ("right", rs)):
        st = result.hands[label]
        if st.valid:
            lm = st.landmarks
            palm = _palm_center(lm)
            wrist = lm[0].astype(np.float32)
            middle_mcp = lm[9].astype(np.float32)
            palm_rel_mid = (palm - mid) * inv_scale
            wrist_rel_mid = (wrist - mid) * inv_scale
            palm_rel_shoulder = (palm - own_shoulder) * inv_scale
            wrist_to_palm = (palm - wrist) * inv_scale
            palm_to_middle = (middle_mcp - palm) * inv_scale
            palms[label] = palm
            wrists[label] = wrist
            out.extend(palm_rel_mid.tolist())        # 3
            out.extend(wrist_rel_mid.tolist())       # 3
            out.extend(palm_rel_shoulder.tolist())   # 3
            out.extend(wrist_to_palm.tolist())       # 3
            out.extend([
                _safe_norm(palm_to_middle[:2]),
                st.quality,
                float(st.detected),
                min(st.motion_ema, 1.0),
            ])                                       # 4 => total 16
        else:
            palms[label] = np.zeros(3, dtype=np.float32)
            wrists[label] = np.zeros(3, dtype=np.float32)
            out.extend([0.0] * 16)

    lh, rh = result.left_hand, result.right_hand
    # 8 inter-hand/meta features.
    if lh.valid and rh.valid:
        lp, rp = palms["left"], palms["right"]
        lw, rw = wrists["left"], wrists["right"]
        palm_delta = (lp - rp) * inv_scale
        out.extend(palm_delta.tolist())
        out.extend([
            _safe_norm((lp - rp)[:2]) * inv_scale,
            _safe_norm((lw - rw)[:2]) * inv_scale,
            result.self_handshake_score,
            min(lh.quality, rh.quality),
            float(shoulder_ok),
        ])
    else:
        out.extend([0.0, 0.0, 0.0, 0.0, 0.0, result.self_handshake_score, min(lh.quality, rh.quality), float(shoulder_ok)])

    arr = np.asarray(out, dtype=np.float32)
    # 12 + 16 + 16 + 8 = 52
    if arr.shape[0] != 52:
        raise RuntimeError(f"palm feature bug: expected 52 dims, got {arr.shape[0]}")

    if include_angles:
        la = _hand_angles(lh.landmarks) if lh.valid else np.zeros(16, dtype=np.float32)
        ra = _hand_angles(rh.landmarks) if rh.valid else np.zeros(16, dtype=np.float32)
        arr = np.concatenate([arr, la, ra]).astype(np.float32)  # 84 dims
    return arr


def extract_features(result: RobustFrameResult, mode: str = "compat") -> np.ndarray:
    """Return feature vector.

    mode="compat" -> 179 dims, same old layout:
        pose 18 + left hand 63 + right hand 63 + left angles 16 + right angles 16 + flags 3.

    mode="stable" -> 337 dims, recommended when you want maximum information:
        compat 179 + left relative hand 63 + right relative hand 63 + geometry 16 + quality/meta 16.

    mode="palm" -> 52 dims, lightweight BISINDO gross-motion features:
        shoulders + palm/wrist positions relative to the shoulder frame + hand quality/meta.
        No full finger landmarks.

    mode="palm_angles" -> 84 dims:
        palm mode + 16 left finger angles + 16 right finger angles. Better for sign language
        because it keeps finger shape without raw full hand coordinates.
    """
    mode = mode.lower().strip()

    if mode == "palm":
        return _build_palm_features(result, include_angles=False)
    if mode in ("palm_angles", "palm-angle", "palmangle"):
        return _build_palm_features(result, include_angles=True)

    feat = _build_compat_features(result)
    if mode == "compat":
        return feat

    if mode != "stable":
        raise ValueError("mode must be 'compat', 'stable', 'palm', or 'palm_angles'")

    left_rel = np.zeros((21, 3), dtype=np.float32)
    right_rel = np.zeros((21, 3), dtype=np.float32)
    lh, rh = result.left_hand, result.right_hand
    if lh.valid:
        left_rel = _relative_hand(lh.landmarks)
    if rh.valid:
        right_rel = _relative_hand(rh.landmarks)

    geom = np.zeros(16, dtype=np.float32)
    if lh.valid and rh.valid:
        geom[0] = _safe_norm(lh.landmarks[0, :2] - rh.landmarks[0, :2])       # wrist distance
        geom[1] = _safe_norm(lh.landmarks[9, :2] - rh.landmarks[9, :2])       # palm distance
        geom[2] = _safe_norm(lh.landmarks[8, :2] - rh.landmarks[8, :2])       # index tip distance
        geom[3] = _safe_norm(lh.landmarks[4, :2] - rh.landmarks[4, :2])       # thumb tip distance
        geom[4] = float(np.dot(left_rel[9, :2], right_rel[9, :2]))
        geom[5] = result.self_handshake_score
        geom[6] = lh.quality
        geom[7] = rh.quality
        geom[8] = float(lh.detected)
        geom[9] = float(rh.detected)
        geom[10] = min(lh.age, 99) / 99.0
        geom[11] = min(rh.age, 99) / 99.0
        geom[12] = _hand_scale(lh.landmarks)
        geom[13] = _hand_scale(rh.landmarks)
        geom[14] = lh.arm_link_score
        geom[15] = rh.arm_link_score

    meta = np.zeros(16, dtype=np.float32)
    meta[0] = result.pose_ok
    meta[1] = lh.quality
    meta[2] = rh.quality
    meta[3] = float(lh.detected)
    meta[4] = float(rh.detected)
    meta[5] = result.self_handshake_score
    meta[6] = min(result.infer_ms, 1000.0) / 1000.0
    meta[7] = lh.arm_link_score
    meta[8] = rh.arm_link_score
    meta[9] = float(lh.locked_to_arm)
    meta[10] = float(rh.locked_to_arm)
    meta[11] = lh.motion_ema
    meta[12] = rh.motion_ema
    meta[13] = lh.snap_offset
    meta[14] = rh.snap_offset

    return np.concatenate([feat, left_rel.reshape(-1), right_rel.reshape(-1), geom, meta]).astype(np.float32)

def _pt(lm: np.ndarray, w: int, h: int) -> Tuple[int, int]:
    return int(np.clip(lm[0] * w, 0, w - 1)), int(np.clip(lm[1] * h, 0, h - 1))


def draw_landmarks(
    frame: np.ndarray,
    result: RobustFrameResult,
    draw_full_pose: bool = False,
    draw_gates: bool = True,
    draw_raw_detections: bool = False,
) -> np.ndarray:
    h, w = frame.shape[:2]

    if draw_full_pose and result.pose_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(
            frame,
            result.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp.solutions.drawing_styles.get_default_pose_landmarks_style(),
        )

    # Draw smoothed arms only. No face landmarks by default.
    for label, color in (("left", (70, 220, 255)), ("right", (255, 170, 70))):
        anchor = result.arm_anchors.get(label)
        if anchor is None:
            continue
        shoulder_pt = _pt(anchor.shoulder, w, h)
        elbow_pt = _pt(anchor.elbow, w, h)
        wrist_pt = _pt(anchor.wrist, w, h)

        if anchor.shoulder_ok and anchor.elbow_ok:
            cv2.line(frame, shoulder_pt, elbow_pt, color, 2, cv2.LINE_AA)
        if anchor.elbow_ok and anchor.wrist_ok:
            cv2.line(frame, elbow_pt, wrist_pt, color, 2, cv2.LINE_AA)

        if anchor.shoulder_ok:
            cv2.circle(frame, shoulder_pt, 4, color, -1, cv2.LINE_AA)
        if anchor.elbow_ok:
            cv2.circle(frame, elbow_pt, 4, color, -1, cv2.LINE_AA)
        if anchor.wrist_ok:
            cv2.circle(frame, wrist_pt, 5, color, -1, cv2.LINE_AA)
            if draw_gates:
                gate_px = int(anchor.gate_radius(result.arm_gate) * min(w, h))
                cv2.circle(frame, wrist_pt, max(7, gate_px), color, 1, cv2.LINE_AA)
            cv2.putText(frame, f"{label[0].upper()} arm", (wrist_pt[0] + 5, wrist_pt[1] + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.34, color, 1, cv2.LINE_AA)

    if draw_raw_detections:
        for det in result.raw_detections:
            p = _pt(det.wrist, w, h)
            cv2.drawMarker(frame, p, (220, 220, 220), markerType=cv2.MARKER_CROSS,
                           markerSize=12, thickness=1, line_type=cv2.LINE_AA)
            cv2.putText(frame, f"raw {det.mp_label} {det.score:.2f}", (p[0] + 4, p[1] - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.34, (220, 220, 220), 1, cv2.LINE_AA)

    for label, color in (("left", (70, 220, 255)), ("right", (255, 170, 70))):
        st = result.hands[label]
        if not st.valid:
            continue
        q = st.quality
        draw_color = tuple(int(c * (0.35 + 0.65 * q)) for c in color)
        pts = [_pt(p, w, h) for p in st.landmarks]

        anchor = result.arm_anchors.get(label)
        if anchor is not None and anchor.wrist_ok:
            cv2.line(frame, _pt(anchor.wrist, w, h), pts[0], draw_color, 2 if st.detected else 1, cv2.LINE_AA)

        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], draw_color, 2 if st.detected else 1, cv2.LINE_AA)
        for i, p in enumerate(pts):
            radius = 3 if i in (0, 4, 8, 12, 16, 20) else 2
            cv2.circle(frame, p, radius, draw_color, -1, cv2.LINE_AA)
        wrist = pts[0]
        tag = "L" if label == "left" else "R"
        suffix = " lock" if st.detected and st.locked_to_arm else " hold" if st.valid else ""
        cv2.putText(
            frame,
            f"{tag} q={q:.2f} a={st.arm_link_score:.2f} m={st.motion_ema:.3f}{suffix}",
            (wrist[0] + 6, wrist[1] - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            draw_color,
            1,
            cv2.LINE_AA,
        )

    return frame


def draw_hud(frame: np.ndarray, result: RobustFrameResult, fps: float, feature_mode: str = "compat") -> np.ndarray:
    h, w = frame.shape[:2]
    lh, rh = result.left_hand, result.right_hand

    overlay = frame.copy()
    cv2.rectangle(overlay, (4, 4), (370, 168), (18, 18, 18), -1)
    cv2.addWeighted(overlay, 0.62, frame, 0.38, 0, frame)

    def col(q: float) -> Tuple[int, int, int]:
        if q >= 0.65:
            return (70, 230, 90)
        if q >= 0.25:
            return (80, 190, 255)
        return (60, 80, 230)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, f"FPS {fps:5.1f} | infer {result.infer_ms:5.1f} ms | {feature_mode}",
                (10, 23), font, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Pose/arms: {'OK' if result.pose_ok else '--'} | ARM-LOCK + anti-jitter",
                (10, 48), font, 0.48, col(result.pose_ok), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Left : q={lh.quality:.2f} arm={lh.arm_link_score:.2f} move={lh.motion_ema:.4f} {'det' if lh.detected else 'hold' if lh.valid else '--'}",
                (10, 72), font, 0.45, col(lh.quality), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Right: q={rh.quality:.2f} arm={rh.arm_link_score:.2f} move={rh.motion_ema:.4f} {'det' if rh.detected else 'hold' if rh.valid else '--'}",
                (10, 96), font, 0.45, col(rh.quality), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Self-handshake/overlap: {result.self_handshake_score:.2f}",
                (10, 120), font, 0.44, col(result.self_handshake_score), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Snap L/R: {lh.snap_offset:.3f}/{rh.snap_offset:.3f} | gate={result.arm_gate:.2f}",
                (10, 144), font, 0.38, (170, 170, 170), 1, cv2.LINE_AA)
    cv2.putText(frame, "Q/ESC quit | G gif | S npy | R reset", (6, h - 6),
                font, 0.38, (145, 145, 145), 1, cv2.LINE_AA)
    return frame
