#!/usr/bin/env python3
"""
Fast live 3D hand + shoulder feature extractor for BISINDO experiments.

What is optimized compared to the Holistic version:
- Uses MediaPipe Pose only for shoulders, not full Holistic.
- Uses MediaPipe Hands for two 21-point 3D hands.
- Pose can run every N frames while shoulders are held between frames.
- Internal processing resolution can be lower than camera/display resolution.
- Optional drawing throttling and lower-fps GIF buffer.
- Keeps the same 268-dim feature layout as the previous script.

Feature layout per frame, dim=268:
  0:6       shoulders relative to shoulder center / shoulder width
  6:69      left hand global 21*xyz relative to shoulder center / shoulder width
  69:132    right hand global 21*xyz relative to shoulder center / shoulder width
  132:195   left hand local 21*xyz relative to wrist / palm scale
  195:258   right hand local 21*xyz relative to wrist / palm scale
  258:268   meta: present L/R, detected L/R, held L/R, shoulder_ok, shoulder visibility L/R, shoulder_scale

Keys:
  Q / ESC : quit
  G       : save recent annotated frames as GIF
  R       : start/stop feature recording to NPZ + CSV
  S       : save one-frame snapshot
  H       : toggle hold-last-good-frame
  D       : toggle drawing overlay
"""

from __future__ import annotations

import argparse
import csv
import json
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, List, Optional, Sequence, Tuple

import cv2
import imageio.v2 as imageio
import mediapipe as mp
import numpy as np


POSE_LEFT_SHOULDER = 11
POSE_RIGHT_SHOULDER = 12
FEATURE_DIM = 268

# MediaPipe hand connection edges, copied into simple tuples at runtime.
# Drawing our own simple skeleton is faster than mp_drawing for live preview.
HAND_CONNECTIONS = tuple(mp.solutions.hands.HAND_CONNECTIONS)


@dataclass
class CandidateHand:
    xyz: np.ndarray                 # (21, 3), normalized MediaPipe coordinates
    label: str = ""                 # anatomical "Left" or "Right" after mirror correction, if available
    score: float = 0.0


@dataclass
class TrackState:
    left: Optional[np.ndarray] = None
    right: Optional[np.ndarray] = None
    left_age: int = 10000
    right_age: int = 10000
    shoulders: np.ndarray = None    # (2, 4): x,y,z,visibility
    shoulder_age: int = 10000
    frame_idx: int = 0

    def __post_init__(self):
        if self.shoulders is None:
            self.shoulders = np.full((2, 4), np.nan, dtype=np.float32)


@dataclass
class FrameFeatures:
    ok: bool
    timestamp: float
    frame_idx: int
    vector: np.ndarray
    left_hand: np.ndarray
    right_hand: np.ndarray
    shoulders: np.ndarray
    present: np.ndarray
    detected: np.ndarray
    held: np.ndarray
    quality: np.ndarray
    pose_ran: bool


class LatestFrameCamera:
    """Small latest-frame reader to reduce camera latency on Jetson/USB webcams."""

    def __init__(self, cap: cv2.VideoCapture):
        self.cap = cap
        self.lock = threading.Lock()
        self.frame: Optional[np.ndarray] = None
        self.ok = False
        self.running = False
        self.thread: Optional[threading.Thread] = None

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        return self

    def _loop(self):
        while self.running:
            ok, frame = self.cap.read()
            if ok:
                with self.lock:
                    self.ok = True
                    self.frame = frame
            else:
                time.sleep(0.002)

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        with self.lock:
            if self.frame is None:
                return False, None
            return self.ok, self.frame.copy()

    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=0.5)


class FastHandShoulderExtractor:
    def __init__(
        self,
        pose_every: int = 3,
        proc_width: int = 416,
        model_complexity_pose: int = 0,
        model_complexity_hands: int = 0,
        det_conf: float = 0.55,
        track_conf: float = 0.60,
        smooth_alpha: float = 0.70,
        hold_frames: int = 8,
        mirror_input: bool = False,
    ):
        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands

        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=int(model_complexity_pose),
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=float(det_conf),
            min_tracking_confidence=float(track_conf),
        )
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=int(model_complexity_hands),
            min_detection_confidence=float(det_conf),
            min_tracking_confidence=float(track_conf),
        )

        self.state = TrackState()
        self.pose_every = max(1, int(pose_every))
        self.proc_width = int(proc_width)
        self.smooth_alpha = float(smooth_alpha)
        self.hold_frames = int(hold_frames)
        self.enable_hold = True
        self.mirror_input = bool(mirror_input)

    def close(self):
        self.pose.close()
        self.hands.close()

    @staticmethod
    def _blank_hand() -> np.ndarray:
        return np.full((21, 3), np.nan, dtype=np.float32)

    @staticmethod
    def _lm_to_np(landmark_list, n: int = 21) -> Optional[np.ndarray]:
        if landmark_list is None:
            return None
        arr = np.empty((n, 3), dtype=np.float32)
        for i, lm in enumerate(landmark_list.landmark[:n]):
            arr[i, 0] = lm.x
            arr[i, 1] = lm.y
            arr[i, 2] = lm.z
        return arr

    @staticmethod
    def _pose_shoulders(pose_landmarks) -> np.ndarray:
        shoulders = np.full((2, 4), np.nan, dtype=np.float32)
        if pose_landmarks is None:
            return shoulders
        lms = pose_landmarks.landmark
        for out_i, idx in enumerate((POSE_LEFT_SHOULDER, POSE_RIGHT_SHOULDER)):
            lm = lms[idx]
            shoulders[out_i] = [lm.x, lm.y, lm.z, getattr(lm, "visibility", 0.0)]
        return shoulders

    @staticmethod
    def _opposite_label(label: str) -> str:
        if label == "Left":
            return "Right"
        if label == "Right":
            return "Left"
        return label

    def _resize_for_processing(self, frame_bgr: np.ndarray) -> np.ndarray:
        if self.proc_width <= 0 or self.proc_width >= frame_bgr.shape[1]:
            return frame_bgr
        h, w = frame_bgr.shape[:2]
        proc_h = int(round(h * (self.proc_width / float(w))))
        return cv2.resize(frame_bgr, (self.proc_width, proc_h), interpolation=cv2.INTER_AREA)

    @staticmethod
    def _valid_prev(hand: Optional[np.ndarray], age: int, hold_frames: int) -> bool:
        return hand is not None and age <= hold_frames and np.isfinite(hand).all()

    @staticmethod
    def _wrist_dist(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.linalg.norm(a[0, :2] - b[0, :2]))

    def _candidate_cost(self, cand: CandidateHand, side: str, shoulders: np.ndarray) -> float:
        """Lower is better. Uses previous wrist first, then shoulder proximity, then handedness."""
        wrist_xy = cand.xyz[0, :2]
        cost = 0.0
        weight_sum = 0.0

        if side == "left" and self._valid_prev(self.state.left, self.state.left_age, self.hold_frames):
            cost += 4.0 * float(np.linalg.norm(wrist_xy - self.state.left[0, :2]))
            weight_sum += 4.0
        elif side == "right" and self._valid_prev(self.state.right, self.state.right_age, self.hold_frames):
            cost += 4.0 * float(np.linalg.norm(wrist_xy - self.state.right[0, :2]))
            weight_sum += 4.0

        shoulder_idx = 0 if side == "left" else 1
        if np.isfinite(shoulders[shoulder_idx, :2]).all():
            cost += 1.0 * float(np.linalg.norm(wrist_xy - shoulders[shoulder_idx, :2]))
            weight_sum += 1.0

        # Small penalty only. Previous track and shoulder proximity are more reliable during overlap.
        if cand.label:
            anatomical = "Left" if side == "left" else "Right"
            if cand.label != anatomical:
                cost += 0.20 * max(0.5, cand.score)
            else:
                cost -= 0.05 * max(0.5, cand.score)

        if weight_sum == 0.0:
            # Last fallback: prefer labels. If no label, neutral cost.
            if cand.label:
                anatomical = "Left" if side == "left" else "Right"
                return 0.0 if cand.label == anatomical else 0.25
            return 0.10
        return cost / weight_sum

    def _assign_hands(self, candidates: Sequence[CandidateHand], shoulders: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if not candidates:
            return None, None

        if len(candidates) == 1:
            cand = candidates[0]
            cl = self._candidate_cost(cand, "left", shoulders)
            cr = self._candidate_cost(cand, "right", shoulders)
            if cl <= cr:
                return cand.xyz, None
            return None, cand.xyz

        # Use only top two by handedness confidence. MediaPipe max_num_hands=2 anyway.
        cands = sorted(candidates, key=lambda c: c.score, reverse=True)[:2]
        a, b = cands[0], cands[1]
        cost_ab = self._candidate_cost(a, "left", shoulders) + self._candidate_cost(b, "right", shoulders)
        cost_ba = self._candidate_cost(b, "left", shoulders) + self._candidate_cost(a, "right", shoulders)
        if cost_ab <= cost_ba:
            return a.xyz, b.xyz
        return b.xyz, a.xyz

    def _swap_guard(self, left: Optional[np.ndarray], right: Optional[np.ndarray]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if left is None or right is None:
            return left, right
        if not self._valid_prev(self.state.left, self.state.left_age, self.hold_frames):
            return left, right
        if not self._valid_prev(self.state.right, self.state.right_age, self.hold_frames):
            return left, right

        direct = self._wrist_dist(left, self.state.left) + self._wrist_dist(right, self.state.right)
        cross = self._wrist_dist(left, self.state.right) + self._wrist_dist(right, self.state.left)
        if cross + 0.030 < direct:
            return right, left
        return left, right

    def _smooth_hold(self, left: Optional[np.ndarray], right: Optional[np.ndarray]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
        detected = np.array([left is not None, right is not None], dtype=np.float32)
        held = np.zeros(2, dtype=np.float32)
        alpha = self.smooth_alpha

        if left is not None:
            if self._valid_prev(self.state.left, self.state.left_age, self.hold_frames):
                left = alpha * left + (1.0 - alpha) * self.state.left
            self.state.left = left.astype(np.float32, copy=False)
            self.state.left_age = 0
        else:
            self.state.left_age += 1
            if self.enable_hold and self._valid_prev(self.state.left, self.state.left_age, self.hold_frames):
                left = self.state.left.copy()
                held[0] = 1.0

        if right is not None:
            if self._valid_prev(self.state.right, self.state.right_age, self.hold_frames):
                right = alpha * right + (1.0 - alpha) * self.state.right
            self.state.right = right.astype(np.float32, copy=False)
            self.state.right_age = 0
        else:
            self.state.right_age += 1
            if self.enable_hold and self._valid_prev(self.state.right, self.state.right_age, self.hold_frames):
                right = self.state.right.copy()
                held[1] = 1.0

        present = np.array([left is not None, right is not None], dtype=np.float32)
        return left, right, present, detected, held

    @staticmethod
    def _feature_vector(left: Optional[np.ndarray], right: Optional[np.ndarray], shoulders: np.ndarray, present: np.ndarray, detected: np.ndarray, held: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        shoulder_ok = bool(np.isfinite(shoulders[:, :3]).all())

        if shoulder_ok:
            ls = shoulders[0, :3]
            rs = shoulders[1, :3]
            anchor = (ls + rs) * 0.5
            scale = float(np.linalg.norm(ls[:2] - rs[:2]))
            scale = max(scale, 1e-4)
            shoulder_rel = np.concatenate(((ls - anchor) / scale, (rs - anchor) / scale)).astype(np.float32)
            shoulder_vis = shoulders[:, 3].astype(np.float32)
        else:
            anchor = np.array([0.5, 0.5, 0.0], dtype=np.float32)
            scale = 0.25
            shoulder_rel = np.zeros(6, dtype=np.float32)
            shoulder_vis = np.zeros(2, dtype=np.float32)

        def one_hand(hand: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
            if hand is None:
                z = np.zeros(63, dtype=np.float32)
                return z, z.copy()
            global_rel = ((hand - anchor) / scale).reshape(-1).astype(np.float32)
            wrist = hand[0]
            palm_scale = float(np.linalg.norm(hand[9, :2] - wrist[:2]))
            palm_scale = max(palm_scale, 1e-4)
            local_rel = ((hand - wrist) / palm_scale).reshape(-1).astype(np.float32)
            return global_rel, local_rel

        left_global, left_local = one_hand(left)
        right_global, right_local = one_hand(right)

        meta = np.array([
            present[0], present[1],
            detected[0], detected[1],
            held[0], held[1],
            1.0 if shoulder_ok else 0.0,
            shoulder_vis[0], shoulder_vis[1],
            scale,
        ], dtype=np.float32)

        vector = np.concatenate((shoulder_rel, left_global, right_global, left_local, right_local, meta)).astype(np.float32)
        assert vector.shape[0] == FEATURE_DIM
        quality = np.array([present[0], present[1], 1.0 if shoulder_ok else 0.0], dtype=np.float32)
        return vector, quality

    def process(self, frame_bgr: np.ndarray, timestamp: float) -> FrameFeatures:
        self.state.frame_idx += 1
        proc = self._resize_for_processing(frame_bgr)
        rgb = cv2.cvtColor(proc, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False

        pose_ran = False
        shoulders = self.state.shoulders
        if self.state.frame_idx == 1 or self.state.frame_idx % self.pose_every == 0 or self.state.shoulder_age > self.pose_every * 4:
            pose_res = self.pose.process(rgb)
            new_shoulders = self._pose_shoulders(pose_res.pose_landmarks)
            if np.isfinite(new_shoulders[:, :2]).any():
                self.state.shoulders = new_shoulders
                self.state.shoulder_age = 0
                shoulders = new_shoulders
            else:
                self.state.shoulder_age += 1
                shoulders = self.state.shoulders
            pose_ran = True
        else:
            self.state.shoulder_age += 1

        hand_res = self.hands.process(rgb)
        candidates: List[CandidateHand] = []
        if hand_res.multi_hand_landmarks:
            handedness = hand_res.multi_handedness or []
            for i, lms in enumerate(hand_res.multi_hand_landmarks):
                xyz = self._lm_to_np(lms, 21)
                if xyz is None:
                    continue
                label = ""
                score = 0.0
                if i < len(handedness) and handedness[i].classification:
                    cls = handedness[i].classification[0]
                    label = cls.label
                    score = float(cls.score)
                    # MediaPipe Hands handedness is designed around selfie-style inputs.
                    # If the frame is not mirrored before processing, anatomical labels need flipping.
                    if not self.mirror_input:
                        label = self._opposite_label(label)
                candidates.append(CandidateHand(xyz=xyz, label=label, score=score))

        left, right = self._assign_hands(candidates, shoulders)
        left, right = self._swap_guard(left, right)
        left, right, present, detected, held = self._smooth_hold(left, right)
        vector, quality = self._feature_vector(left, right, shoulders, present, detected, held)

        return FrameFeatures(
            ok=bool(present[0] or present[1]),
            timestamp=timestamp,
            frame_idx=self.state.frame_idx,
            vector=vector,
            left_hand=left if left is not None else self._blank_hand(),
            right_hand=right if right is not None else self._blank_hand(),
            shoulders=shoulders.copy(),
            present=present,
            detected=detected,
            held=held,
            quality=quality,
            pose_ran=pose_ran,
        )

    def draw(self, frame_bgr: np.ndarray, feat: FrameFeatures, fps: float, recording: bool, overlay: bool = True):
        h, w = frame_bgr.shape[:2]
        if not overlay:
            cv2.putText(frame_bgr, f"FPS {fps:4.1f}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            return

        shoulders = feat.shoulders
        if np.isfinite(shoulders[:, :2]).all():
            p0 = (int(shoulders[0, 0] * w), int(shoulders[0, 1] * h))
            p1 = (int(shoulders[1, 0] * w), int(shoulders[1, 1] * h))
            cv2.line(frame_bgr, p0, p1, (255, 255, 255), 2)
            cv2.circle(frame_bgr, p0, 6, (0, 255, 255), -1)
            cv2.circle(frame_bgr, p1, 6, (255, 0, 255), -1)

        def draw_hand(hand: np.ndarray, color: Tuple[int, int, int], name: str, held: bool):
            if not np.isfinite(hand).all():
                return
            pts = [(int(x * w), int(y * h)) for x, y, _ in hand]
            for a, b in HAND_CONNECTIONS:
                cv2.line(frame_bgr, pts[a], pts[b], color, 2)
            # Draw fewer circles for speed: all fingertips + wrist + MCPs.
            for idx in (0, 1, 2, 5, 9, 13, 17, 4, 8, 12, 16, 20):
                cv2.circle(frame_bgr, pts[idx], 3, color, -1)
            cv2.putText(frame_bgr, name + (" HELD" if held else ""), (pts[0][0] + 7, pts[0][1] + 7), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        draw_hand(feat.left_hand, (0, 255, 0), "LEFT", bool(feat.held[0]))
        draw_hand(feat.right_hand, (0, 128, 255), "RIGHT", bool(feat.held[1]))

        status = (
            f"FPS {fps:4.1f} | dim {FEATURE_DIM} | "
            f"L {int(feat.present[0])}/D{int(feat.detected[0])} R {int(feat.present[1])}/D{int(feat.detected[1])} | "
            f"pose {'RUN' if feat.pose_ran else 'hold'} | hold {'ON' if self.enable_hold else 'OFF'}"
        )
        cv2.putText(frame_bgr, status, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        if recording:
            cv2.putText(frame_bgr, "REC", (10, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 2)
        cv2.putText(frame_bgr, "Q quit | G gif | R record | S snap | H hold | D overlay", (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)


def open_camera(cam: str, width: int, height: int, fps: int, use_gstreamer: bool = False) -> cv2.VideoCapture:
    if use_gstreamer:
        pipeline = (
            f"nvarguscamerasrc sensor-id={cam} ! "
            f"video/x-raw(memory:NVMM), width={width}, height={height}, framerate={fps}/1 ! "
            "nvvidconv flip-method=0 ! video/x-raw, format=BGRx ! "
            "videoconvert ! video/x-raw, format=BGR ! appsink drop=1 max-buffers=1 sync=false"
        )
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    else:
        try:
            cap = cv2.VideoCapture(int(cam), cv2.CAP_V4L2)
        except ValueError:
            cap = cv2.VideoCapture(cam)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
    cap.set(cv2.CAP_PROP_FPS, int(fps))
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    # MJPG often reduces USB webcam decoding/capture latency at 640x480.
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    return cap


def save_gif(buffer: Deque[np.ndarray], out_dir: Path, fps: int) -> Optional[Path]:
    if not buffer:
        print("[GIF] buffer kosong")
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"live_fast_{time.strftime('%Y%m%d_%H%M%S')}.gif"
    imageio.mimsave(str(path), list(buffer), fps=max(1, int(fps)))
    print(f"[GIF] saved: {path}")
    return path


def save_snapshot(feat: FrameFeatures, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"snapshot_fast_{time.strftime('%Y%m%d_%H%M%S')}.npz"
    np.savez_compressed(
        path,
        vector=feat.vector,
        left_hand=feat.left_hand,
        right_hand=feat.right_hand,
        shoulders=feat.shoulders,
        present=feat.present,
        detected=feat.detected,
        held=feat.held,
        quality=feat.quality,
        timestamp=np.array([feat.timestamp], dtype=np.float64),
        frame_idx=np.array([feat.frame_idx], dtype=np.int32),
    )
    print(f"[SNAPSHOT] saved: {path}")
    return path


def save_sequence(records: List[FrameFeatures], out_dir: Path, label: str = "") -> Optional[Path]:
    if not records:
        print("[REC] tidak ada frame untuk disimpan")
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    safe_label = "".join(c for c in label if c.isalnum() or c in ("-", "_")).strip("_")
    prefix = f"seq_fast_{safe_label}_{stamp}" if safe_label else f"seq_fast_{stamp}"
    npz_path = out_dir / f"{prefix}.npz"
    csv_path = out_dir / f"{prefix}.csv"
    meta_path = out_dir / f"{prefix}_meta.json"

    vectors = np.stack([r.vector for r in records]).astype(np.float32)
    left = np.stack([r.left_hand for r in records]).astype(np.float32)
    right = np.stack([r.right_hand for r in records]).astype(np.float32)
    shoulders = np.stack([r.shoulders for r in records]).astype(np.float32)
    present = np.stack([r.present for r in records]).astype(np.float32)
    detected = np.stack([r.detected for r in records]).astype(np.float32)
    held = np.stack([r.held for r in records]).astype(np.float32)
    quality = np.stack([r.quality for r in records]).astype(np.float32)
    timestamps = np.array([r.timestamp for r in records], dtype=np.float64)
    frame_idx = np.array([r.frame_idx for r in records], dtype=np.int32)

    np.savez_compressed(
        npz_path,
        vectors=vectors,
        left_hand=left,
        right_hand=right,
        shoulders=shoulders,
        present=present,
        detected=detected,
        held=held,
        quality=quality,
        timestamp=timestamps,
        frame_idx=frame_idx,
    )

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "frame_idx"] + [f"f_{i:03d}" for i in range(vectors.shape[1])])
        for i in range(vectors.shape[0]):
            writer.writerow([timestamps[i], int(frame_idx[i])] + vectors[i].astype(float).tolist())

    meta = {
        "num_frames": int(vectors.shape[0]),
        "feature_dim": int(vectors.shape[1]),
        "label": label,
        "optimized": True,
        "layout": {
            "0:6": "shoulders relative to shoulder center / shoulder width",
            "6:69": "left hand global 21*xyz relative to shoulder center / shoulder width",
            "69:132": "right hand global 21*xyz relative to shoulder center / shoulder width",
            "132:195": "left hand local 21*xyz relative to wrist / palm scale",
            "195:258": "right hand local 21*xyz relative to wrist / palm scale",
            "258:268": "meta: present L/R, detected L/R, held L/R, shoulder_ok, shoulder visibility L/R, shoulder_scale",
        },
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[REC] saved: {npz_path}")
    print(f"[REC] saved: {csv_path}")
    print(f"[REC] saved: {meta_path}")
    return npz_path


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--cam", default="0", help="camera index/path. USB: 0. CSI with --gstreamer: sensor id, usually 0")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--out", default="runs_hand_shoulder_fast")
    p.add_argument("--label", default="")
    p.add_argument("--gstreamer", action="store_true", help="use Jetson CSI camera pipeline")
    p.add_argument("--mirror", action="store_true", help="flip frame before processing/display, selfie style. Use consistently for dataset")
    p.add_argument("--threaded-cam", action="store_true", help="read latest camera frame in a background thread to reduce latency")

    p.add_argument("--proc-width", type=int, default=416, help="internal MediaPipe width. 0/full=use full 640. Try 384/416/480")
    p.add_argument("--pose-every", type=int, default=3, help="run shoulder Pose every N frames. 1=more accurate, 3=fast default")
    p.add_argument("--pose-complexity", type=int, default=0, choices=[0, 1, 2])
    p.add_argument("--hand-complexity", type=int, default=0, choices=[0, 1])
    p.add_argument("--det-conf", type=float, default=0.55)
    p.add_argument("--track-conf", type=float, default=0.60)
    p.add_argument("--smooth-alpha", type=float, default=0.70)
    p.add_argument("--hold-frames", type=int, default=8)

    p.add_argument("--draw-every", type=int, default=1, help="draw overlay every N frames. 2/3 can increase FPS")
    p.add_argument("--no-overlay", action="store_true", help="disable skeleton overlay for maximum FPS")
    p.add_argument("--gif-seconds", type=float, default=3.0)
    p.add_argument("--gif-fps", type=int, default=8)
    p.add_argument("--gif-width", type=int, default=360, help="GIF frame width. Smaller = lighter")
    return p


def main():
    args = build_argparser().parse_args()
    out_dir = Path(args.out)

    cap = open_camera(args.cam, args.width, args.height, args.fps, args.gstreamer)
    if not cap.isOpened():
        raise RuntimeError("Camera gagal dibuka. Coba --cam 0/1 atau cek permission /dev/video*.")

    cam_reader: Optional[LatestFrameCamera] = None
    if args.threaded_cam:
        cam_reader = LatestFrameCamera(cap).start()
        time.sleep(0.15)

    extractor = FastHandShoulderExtractor(
        pose_every=args.pose_every,
        proc_width=args.proc_width,
        model_complexity_pose=args.pose_complexity,
        model_complexity_hands=args.hand_complexity,
        det_conf=args.det_conf,
        track_conf=args.track_conf,
        smooth_alpha=args.smooth_alpha,
        hold_frames=args.hold_frames,
        mirror_input=args.mirror,
    )

    recording = False
    records: List[FrameFeatures] = []
    gif_buffer: Deque[np.ndarray] = deque(maxlen=max(2, int(args.gif_seconds * args.gif_fps)))
    overlay_enabled = not args.no_overlay
    last_drawn: Optional[np.ndarray] = None

    fps_ema = 0.0
    last_t = time.perf_counter()
    last_gif_push = 0.0

    print("=" * 76)
    print("FAST Live Hand 3D + Shoulder Feature Extractor")
    print(f"camera={args.cam} size={args.width}x{args.height}@{args.fps} proc_width={args.proc_width}")
    print(f"pose_every={args.pose_every} pose_complexity={args.pose_complexity} hand_complexity={args.hand_complexity}")
    print(f"feature_dim={FEATURE_DIM} out={out_dir}")
    print("Keys: Q/ESC quit | G gif | R record | S snapshot | H hold | D overlay")
    print("=" * 76)

    try:
        while True:
            if cam_reader is not None:
                ok, frame = cam_reader.read()
            else:
                ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.002)
                continue

            if frame.shape[1] != args.width or frame.shape[0] != args.height:
                frame = cv2.resize(frame, (args.width, args.height), interpolation=cv2.INTER_AREA)
            if args.mirror:
                frame = cv2.flip(frame, 1)

            now = time.perf_counter()
            dt = max(1e-6, now - last_t)
            last_t = now
            inst_fps = 1.0 / dt
            fps_ema = inst_fps if fps_ema == 0.0 else (0.90 * fps_ema + 0.10 * inst_fps)

            feat = extractor.process(frame, timestamp=time.time())
            if recording:
                records.append(feat)

            draw_this = (feat.frame_idx % max(1, args.draw_every) == 0) or last_drawn is None
            if draw_this:
                vis = frame.copy()
                extractor.draw(vis, feat, fps_ema, recording, overlay=overlay_enabled)
                last_drawn = vis
            else:
                vis = last_drawn.copy()

            # GIF buffer: annotated, downscaled, low fps.
            now_wall = time.time()
            if now_wall - last_gif_push >= 1.0 / max(1, args.gif_fps):
                gif_frame = vis
                if args.gif_width > 0 and args.gif_width < vis.shape[1]:
                    gh = int(round(vis.shape[0] * (args.gif_width / float(vis.shape[1]))))
                    gif_frame = cv2.resize(vis, (args.gif_width, gh), interpolation=cv2.INTER_AREA)
                gif_buffer.append(cv2.cvtColor(gif_frame, cv2.COLOR_BGR2RGB))
                last_gif_push = now_wall

            cv2.imshow("FAST Hand 3D + Shoulders", vis)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), ord("Q"), 27):
                if recording and records:
                    save_sequence(records, out_dir, label=args.label)
                break
            if key in (ord("g"), ord("G")):
                save_gif(gif_buffer, out_dir, args.gif_fps)
            elif key in (ord("s"), ord("S")):
                save_snapshot(feat, out_dir)
            elif key in (ord("h"), ord("H")):
                extractor.enable_hold = not extractor.enable_hold
                print(f"[HOLD] {'ON' if extractor.enable_hold else 'OFF'}")
            elif key in (ord("d"), ord("D")):
                overlay_enabled = not overlay_enabled
                print(f"[OVERLAY] {'ON' if overlay_enabled else 'OFF'}")
            elif key in (ord("r"), ord("R")):
                if not recording:
                    records = []
                    recording = True
                    print("[REC] start")
                else:
                    recording = False
                    save_sequence(records, out_dir, label=args.label)
                    records = []

    finally:
        extractor.close()
        if cam_reader is not None:
            cam_reader.stop()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
