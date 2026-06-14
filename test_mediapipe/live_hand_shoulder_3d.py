#!/usr/bin/env python3
"""
Live 3D hand + shoulder feature extractor for BISINDO-style gesture experiments.

Focus:
- 640x480 live camera test
- 21 3D landmarks per hand + left/right shoulder only
- no elbow/arm features in the exported vector
- lightweight MediaPipe + OpenCV
- GIF export from the recent live buffer
- feature recording to .npz and .csv

Keys:
  Q / ESC : quit
  G       : save recent annotated frames as GIF
  R       : start/stop feature sequence recording
  S       : save one-frame feature snapshot
  H       : toggle hold-last-good-frame for occlusion gaps
"""

import argparse
import csv
import json
import math
import os
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import imageio.v2 as imageio
import mediapipe as mp
import numpy as np


# MediaPipe Pose landmark indices.
POSE_LEFT_SHOULDER = 11
POSE_RIGHT_SHOULDER = 12


@dataclass
class TrackState:
    left: Optional[np.ndarray] = None       # shape (21, 3)
    right: Optional[np.ndarray] = None      # shape (21, 3)
    left_age: int = 10_000
    right_age: int = 10_000
    frame_idx: int = 0


@dataclass
class FrameFeatures:
    ok: bool
    timestamp: float
    frame_idx: int
    vector: np.ndarray                      # flat feature vector
    left_hand: np.ndarray                   # (21, 3), NaN if unavailable
    right_hand: np.ndarray                  # (21, 3), NaN if unavailable
    shoulders: np.ndarray                   # (2, 4) => x,y,z,visibility; NaN if unavailable
    present: np.ndarray                     # [left_detected_or_held, right_detected_or_held]
    detected: np.ndarray                    # [left_detected_this_frame, right_detected_this_frame]
    held: np.ndarray                        # [left_from_hold, right_from_hold]
    quality: np.ndarray                     # rough quality [left,right,shoulder]


class HandShoulderExtractor:
    def __init__(
        self,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.60,
        min_tracking_confidence: float = 0.65,
        smooth_alpha: float = 0.65,
        hold_frames: int = 8,
        use_refiner: bool = True,
        static_image_mode: bool = False,
    ):
        self.mp_holistic = mp.solutions.holistic
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_styles = mp.solutions.drawing_styles

        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            smooth_landmarks=True,
            enable_segmentation=False,
            refine_face_landmarks=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        self.use_refiner = use_refiner
        self.hands_refiner = None
        if use_refiner:
            # Extra full-frame hand detector. It costs more CPU, but helps when Holistic misses one hand.
            self.hands_refiner = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                model_complexity=0,
                min_detection_confidence=max(0.50, min_detection_confidence - 0.10),
                min_tracking_confidence=max(0.50, min_tracking_confidence - 0.10),
            )

        self.state = TrackState()
        self.smooth_alpha = float(smooth_alpha)
        self.hold_frames = int(hold_frames)
        self.enable_hold = True

    def close(self):
        self.holistic.close()
        if self.hands_refiner is not None:
            self.hands_refiner.close()

    @staticmethod
    def _lm_to_np(landmark_list, n: int) -> Optional[np.ndarray]:
        if landmark_list is None:
            return None
        arr = np.zeros((n, 3), dtype=np.float32)
        for i, lm in enumerate(landmark_list.landmark[:n]):
            arr[i] = [lm.x, lm.y, lm.z]
        return arr

    @staticmethod
    def _pose_shoulders(pose_landmarks) -> np.ndarray:
        shoulders = np.full((2, 4), np.nan, dtype=np.float32)
        if pose_landmarks is None:
            return shoulders
        lms = pose_landmarks.landmark
        for out_i, idx in enumerate([POSE_LEFT_SHOULDER, POSE_RIGHT_SHOULDER]):
            lm = lms[idx]
            visibility = getattr(lm, "visibility", 0.0)
            shoulders[out_i] = [lm.x, lm.y, lm.z, visibility]
        return shoulders

    @staticmethod
    def _valid_hand(hand: Optional[np.ndarray]) -> bool:
        return hand is not None and hand.shape == (21, 3) and np.isfinite(hand).all()

    @staticmethod
    def _wrist_dist(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
        if a is None or b is None:
            return 1e9
        return float(np.linalg.norm(a[0, :2] - b[0, :2]))

    @staticmethod
    def _hand_nan() -> np.ndarray:
        return np.full((21, 3), np.nan, dtype=np.float32)

    def _assign_candidates(
        self,
        candidates: List[np.ndarray],
        shoulders: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Assign unlabelled hand candidates to person's left/right hands.

        Priority:
        1. Nearest to previous tracked wrists.
        2. If no previous hand, nearest to left/right shoulder.
        """
        if not candidates:
            return None, None

        prev_l = self.state.left if self.state.left_age <= self.hold_frames else None
        prev_r = self.state.right if self.state.right_age <= self.hold_frames else None

        ls = shoulders[0, :3] if np.isfinite(shoulders[0, :3]).all() else None
        rs = shoulders[1, :3] if np.isfinite(shoulders[1, :3]).all() else None

        def cost_to_side(hand: np.ndarray, side: str) -> float:
            wrist = hand[0, :2]
            if side == "left" and prev_l is not None:
                return float(np.linalg.norm(wrist - prev_l[0, :2]))
            if side == "right" and prev_r is not None:
                return float(np.linalg.norm(wrist - prev_r[0, :2]))
            if side == "left" and ls is not None:
                return float(np.linalg.norm(wrist - ls[:2]))
            if side == "right" and rs is not None:
                return float(np.linalg.norm(wrist - rs[:2]))
            # Fallback without prior/shoulders: person-left is often image-right in non-mirrored camera,
            # but this is weak, so give a neutral cost.
            return 0.5

        if len(candidates) == 1:
            c = candidates[0]
            cl = cost_to_side(c, "left")
            cr = cost_to_side(c, "right")
            return (c, None) if cl <= cr else (None, c)

        # Use the best two candidates. If more than two somehow appear, choose two closest to tracked/shoulder sides.
        candidates = candidates[:2]
        a, b = candidates[0], candidates[1]
        cost_ab = cost_to_side(a, "left") + cost_to_side(b, "right")
        cost_ba = cost_to_side(b, "left") + cost_to_side(a, "right")
        if cost_ab <= cost_ba:
            return a, b
        return b, a

    def _swap_guard(self, left: Optional[np.ndarray], right: Optional[np.ndarray]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Prevent left/right hand ID swap using previous wrist positions."""
        if left is None or right is None:
            return left, right
        if self.state.left is None or self.state.right is None:
            return left, right
        if self.state.left_age > self.hold_frames or self.state.right_age > self.hold_frames:
            return left, right

        direct = self._wrist_dist(left, self.state.left) + self._wrist_dist(right, self.state.right)
        cross = self._wrist_dist(left, self.state.right) + self._wrist_dist(right, self.state.left)

        # Margin avoids random swapping when both hands overlap closely.
        if cross + 0.035 < direct:
            return right, left
        return left, right

    def _smooth_and_hold(
        self,
        left: Optional[np.ndarray],
        right: Optional[np.ndarray],
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
        detected = np.array([left is not None, right is not None], dtype=np.float32)
        held = np.array([0.0, 0.0], dtype=np.float32)

        alpha = self.smooth_alpha

        if left is not None:
            if self.state.left is not None and self.state.left_age <= self.hold_frames:
                left = alpha * left + (1.0 - alpha) * self.state.left
            self.state.left = left.astype(np.float32)
            self.state.left_age = 0
        else:
            self.state.left_age += 1
            if self.enable_hold and self.state.left is not None and self.state.left_age <= self.hold_frames:
                left = self.state.left.copy()
                held[0] = 1.0

        if right is not None:
            if self.state.right is not None and self.state.right_age <= self.hold_frames:
                right = alpha * right + (1.0 - alpha) * self.state.right
            self.state.right = right.astype(np.float32)
            self.state.right_age = 0
        else:
            self.state.right_age += 1
            if self.enable_hold and self.state.right is not None and self.state.right_age <= self.hold_frames:
                right = self.state.right.copy()
                held[1] = 1.0

        present = np.array([left is not None, right is not None], dtype=np.float32)
        return left, right, present, detected, held

    @staticmethod
    def _feature_vector(
        left: Optional[np.ndarray],
        right: Optional[np.ndarray],
        shoulders: np.ndarray,
        present: np.ndarray,
        detected: np.ndarray,
        held: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Build a model-friendly flat vector.

        Coordinate system:
        - Anchor: center point between left/right shoulders.
        - Scale: shoulder width in normalized image coordinates.
        - Global hand coordinates: each landmark relative to shoulder center / shoulder width.
        - Local hand coordinates: each landmark relative to its wrist / palm scale.
        """
        finite_shoulders = np.isfinite(shoulders[:, :3]).all(axis=1)
        shoulder_ok = bool(finite_shoulders.all())

        if shoulder_ok:
            ls = shoulders[0, :3]
            rs = shoulders[1, :3]
            anchor = (ls + rs) / 2.0
            scale = float(np.linalg.norm(ls[:2] - rs[:2]))
            scale = max(scale, 1e-4)
            shoulder_rel = np.concatenate([(ls - anchor) / scale, (rs - anchor) / scale]).astype(np.float32)
            shoulder_vis = shoulders[:, 3].astype(np.float32)
        else:
            anchor = np.array([0.5, 0.5, 0.0], dtype=np.float32)
            scale = 0.25
            shoulder_rel = np.zeros(6, dtype=np.float32)
            shoulder_vis = np.zeros(2, dtype=np.float32)

        def hand_features(hand: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
            if hand is None:
                return np.zeros(63, dtype=np.float32), np.zeros(63, dtype=np.float32)
            global_rel = ((hand - anchor) / scale).reshape(-1).astype(np.float32)

            wrist = hand[0]
            # Palm scale: wrist to middle-finger MCP (landmark 9). Fallback to shoulder scale.
            palm_scale = float(np.linalg.norm(hand[9, :2] - wrist[:2]))
            palm_scale = max(palm_scale, 1e-4)
            local_rel = ((hand - wrist) / palm_scale).reshape(-1).astype(np.float32)
            return global_rel, local_rel

        left_global, left_local = hand_features(left)
        right_global, right_local = hand_features(right)

        meta = np.array([
            present[0], present[1],
            detected[0], detected[1],
            held[0], held[1],
            1.0 if shoulder_ok else 0.0,
            shoulder_vis[0], shoulder_vis[1],
            scale,
        ], dtype=np.float32)

        # Vector layout:
        # 0..5     shoulder_rel
        # 6..68    left_hand_global
        # 69..131  right_hand_global
        # 132..194 left_hand_local
        # 195..257 right_hand_local
        # 258..267 meta
        vector = np.concatenate([
            shoulder_rel,
            left_global,
            right_global,
            left_local,
            right_local,
            meta,
        ]).astype(np.float32)
        return vector, np.array([present[0], present[1], 1.0 if shoulder_ok else 0.0], dtype=np.float32)

    def process(self, frame_bgr: np.ndarray, timestamp: float) -> Tuple[FrameFeatures, object]:
        self.state.frame_idx += 1

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        hol = self.holistic.process(rgb)

        shoulders = self._pose_shoulders(hol.pose_landmarks)
        left = self._lm_to_np(hol.left_hand_landmarks, 21)
        right = self._lm_to_np(hol.right_hand_landmarks, 21)

        # Optional second detector to recover missed hand under overlap/crossing.
        if self.hands_refiner is not None:
            ref = self.hands_refiner.process(rgb)
            candidates = []
            if ref.multi_hand_landmarks:
                for hand_lms in ref.multi_hand_landmarks:
                    c = self._lm_to_np(hand_lms, 21)
                    if c is not None:
                        candidates.append(c)
            cand_l, cand_r = self._assign_candidates(candidates, shoulders)

            # Fill missing or obviously worse-than-prior Holistic result.
            if cand_l is not None:
                if left is None:
                    left = cand_l
                elif self.state.left is not None and self.state.left_age <= self.hold_frames:
                    if self._wrist_dist(cand_l, self.state.left) + 0.04 < self._wrist_dist(left, self.state.left):
                        left = cand_l
            if cand_r is not None:
                if right is None:
                    right = cand_r
                elif self.state.right is not None and self.state.right_age <= self.hold_frames:
                    if self._wrist_dist(cand_r, self.state.right) + 0.04 < self._wrist_dist(right, self.state.right):
                        right = cand_r

        left, right = self._swap_guard(left, right)
        left, right, present, detected, held = self._smooth_and_hold(left, right)

        vector, quality = self._feature_vector(left, right, shoulders, present, detected, held)

        features = FrameFeatures(
            ok=bool(present[0] or present[1]),
            timestamp=timestamp,
            frame_idx=self.state.frame_idx,
            vector=vector,
            left_hand=left if left is not None else self._hand_nan(),
            right_hand=right if right is not None else self._hand_nan(),
            shoulders=shoulders,
            present=present,
            detected=detected,
            held=held,
            quality=quality,
        )
        return features, hol

    def draw(self, frame_bgr: np.ndarray, hol, features: FrameFeatures, fps: float, is_recording: bool):
        # Draw pose shoulders only.
        h, w = frame_bgr.shape[:2]
        shoulders = features.shoulders
        if np.isfinite(shoulders[:, :2]).all():
            pts = []
            for p in shoulders[:, :2]:
                pts.append((int(p[0] * w), int(p[1] * h)))
            cv2.line(frame_bgr, pts[0], pts[1], (255, 255, 255), 2)
            cv2.circle(frame_bgr, pts[0], 7, (0, 255, 255), -1)
            cv2.circle(frame_bgr, pts[1], 7, (255, 0, 255), -1)
            cv2.putText(frame_bgr, "L shoulder", (pts[0][0] + 6, pts[0][1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            cv2.putText(frame_bgr, "R shoulder", (pts[1][0] + 6, pts[1][1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 1)

        # Draw smoothed/held hands, not raw MediaPipe only.
        def draw_hand(hand: np.ndarray, color: Tuple[int, int, int], label: str, held: bool):
            if not np.isfinite(hand).all():
                return
            pts = [(int(x * w), int(y * h)) for x, y, _ in hand]
            connections = self.mp_hands.HAND_CONNECTIONS
            for a, b in connections:
                cv2.line(frame_bgr, pts[a], pts[b], color, 2)
            for p in pts:
                cv2.circle(frame_bgr, p, 3, color, -1)
            suffix = " HELD" if held else ""
            cv2.putText(frame_bgr, label + suffix, (pts[0][0] + 8, pts[0][1] + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        draw_hand(features.left_hand, (0, 255, 0), "LEFT", bool(features.held[0]))
        draw_hand(features.right_hand, (0, 128, 255), "RIGHT", bool(features.held[1]))

        status = f"FPS {fps:4.1f} | L {int(features.present[0])}/D{int(features.detected[0])} R {int(features.present[1])}/D{int(features.detected[1])} | hold {'ON' if self.enable_hold else 'OFF'}"
        cv2.putText(frame_bgr, status, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2)
        if is_recording:
            cv2.putText(frame_bgr, "REC", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.80, (0, 0, 255), 2)
        cv2.putText(frame_bgr, "Q quit | G gif | R record npz/csv | S snapshot | H hold", (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)


def open_camera(cam: str, width: int, height: int, fps: int, use_gstreamer: bool = False):
    if use_gstreamer:
        # Good for some CSI cameras on Jetson. For USB webcam, use normal --cam 0.
        pipeline = (
            f"nvarguscamerasrc sensor-id={cam} ! "
            f"video/x-raw(memory:NVMM), width={width}, height={height}, framerate={fps}/1 ! "
            "nvvidconv flip-method=0 ! video/x-raw, format=BGRx ! "
            "videoconvert ! video/x-raw, format=BGR ! appsink drop=1 max-buffers=1 sync=false"
        )
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    else:
        try:
            cam_idx = int(cam)
            cap = cv2.VideoCapture(cam_idx, cv2.CAP_V4L2)
        except ValueError:
            cap = cv2.VideoCapture(cam)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def save_gif(buffer: deque, out_dir: Path, fps: int):
    if not buffer:
        print("[GIF] buffer kosong")
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = out_dir / f"live_hand_shoulder_{time.strftime('%Y%m%d_%H%M%S')}.gif"
    frames_rgb = list(buffer)
    imageio.mimsave(str(filename), frames_rgb, fps=max(1, int(fps)))
    print(f"[GIF] saved: {filename}")
    return filename


def save_snapshot(feat: FrameFeatures, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = out_dir / f"snapshot_{time.strftime('%Y%m%d_%H%M%S')}.npz"
    np.savez_compressed(
        filename,
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
    print(f"[SNAPSHOT] saved: {filename}")
    return filename


def save_sequence(records: List[FrameFeatures], out_dir: Path, label: str = ""):
    if not records:
        print("[REC] tidak ada frame untuk disimpan")
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime('%Y%m%d_%H%M%S')
    safe_label = ''.join(c for c in label if c.isalnum() or c in ('-', '_')).strip('_')
    prefix = f"seq_{safe_label}_{stamp}" if safe_label else f"seq_{stamp}"
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
        header = ["timestamp", "frame_idx"] + [f"f_{i:03d}" for i in range(vectors.shape[1])]
        writer.writerow(header)
        for i, r in enumerate(records):
            writer.writerow([timestamps[i], int(frame_idx[i])] + vectors[i].astype(float).tolist())

    meta = {
        "label": label,
        "num_frames": int(vectors.shape[0]),
        "feature_dim": int(vectors.shape[1]),
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cam", default="0", help="camera index/path. USB webcam: 0. CSI with --gstreamer: sensor id, usually 0")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--out", default="runs_hand_shoulder", help="output directory for gif/npz/csv")
    parser.add_argument("--label", default="", help="optional label prefix for recorded sequences")
    parser.add_argument("--model-complexity", type=int, default=1, choices=[0, 1, 2])
    parser.add_argument("--det-conf", type=float, default=0.60)
    parser.add_argument("--track-conf", type=float, default=0.65)
    parser.add_argument("--smooth-alpha", type=float, default=0.65, help="EMA new-frame weight. Higher = more responsive, lower = smoother")
    parser.add_argument("--hold-frames", type=int, default=8, help="reuse last good hand for this many frames when occluded")
    parser.add_argument("--gif-seconds", type=float, default=4.0)
    parser.add_argument("--gif-fps", type=int, default=12)
    parser.add_argument("--no-refiner", action="store_true", help="disable second full-frame hand detector for lighter CPU")
    parser.add_argument("--gstreamer", action="store_true", help="use Jetson CSI GStreamer pipeline instead of V4L2")
    parser.add_argument("--mirror", action="store_true", help="mirror display and processing like selfie camera. Use consistently for dataset")
    args = parser.parse_args()

    out_dir = Path(args.out)
    cap = open_camera(args.cam, args.width, args.height, args.fps, args.gstreamer)
    if not cap.isOpened():
        raise RuntimeError("Camera gagal dibuka. Coba --cam 0, --cam 1, atau cek permission /dev/video*." )

    extractor = HandShoulderExtractor(
        model_complexity=args.model_complexity,
        min_detection_confidence=args.det_conf,
        min_tracking_confidence=args.track_conf,
        smooth_alpha=args.smooth_alpha,
        hold_frames=args.hold_frames,
        use_refiner=not args.no_refiner,
    )

    gif_maxlen = max(2, int(args.gif_seconds * args.gif_fps))
    gif_buffer = deque(maxlen=gif_maxlen)
    recording = False
    records: List[FrameFeatures] = []

    last_t = time.perf_counter()
    fps_ema = 0.0

    print("=" * 72)
    print("Live Hand 3D + Shoulder Feature Extractor")
    print(f"camera={args.cam} size={args.width}x{args.height} fps={args.fps} refiner={not args.no_refiner}")
    print("Keys: Q/ESC quit | G save GIF | R record features | S snapshot | H hold on/off")
    print("=" * 72)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[WARN] frame kosong dari kamera")
                continue

            frame = cv2.resize(frame, (args.width, args.height), interpolation=cv2.INTER_LINEAR)
            if args.mirror:
                frame = cv2.flip(frame, 1)

            now = time.perf_counter()
            dt = max(1e-6, now - last_t)
            last_t = now
            inst_fps = 1.0 / dt
            fps_ema = inst_fps if fps_ema == 0.0 else (0.90 * fps_ema + 0.10 * inst_fps)

            feat, hol = extractor.process(frame, timestamp=time.time())
            if recording:
                records.append(feat)

            vis = frame.copy()
            extractor.draw(vis, hol, feat, fps=fps_ema, is_recording=recording)

            # GIF buffer stores RGB annotated frames at lower fps by frame interval.
            if len(gif_buffer) == 0 or (time.time() - getattr(save_gif, "_last_push", 0.0)) >= (1.0 / max(1, args.gif_fps)):
                gif_buffer.append(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
                save_gif._last_push = time.time()

            cv2.imshow("Live Hand 3D + Shoulders", vis)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord('q'), ord('Q'), 27):
                if recording and records:
                    save_sequence(records, out_dir, label=args.label)
                break
            elif key in (ord('g'), ord('G')):
                save_gif(gif_buffer, out_dir, fps=args.gif_fps)
            elif key in (ord('s'), ord('S')):
                save_snapshot(feat, out_dir)
            elif key in (ord('h'), ord('H')):
                extractor.enable_hold = not extractor.enable_hold
                print(f"[HOLD] {'ON' if extractor.enable_hold else 'OFF'}")
            elif key in (ord('r'), ord('R')):
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
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
