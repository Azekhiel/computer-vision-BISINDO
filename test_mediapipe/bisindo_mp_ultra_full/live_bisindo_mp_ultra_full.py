#!/usr/bin/env python3
"""
live_bisindo_mp_ultra_full.py
================================
Pure MediaPipe live feature extractor optimized for Jetson/CPU.

Goal:
- Keep MediaPipe Hands as the only per-frame heavy model.
- Make everything else optional or rare: pose shoulders, overlay, GIF, recording.
- Latest-frame camera reader to avoid accumulated latency.
- Multiple feature modes for BISINDO experiments:
    84, 179, 228, 268, 288, btj_global, btj_local, btj_global_local

Core idea:
- Important body context: shoulders only.
- Important hand context: palm/wrist/fingers.
- Full landmark modes are available when accuracy needs more detail.

Keys:
  Q / ESC : quit
  R       : start/stop feature recording
  S       : save single feature snapshot
  O       : toggle overlay skeleton
  H       : toggle hold last good hand
  X       : swap left/right assignment labels
  G       : save GIF only if --enable-gif-buffer is used
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import mediapipe as mp

try:
    import imageio.v2 as imageio
except Exception:  # pragma: no cover
    imageio = None

mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
HAND_CONNECTIONS = tuple(mp_hands.HAND_CONNECTIONS)

POSE_LEFT_SHOULDER = 11
POSE_RIGHT_SHOULDER = 12

FEATURE_DIMS: Dict[str, int] = {
    "84": 84,
    "179": 179,
    "228": 228,
    "268": 268,
    "288": 288,
    "btj_global": 114,
    "btj_local": 114,
    "btj_global_local": 180,
}

ANGLE_TRIPLETS = [
    (5, 0, 1), (0, 1, 2), (1, 2, 3), (2, 3, 4),        # thumb, 4
    (0, 5, 6), (5, 6, 7), (6, 7, 8),                   # index, 3
    (0, 9, 10), (9, 10, 11), (10, 11, 12),             # middle, 3
    (0, 13, 14), (13, 14, 15), (14, 15, 16),           # ring, 3
    (0, 17, 18), (17, 18, 19), (18, 19, 20),           # pinky, 3
]  # total 16

# Selected points for bahu + telapak + jari modes.
# Includes wrist + palm center + important finger MCP/tips.
BTJ_SELECTED_KIND = [
    "wrist",
    "palm_center",
    "thumb_tip",
    "index_mcp",
    "index_tip",
    "middle_mcp",
    "middle_tip",
    "ring_mcp",
    "ring_tip",
    "pinky_mcp",
    "pinky_tip",
]  # 11 points x 3 = 33 per hand

# 10 selected local points for 228 mode. Wrist local is always zero, so omit it.
COMPACT_LOCAL_KIND = [
    "palm_center",
    "thumb_tip",
    "index_mcp",
    "index_tip",
    "middle_mcp",
    "middle_tip",
    "ring_mcp",
    "ring_tip",
    "pinky_mcp",
    "pinky_tip",
]  # 10 points x 3 = 30 per hand


@dataclass
class CandidateHand:
    xyz: np.ndarray  # (21,3), normalized coords in processed/cropped image
    label: str = ""  # MediaPipe label: Left / Right
    score: float = 0.0


@dataclass
class HandTrack:
    xyz: Optional[np.ndarray] = None
    age: int = 10000
    detected: bool = False
    held: bool = False
    score: float = 0.0


@dataclass
class FrameResult:
    vector: np.ndarray
    left: Optional[np.ndarray]
    right: Optional[np.ndarray]
    shoulders: np.ndarray  # (2,4) x,y,z,visibility
    detected: np.ndarray   # [L,R]
    held: np.ndarray       # [L,R]
    present: np.ndarray    # [L,R]
    scores: np.ndarray     # [L,R]
    fps_infer: float
    hand_ms: float
    pose_ms: float
    feature_mode: str


class LatestFrameCamera:
    """Camera reader that always serves the newest frame.

    This prevents visual lag when inference is slower than camera FPS.
    """

    def __init__(
        self,
        src: int = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        use_csi: bool = False,
        use_gstreamer: bool = True,
        fourcc: str = "MJPG",
    ) -> None:
        self.src = src
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.use_csi = bool(use_csi)
        self.use_gstreamer = bool(use_gstreamer)
        self.fourcc = fourcc.upper()

        self.cap = self._open()
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera src={src}")

        self.lock = threading.Lock()
        self.frame: Optional[np.ndarray] = None
        self.ret = False
        self.running = False
        self.thread: Optional[threading.Thread] = None

    def _open(self) -> cv2.VideoCapture:
        if self.use_gstreamer:
            if self.use_csi:
                gst = (
                    f"nvarguscamerasrc sensor-id={self.src} ! "
                    f"video/x-raw(memory:NVMM), width={self.width}, height={self.height}, "
                    f"format=NV12, framerate={self.fps}/1 ! "
                    f"nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! "
                    f"video/x-raw, format=BGR ! appsink max-buffers=1 drop=true sync=false"
                )
                cap = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
                if cap.isOpened():
                    return cap
            else:
                # Prefer MJPEG for USB webcams, because 640x480@30 is often more stable.
                if self.fourcc == "MJPG":
                    gst = (
                        f"v4l2src device=/dev/video{self.src} ! "
                        f"image/jpeg, width={self.width}, height={self.height}, framerate={self.fps}/1 ! "
                        f"jpegdec ! videoconvert ! video/x-raw, format=BGR ! "
                        f"appsink max-buffers=1 drop=true sync=false"
                    )
                    cap = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
                    if cap.isOpened():
                        return cap
                gst = (
                    f"v4l2src device=/dev/video{self.src} ! "
                    f"video/x-raw, width={self.width}, height={self.height}, framerate={self.fps}/1 ! "
                    f"videoconvert ! video/x-raw, format=BGR ! "
                    f"appsink max-buffers=1 drop=true sync=false"
                )
                cap = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
                if cap.isOpened():
                    return cap

        print("[WARN] GStreamer camera open failed/disabled, fallback CAP_V4L2")
        cap = cv2.VideoCapture(self.src, cv2.CAP_V4L2)
        if self.fourcc:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.fourcc[:4]))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def start(self) -> "LatestFrameCamera":
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        t0 = time.perf_counter()
        while self.frame is None and time.perf_counter() - t0 < 2.0:
            time.sleep(0.01)
        return self

    def _loop(self) -> None:
        while self.running:
            ret, frame = self.cap.read()
            if ret and frame is not None:
                with self.lock:
                    self.ret = True
                    self.frame = frame
            else:
                time.sleep(0.003)

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        with self.lock:
            if not self.ret or self.frame is None:
                return False, None
            return True, self.frame.copy()

    def release(self) -> None:
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        self.cap.release()


def center_crop(frame: np.ndarray, ratio: float) -> np.ndarray:
    ratio = float(ratio)
    if ratio >= 0.999:
        return frame
    h, w = frame.shape[:2]
    nw, nh = int(w * ratio), int(h * ratio)
    x0 = max(0, (w - nw) // 2)
    y0 = max(0, (h - nh) // 2)
    return frame[y0:y0 + nh, x0:x0 + nw]


def resize_width(frame: np.ndarray, width: int) -> np.ndarray:
    if width <= 0 or frame.shape[1] == width:
        return frame
    h, w = frame.shape[:2]
    scale = width / float(w)
    return cv2.resize(frame, (width, int(round(h * scale))), interpolation=cv2.INTER_AREA)


def safe_norm(v: np.ndarray, eps: float = 1e-6) -> float:
    return float(max(np.linalg.norm(v), eps))


def lm_to_np(lm_list, n: int = 21) -> Optional[np.ndarray]:
    if lm_list is None:
        return None
    arr = np.zeros((n, 3), dtype=np.float32)
    for i, p in enumerate(lm_list.landmark[:n]):
        arr[i] = (p.x, p.y, p.z)
    return arr


def pose_shoulders(pose_landmarks) -> np.ndarray:
    out = np.full((2, 4), np.nan, dtype=np.float32)
    if pose_landmarks is None:
        return out
    lms = pose_landmarks.landmark
    for row, idx in enumerate((POSE_LEFT_SHOULDER, POSE_RIGHT_SHOULDER)):
        lm = lms[idx]
        out[row] = (lm.x, lm.y, lm.z, float(getattr(lm, "visibility", 0.0) or 0.0))
    return out


def blank_shoulders() -> np.ndarray:
    # Fixed shoulder anchor for speed. Works as a stable normalization reference.
    return np.array([
        [0.36, 0.43, 0.0, 1.0],
        [0.64, 0.43, 0.0, 1.0],
    ], dtype=np.float32)


def angle_at_b(pa: np.ndarray, pb: np.ndarray, pc: np.ndarray) -> float:
    ba = pa - pb
    bc = pc - pb
    denom = safe_norm(ba) * safe_norm(bc)
    c = float(np.dot(ba, bc) / denom)
    return float(np.arccos(np.clip(c, -1.0, 1.0)))


def hand_angles(hand: Optional[np.ndarray]) -> np.ndarray:
    if hand is None:
        return np.zeros(16, dtype=np.float32)
    vals = np.zeros(16, dtype=np.float32)
    for i, (a, b, c) in enumerate(ANGLE_TRIPLETS):
        vals[i] = angle_at_b(hand[a], hand[b], hand[c])
    return vals


def hand_scale(hand: Optional[np.ndarray]) -> float:
    if hand is None:
        return 1.0
    # wrist -> middle MCP and index MCP -> pinky MCP are fairly stable.
    return max(
        safe_norm(hand[0, :2] - hand[9, :2]),
        safe_norm(hand[5, :2] - hand[17, :2]),
        1e-4,
    )


def palm_center(hand: np.ndarray) -> np.ndarray:
    return np.mean(hand[[0, 5, 9, 13, 17]], axis=0).astype(np.float32)


def palm_normal(hand: np.ndarray) -> np.ndarray:
    # Approximate 3D palm orientation from wrist-index-pinky plane.
    v1 = hand[5] - hand[0]
    v2 = hand[17] - hand[0]
    n = np.cross(v1, v2).astype(np.float32)
    denom = safe_norm(n)
    return (n / denom).astype(np.float32)


def selected_points(hand: Optional[np.ndarray], kinds: Sequence[str]) -> np.ndarray:
    if hand is None:
        return np.zeros((len(kinds), 3), dtype=np.float32)
    pc = palm_center(hand)
    mapping = {
        "wrist": hand[0],
        "palm_center": pc,
        "thumb_tip": hand[4],
        "index_mcp": hand[5],
        "index_tip": hand[8],
        "middle_mcp": hand[9],
        "middle_tip": hand[12],
        "ring_mcp": hand[13],
        "ring_tip": hand[16],
        "pinky_mcp": hand[17],
        "pinky_tip": hand[20],
    }
    return np.stack([mapping[k] for k in kinds]).astype(np.float32)


def shoulder_anchor_scale(shoulders: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray, float]:
    if shoulders is not None and np.isfinite(shoulders[:, :3]).all():
        ls = shoulders[0, :3].astype(np.float32)
        rs = shoulders[1, :3].astype(np.float32)
        anchor = ((ls + rs) * 0.5).astype(np.float32)
        scale = max(safe_norm(ls[:2] - rs[:2]), 1e-4)
        shoulder_rel = np.concatenate(((ls - anchor) / scale, (rs - anchor) / scale)).astype(np.float32)
        ok = 1.0
        return anchor, scale, shoulder_rel, ok
    anchor = np.array([0.5, 0.43, 0.0], dtype=np.float32)
    scale = 0.28
    return anchor, scale, np.zeros(6, dtype=np.float32), 0.0


def full_global(hand: Optional[np.ndarray], anchor: np.ndarray, scale: float) -> np.ndarray:
    if hand is None:
        return np.zeros(63, dtype=np.float32)
    return ((hand - anchor) / scale).reshape(-1).astype(np.float32)


def full_local(hand: Optional[np.ndarray]) -> np.ndarray:
    if hand is None:
        return np.zeros(63, dtype=np.float32)
    s = hand_scale(hand)
    return ((hand - hand[0]) / s).reshape(-1).astype(np.float32)


def selected_global(hand: Optional[np.ndarray], anchor: np.ndarray, scale: float, kinds: Sequence[str]) -> np.ndarray:
    pts = selected_points(hand, kinds)
    return ((pts - anchor) / scale).reshape(-1).astype(np.float32)


def selected_local(hand: Optional[np.ndarray], kinds: Sequence[str]) -> np.ndarray:
    if hand is None:
        return np.zeros(len(kinds) * 3, dtype=np.float32)
    pts = selected_points(hand, kinds)
    s = hand_scale(hand)
    return ((pts - hand[0]) / s).reshape(-1).astype(np.float32)


def geometry10(
    hand: Optional[np.ndarray],
    present: float,
    detected: float,
    held: float,
    score: float,
    shoulder_scale: float,
) -> np.ndarray:
    if hand is None:
        return np.array([present, detected, held, score, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    xy = hand[:, :2]
    mn = xy.min(axis=0)
    mx = xy.max(axis=0)
    bbox_w = float(mx[0] - mn[0])
    bbox_h = float(mx[1] - mn[1])
    bbox_area = bbox_w * bbox_h
    palm_s = hand_scale(hand)
    scale_vs_shoulder = palm_s / max(float(shoulder_scale), 1e-4)
    # Pseudo-z: larger hands = closer to camera. This is relative, not real depth.
    pseudo_z = scale_vs_shoulder
    return np.array([
        present, detected, held, score,
        bbox_w, bbox_h, bbox_area,
        palm_s, scale_vs_shoulder, pseudo_z,
    ], dtype=np.float32)


def palm_descriptor16(
    hand: Optional[np.ndarray],
    anchor: np.ndarray,
    shoulder_scale: float,
    present: float,
    score: float,
) -> np.ndarray:
    if hand is None:
        return np.zeros(16, dtype=np.float32)
    pc = palm_center(hand)
    s = hand_scale(hand)
    wrist_rel = (hand[0] - anchor) / shoulder_scale      # 3
    palm_rel = (pc - anchor) / shoulder_scale            # 3
    normal = palm_normal(hand)                           # 3
    wrist_to_palm = (pc - hand[0]) / max(s, 1e-4)         # 3
    scale_vs_shoulder = np.array([s / max(shoulder_scale, 1e-4)], dtype=np.float32)  # 1
    pseudo_z = np.array([s / max(shoulder_scale, 1e-4)], dtype=np.float32)           # 1
    flags = np.array([present, score], dtype=np.float32) # 2
    return np.concatenate((wrist_rel, palm_rel, normal, wrist_to_palm, scale_vs_shoulder, pseudo_z, flags)).astype(np.float32)


def build_feature(
    mode: str,
    left: Optional[np.ndarray],
    right: Optional[np.ndarray],
    shoulders: np.ndarray,
    present: np.ndarray,
    detected: np.ndarray,
    held: np.ndarray,
    scores: np.ndarray,
) -> np.ndarray:
    anchor, shoulder_scale, shoulder_rel, shoulder_ok = shoulder_anchor_scale(shoulders)
    l_ang = hand_angles(left)
    r_ang = hand_angles(right)

    if mode == "84":
        l_desc = palm_descriptor16(left, anchor, shoulder_scale, float(present[0]), float(scores[0]))
        r_desc = palm_descriptor16(right, anchor, shoulder_scale, float(present[1]), float(scores[1]))
        if left is not None and right is not None:
            lpc, rpc = palm_center(left), palm_center(right)
            pair = np.array([
                *((lpc - rpc) / shoulder_scale).tolist(),
                safe_norm(left[0, :2] - right[0, :2]) / shoulder_scale,
                safe_norm(lpc[:2] - rpc[:2]) / shoulder_scale,
                1.0 if safe_norm(lpc[:2] - rpc[:2]) < 0.20 else 0.0,
            ], dtype=np.float32)
        else:
            pair = np.zeros(6, dtype=np.float32)
        meta8 = np.array([
            shoulder_ok,
            shoulder_scale,
            detected[0], detected[1],
            held[0], held[1],
            scores[0], scores[1],
        ], dtype=np.float32)
        v = np.concatenate((shoulder_rel, l_desc, r_desc, pair, meta8, l_ang, r_ang)).astype(np.float32)

    elif mode == "179":
        meta15 = np.array([
            present[0], present[1],
            detected[0], detected[1],
            held[0], held[1],
            shoulder_ok,
            float(shoulders[0, 3]) if np.isfinite(shoulders[0, 3]) else 0.0,
            float(shoulders[1, 3]) if np.isfinite(shoulders[1, 3]) else 0.0,
            shoulder_scale,
            scores[0], scores[1],
            hand_scale(left) if left is not None else 0.0,
            hand_scale(right) if right is not None else 0.0,
            1.0 if (left is not None and right is not None and safe_norm(palm_center(left)[:2] - palm_center(right)[:2]) < 0.20) else 0.0,
        ], dtype=np.float32)
        v = np.concatenate((
            shoulder_rel,
            full_global(left, anchor, shoulder_scale),
            full_global(right, anchor, shoulder_scale),
            l_ang, r_ang,
            meta15,
        )).astype(np.float32)

    elif mode == "228":
        meta4 = np.array([shoulder_ok, present[0], present[1], shoulder_scale], dtype=np.float32)
        v = np.concatenate((
            shoulder_rel,
            full_global(left, anchor, shoulder_scale),
            full_global(right, anchor, shoulder_scale),
            selected_local(left, COMPACT_LOCAL_KIND),
            selected_local(right, COMPACT_LOCAL_KIND),
            l_ang, r_ang,
            meta4,
        )).astype(np.float32)

    elif mode == "268":
        meta10 = np.array([
            present[0], present[1],
            detected[0], detected[1],
            held[0], held[1],
            shoulder_ok,
            float(shoulders[0, 3]) if np.isfinite(shoulders[0, 3]) else 0.0,
            float(shoulders[1, 3]) if np.isfinite(shoulders[1, 3]) else 0.0,
            shoulder_scale,
        ], dtype=np.float32)
        v = np.concatenate((
            shoulder_rel,
            full_global(left, anchor, shoulder_scale),
            full_global(right, anchor, shoulder_scale),
            full_local(left),
            full_local(right),
            meta10,
        )).astype(np.float32)

    elif mode == "288":
        base = build_feature("268", left, right, shoulders, present, detected, held, scores)
        extra = np.concatenate((
            geometry10(left, present[0], detected[0], held[0], scores[0], shoulder_scale),
            geometry10(right, present[1], detected[1], held[1], scores[1], shoulder_scale),
        )).astype(np.float32)
        v = np.concatenate((base, extra)).astype(np.float32)

    elif mode == "btj_global":
        meta10 = np.array([
            present[0], present[1], detected[0], detected[1], held[0], held[1],
            shoulder_ok, shoulder_scale, scores[0], scores[1],
        ], dtype=np.float32)
        v = np.concatenate((
            shoulder_rel,
            selected_global(left, anchor, shoulder_scale, BTJ_SELECTED_KIND),
            selected_global(right, anchor, shoulder_scale, BTJ_SELECTED_KIND),
            l_ang, r_ang,
            meta10,
        )).astype(np.float32)

    elif mode == "btj_local":
        meta10 = np.array([
            present[0], present[1], detected[0], detected[1], held[0], held[1],
            shoulder_ok, shoulder_scale, scores[0], scores[1],
        ], dtype=np.float32)
        v = np.concatenate((
            shoulder_rel,
            selected_local(left, BTJ_SELECTED_KIND),
            selected_local(right, BTJ_SELECTED_KIND),
            l_ang, r_ang,
            meta10,
        )).astype(np.float32)

    elif mode == "btj_global_local":
        meta10 = np.array([
            present[0], present[1], detected[0], detected[1], held[0], held[1],
            shoulder_ok, shoulder_scale, scores[0], scores[1],
        ], dtype=np.float32)
        v = np.concatenate((
            shoulder_rel,
            selected_global(left, anchor, shoulder_scale, BTJ_SELECTED_KIND),
            selected_global(right, anchor, shoulder_scale, BTJ_SELECTED_KIND),
            selected_local(left, BTJ_SELECTED_KIND),
            selected_local(right, BTJ_SELECTED_KIND),
            l_ang, r_ang,
            meta10,
        )).astype(np.float32)
    else:
        raise ValueError(f"Unknown feature mode: {mode}")

    expected = FEATURE_DIMS[mode]
    if v.shape[0] != expected:
        raise RuntimeError(f"Feature dim mismatch for mode={mode}: got {v.shape[0]}, expected {expected}")
    return v


class UltraMediaPipeExtractor:
    def __init__(
        self,
        feature_mode: str = "btj_global_local",
        shoulder_backend: str = "none",
        proc_width: int = 256,
        pose_proc_width: int = 160,
        pose_every: int = 30,
        hand_every: int = 1,
        hand_model_complexity: int = 0,
        pose_model_complexity: int = 0,
        det_conf: float = 0.50,
        track_conf: float = 0.50,
        smooth_alpha: float = 0.85,
        hold_frames: int = 2,
        mirror_input: bool = False,
        mirror_handedness: bool = True,
    ) -> None:
        if feature_mode not in FEATURE_DIMS:
            raise ValueError(f"Unknown feature mode {feature_mode}")
        self.feature_mode = feature_mode
        self.shoulder_backend = shoulder_backend
        self.proc_width = int(proc_width)
        self.pose_proc_width = int(pose_proc_width)
        self.pose_every = max(1, int(pose_every))
        self.hand_every = max(1, int(hand_every))
        self.smooth_alpha = float(smooth_alpha)
        self.hold_frames = int(hold_frames)
        self.mirror_input = bool(mirror_input)
        self.mirror_handedness = bool(mirror_handedness)
        self.hold_enabled = True
        self.swap_lr = False

        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=int(hand_model_complexity),
            min_detection_confidence=float(det_conf),
            min_tracking_confidence=float(track_conf),
        )
        self.pose = None
        if self.shoulder_backend == "mp-pose":
            self.pose = mp_pose.Pose(
                static_image_mode=False,
                model_complexity=int(pose_model_complexity),
                smooth_landmarks=True,
                enable_segmentation=False,
                min_detection_confidence=float(det_conf),
                min_tracking_confidence=float(track_conf),
            )

        self.left = HandTrack()
        self.right = HandTrack()
        self.shoulders = blank_shoulders()
        self.shoulder_age = 10000
        self.frame_i = 0
        self.last_hand_ms = 0.0
        self.last_pose_ms = 0.0

    def close(self) -> None:
        self.hands.close()
        if self.pose is not None:
            self.pose.close()

    def _process_pose(self, frame_bgr: np.ndarray) -> None:
        if self.pose is None:
            self.shoulders = blank_shoulders()
            self.shoulder_age += 1
            self.last_pose_ms = 0.0
            return
        if self.frame_i == 1 or self.frame_i % self.pose_every == 0 or self.shoulder_age > self.pose_every * 4:
            pose_frame = resize_width(frame_bgr, self.pose_proc_width)
            rgb = cv2.cvtColor(pose_frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            t0 = time.perf_counter()
            res = self.pose.process(rgb)
            self.last_pose_ms = (time.perf_counter() - t0) * 1000.0
            sh = pose_shoulders(res.pose_landmarks)
            if np.isfinite(sh[:, :2]).all():
                self.shoulders = sh
                self.shoulder_age = 0
            else:
                self.shoulder_age += 1
        else:
            self.shoulder_age += 1
            self.last_pose_ms = 0.0

    @staticmethod
    def _valid(track: HandTrack, hold_frames: int) -> bool:
        return track.xyz is not None and track.age <= hold_frames and np.isfinite(track.xyz).all()

    def _candidate_cost(self, cand: CandidateHand, side: str) -> float:
        track = self.left if side == "left" else self.right
        shoulder_idx = 0 if side == "left" else 1
        wrist = cand.xyz[0, :2]
        cost = 0.0
        weight = 0.0
        if self._valid(track, self.hold_frames):
            cost += 5.0 * safe_norm(wrist - track.xyz[0, :2])
            weight += 5.0
        if np.isfinite(self.shoulders[shoulder_idx, :2]).all():
            cost += 1.0 * safe_norm(wrist - self.shoulders[shoulder_idx, :2])
            weight += 1.0
        if cand.label:
            anatomical = "Left" if side == "left" else "Right"
            if cand.label == anatomical:
                cost -= 0.05 * max(0.5, cand.score)
            else:
                cost += 0.18 * max(0.5, cand.score)
        return cost / max(weight, 1.0)

    def _assign(self, candidates: Sequence[CandidateHand]) -> Tuple[Optional[CandidateHand], Optional[CandidateHand]]:
        if not candidates:
            return None, None
        cands = sorted(candidates, key=lambda c: c.score, reverse=True)[:2]
        if len(cands) == 1:
            c = cands[0]
            return (c, None) if self._candidate_cost(c, "left") <= self._candidate_cost(c, "right") else (None, c)
        a, b = cands[0], cands[1]
        ab = self._candidate_cost(a, "left") + self._candidate_cost(b, "right")
        ba = self._candidate_cost(b, "left") + self._candidate_cost(a, "right")
        return (a, b) if ab <= ba else (b, a)

    def _swap_guard(self, l: Optional[CandidateHand], r: Optional[CandidateHand]) -> Tuple[Optional[CandidateHand], Optional[CandidateHand]]:
        if l is None or r is None:
            return l, r
        if not (self._valid(self.left, self.hold_frames) and self._valid(self.right, self.hold_frames)):
            return l, r
        direct = safe_norm(l.xyz[0, :2] - self.left.xyz[0, :2]) + safe_norm(r.xyz[0, :2] - self.right.xyz[0, :2])
        cross = safe_norm(l.xyz[0, :2] - self.right.xyz[0, :2]) + safe_norm(r.xyz[0, :2] - self.left.xyz[0, :2])
        if cross + 0.030 < direct:
            return r, l
        return l, r

    def _update_track(self, track: HandTrack, cand: Optional[CandidateHand]) -> None:
        if cand is not None:
            new = cand.xyz.astype(np.float32)
            if self._valid(track, self.hold_frames):
                # High alpha = responsive, less skeleton lag.
                new = (self.smooth_alpha * new + (1.0 - self.smooth_alpha) * track.xyz).astype(np.float32)
            track.xyz = new
            track.age = 0
            track.detected = True
            track.held = False
            track.score = float(cand.score)
        else:
            track.detected = False
            track.age += 1
            if self.hold_enabled and track.xyz is not None and track.age <= self.hold_frames:
                track.held = True
                track.score *= 0.60
            else:
                track.held = False
                if track.age > self.hold_frames:
                    track.xyz = None
                    track.score = 0.0

    def _process_hands(self, frame_bgr: np.ndarray) -> None:
        if self.frame_i % self.hand_every != 0 and (self.left.xyz is not None or self.right.xyz is not None):
            self.left.detected = False
            self.right.detected = False
            self.left.held = self.left.xyz is not None
            self.right.held = self.right.xyz is not None
            self.left.age += 1
            self.right.age += 1
            self.last_hand_ms = 0.0
            return

        proc = resize_width(frame_bgr, self.proc_width)
        rgb = cv2.cvtColor(proc, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        t0 = time.perf_counter()
        res = self.hands.process(rgb)
        self.last_hand_ms = (time.perf_counter() - t0) * 1000.0

        candidates: List[CandidateHand] = []
        if res.multi_hand_landmarks:
            handed = res.multi_handedness or []
            for i, lms in enumerate(res.multi_hand_landmarks):
                xyz = lm_to_np(lms, 21)
                if xyz is None:
                    continue
                label = ""
                score = 1.0
                if i < len(handed) and handed[i].classification:
                    cls = handed[i].classification[0]
                    label = cls.label or ""
                    score = float(cls.score or 0.0)
                if self.mirror_handedness:
                    if label == "Left":
                        label = "Right"
                    elif label == "Right":
                        label = "Left"
                candidates.append(CandidateHand(xyz=xyz, label=label, score=score))

        l, r = self._assign(candidates)
        l, r = self._swap_guard(l, r)
        if self.swap_lr:
            l, r = r, l
        self._update_track(self.left, l)
        self._update_track(self.right, r)

    def process(self, frame_bgr: np.ndarray) -> FrameResult:
        self.frame_i += 1
        frame = frame_bgr
        if self.mirror_input:
            frame = cv2.flip(frame, 1)
        self._process_pose(frame)
        self._process_hands(frame)

        left = self.left.xyz if self.left.xyz is not None and self.left.age <= self.hold_frames else None
        right = self.right.xyz if self.right.xyz is not None and self.right.age <= self.hold_frames else None
        present = np.array([left is not None, right is not None], dtype=np.float32)
        detected = np.array([self.left.detected, self.right.detected], dtype=np.float32)
        held = np.array([self.left.held, self.right.held], dtype=np.float32)
        scores = np.array([self.left.score, self.right.score], dtype=np.float32)

        t0 = time.perf_counter()
        vec = build_feature(self.feature_mode, left, right, self.shoulders, present, detected, held, scores)
        feature_ms = (time.perf_counter() - t0) * 1000.0
        total_ms = max(self.last_hand_ms + self.last_pose_ms + feature_ms, 1e-6)
        return FrameResult(
            vector=vec,
            left=left,
            right=right,
            shoulders=self.shoulders.copy(),
            detected=detected,
            held=held,
            present=present,
            scores=scores,
            fps_infer=1000.0 / total_ms,
            hand_ms=self.last_hand_ms,
            pose_ms=self.last_pose_ms,
            feature_mode=self.feature_mode,
        )


def draw_simple_hand(img: np.ndarray, hand: Optional[np.ndarray], color: Tuple[int, int, int]) -> None:
    if hand is None:
        return
    h, w = img.shape[:2]
    pts = np.round(hand[:, :2] * np.array([w, h], dtype=np.float32)).astype(int)
    for a, b in HAND_CONNECTIONS:
        pa, pb = tuple(pts[a]), tuple(pts[b])
        cv2.line(img, pa, pb, color, 1, cv2.LINE_AA)
    for p in pts:
        cv2.circle(img, tuple(p), 2, color, -1, cv2.LINE_AA)


def draw_shoulders(img: np.ndarray, shoulders: np.ndarray) -> None:
    if shoulders is None or not np.isfinite(shoulders[:, :2]).all():
        return
    h, w = img.shape[:2]
    pts = np.round(shoulders[:, :2] * np.array([w, h], dtype=np.float32)).astype(int)
    cv2.circle(img, tuple(pts[0]), 5, (255, 180, 80), -1)
    cv2.circle(img, tuple(pts[1]), 5, (80, 180, 255), -1)
    cv2.line(img, tuple(pts[0]), tuple(pts[1]), (180, 180, 180), 1, cv2.LINE_AA)


def save_sequence(out_dir: Path, prefix: str, rows: List[np.ndarray], meta: List[dict], feature_mode: str) -> None:
    if not rows:
        print("[REC] No frames recorded.")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    arr = np.stack(rows).astype(np.float32)
    npz_path = out_dir / f"{prefix}_{feature_mode}_{ts}.npz"
    csv_path = out_dir / f"{prefix}_{feature_mode}_{ts}.csv"
    meta_path = out_dir / f"{prefix}_{feature_mode}_{ts}_meta.json"
    np.savez_compressed(npz_path, features=arr, feature_mode=feature_mode, feature_dim=arr.shape[1])
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([f"f{i}" for i in range(arr.shape[1])])
        writer.writerows(arr.tolist())
    with meta_path.open("w") as f:
        json.dump({"feature_mode": feature_mode, "feature_dim": int(arr.shape[1]), "frames": meta}, f, indent=2)
    print(f"[REC] Saved {arr.shape} -> {npz_path}")


def save_gif(frames: Sequence[np.ndarray], path: Path, fps: int = 10, max_width: int = 360) -> None:
    if imageio is None:
        print("[GIF] imageio not installed.")
        return
    if not frames:
        print("[GIF] Buffer empty.")
        return
    out = []
    for bgr in frames:
        h, w = bgr.shape[:2]
        if w > max_width:
            sc = max_width / float(w)
            bgr = cv2.resize(bgr, (max_width, int(h * sc)), interpolation=cv2.INTER_AREA)
        out.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    imageio.mimsave(path, out, duration=1.0 / max(fps, 1), loop=0)
    print(f"[GIF] Saved {len(out)} frames -> {path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pure MediaPipe ultra-minimal BISINDO feature extractor")
    p.add_argument("--cam", type=int, default=0)
    p.add_argument("--csi", action="store_true")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--fourcc", type=str, default="MJPG")
    p.add_argument("--no-gstreamer", action="store_true")
    p.add_argument("--feature-mode", choices=list(FEATURE_DIMS), default="btj_global_local")
    p.add_argument("--shoulder-backend", choices=["none", "mp-pose"], default="none")
    p.add_argument("--proc-width", type=int, default=256, help="Hand processing width. Lower = faster.")
    p.add_argument("--pose-proc-width", type=int, default=144, help="Pose processing width if mp-pose is enabled.")
    p.add_argument("--pose-every", type=int, default=30, help="Run pose every N frames. Use none backend for fastest.")
    p.add_argument("--hand-every", type=int, default=1, help="Run hands every N frames. 1 = most accurate.")
    p.add_argument("--hand-model-complexity", type=int, choices=[0, 1], default=0)
    p.add_argument("--pose-model-complexity", type=int, choices=[0, 1], default=0)
    p.add_argument("--det-conf", type=float, default=0.50)
    p.add_argument("--track-conf", type=float, default=0.50)
    p.add_argument("--smooth-alpha", type=float, default=0.85, help="Higher = more responsive, lower = smoother.")
    p.add_argument("--hold-frames", type=int, default=2)
    p.add_argument("--center-crop", type=float, default=0.86, help="Crop wide-camera edges. 1.0 disables crop.")
    p.add_argument("--preview-width", type=int, default=426)
    p.add_argument("--mirror-input", action="store_true")
    p.add_argument("--no-mirror-handedness", action="store_true")
    p.add_argument("--no-overlay", action="store_true")
    p.add_argument("--no-display", action="store_true")
    p.add_argument("--enable-gif-buffer", action="store_true")
    p.add_argument("--gif-sec", type=float, default=5.0)
    p.add_argument("--gif-fps", type=int, default=10)
    p.add_argument("--out-dir", type=str, default="runs_mp_ultra_full")
    p.add_argument("--perf-log-every", type=int, default=60)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("BISINDO MediaPipe Ultra Full")
    print(f"feature_mode={args.feature_mode} dim={FEATURE_DIMS[args.feature_mode]}")
    print(f"camera={args.cam} size={args.width}x{args.height}@{args.fps} fourcc={args.fourcc}")
    print(f"proc_width={args.proc_width} center_crop={args.center_crop} shoulder={args.shoulder_backend}")
    print("Keys: Q quit | R record | S snapshot | O overlay | H hold | X swap L/R | G gif")
    print("=" * 72)

    cam = LatestFrameCamera(
        src=args.cam,
        width=args.width,
        height=args.height,
        fps=args.fps,
        use_csi=args.csi,
        use_gstreamer=not args.no_gstreamer,
        fourcc=args.fourcc,
    ).start()

    extractor = UltraMediaPipeExtractor(
        feature_mode=args.feature_mode,
        shoulder_backend=args.shoulder_backend,
        proc_width=args.proc_width,
        pose_proc_width=args.pose_proc_width,
        pose_every=args.pose_every,
        hand_every=args.hand_every,
        hand_model_complexity=args.hand_model_complexity,
        pose_model_complexity=args.pose_model_complexity,
        det_conf=args.det_conf,
        track_conf=args.track_conf,
        smooth_alpha=args.smooth_alpha,
        hold_frames=args.hold_frames,
        mirror_input=args.mirror_input,
        mirror_handedness=not args.no_mirror_handedness,
    )

    recording = False
    rec_rows: List[np.ndarray] = []
    rec_meta: List[dict] = []
    gif_ring: Optional[Deque[np.ndarray]] = None
    if args.enable_gif_buffer:
        gif_ring = deque(maxlen=max(1, int(args.gif_sec * args.gif_fps)))

    fps_ema = 0.0
    loop_i = 0
    last_print = time.perf_counter()
    overlay = not args.no_overlay

    try:
        while True:
            ok, frame = cam.read()
            if not ok or frame is None:
                time.sleep(0.002)
                continue

            loop_t0 = time.perf_counter()
            frame = center_crop(frame, args.center_crop)
            if args.mirror_input:
                frame = cv2.flip(frame, 1)
            res = extractor.process(frame)

            loop_dt = time.perf_counter() - loop_t0
            inst_fps = 1.0 / max(loop_dt, 1e-6)
            fps_ema = inst_fps if fps_ema <= 0 else 0.90 * fps_ema + 0.10 * inst_fps
            loop_i += 1

            if recording:
                rec_rows.append(res.vector.copy())
                rec_meta.append({
                    "t": time.time(),
                    "present": res.present.tolist(),
                    "detected": res.detected.tolist(),
                    "held": res.held.tolist(),
                    "scores": res.scores.tolist(),
                    "fps": float(fps_ema),
                    "hand_ms": float(res.hand_ms),
                    "pose_ms": float(res.pose_ms),
                })

            if args.perf_log_every > 0 and loop_i % args.perf_log_every == 0:
                now = time.perf_counter()
                print(
                    f"[PERF] fps={fps_ema:.1f} hand={res.hand_ms:.1f}ms pose={res.pose_ms:.1f}ms "
                    f"mode={args.feature_mode} dim={res.vector.shape[0]} rec={len(rec_rows)} "
                    f"elapsed={now - last_print:.1f}s"
                )
                last_print = now

            if not args.no_display:
                vis = resize_width(frame, args.preview_width)
                if overlay:
                    draw_shoulders(vis, res.shoulders)
                    draw_simple_hand(vis, res.left, (0, 255, 0))
                    draw_simple_hand(vis, res.right, (0, 180, 255))
                # Minimal HUD only. Avoid heavy text/drawing.
                cv2.putText(
                    vis,
                    f"{fps_ema:.1f} FPS | {args.feature_mode}:{res.vector.shape[0]} | L{int(res.present[0])} R{int(res.present[1])} | rec {len(rec_rows) if recording else '-'}",
                    (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA,
                )
                if recording:
                    cv2.circle(vis, (vis.shape[1] - 18, 18), 7, (0, 0, 255), -1)
                cv2.imshow("BISINDO MP Ultra Full", vis)
                if gif_ring is not None and loop_i % max(1, int(max(fps_ema, 1) / args.gif_fps)) == 0:
                    gif_ring.append(vis.copy())
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), ord('Q'), 27):
                    break
                elif key in (ord('o'), ord('O')):
                    overlay = not overlay
                    print(f"[KEY] overlay={overlay}")
                elif key in (ord('h'), ord('H')):
                    extractor.hold_enabled = not extractor.hold_enabled
                    print(f"[KEY] hold_enabled={extractor.hold_enabled}")
                elif key in (ord('x'), ord('X')):
                    extractor.swap_lr = not extractor.swap_lr
                    print(f"[KEY] swap_lr={extractor.swap_lr}")
                elif key in (ord('r'), ord('R')):
                    if recording:
                        save_sequence(out_dir, "seq", rec_rows, rec_meta, args.feature_mode)
                        rec_rows.clear()
                        rec_meta.clear()
                        recording = False
                    else:
                        rec_rows.clear()
                        rec_meta.clear()
                        recording = True
                        print("[REC] started")
                elif key in (ord('s'), ord('S')):
                    save_sequence(out_dir, "snapshot", [res.vector.copy()], [{"t": time.time()}], args.feature_mode)
                elif key in (ord('g'), ord('G')):
                    if gif_ring is None:
                        print("[GIF] disabled. Run with --enable-gif-buffer")
                    else:
                        save_gif(list(gif_ring), out_dir / f"live_{args.feature_mode}_{time.strftime('%Y%m%d_%H%M%S')}.gif", fps=args.gif_fps)
            else:
                # In benchmark mode, still allow Ctrl+C. No imshow/drawing at all.
                pass

    except KeyboardInterrupt:
        print("\n[INFO] interrupted")
    finally:
        if recording and rec_rows:
            save_sequence(out_dir, "seq", rec_rows, rec_meta, args.feature_mode)
        extractor.close()
        cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
