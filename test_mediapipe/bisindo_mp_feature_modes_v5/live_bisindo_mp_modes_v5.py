#!/usr/bin/env python3
"""
BISINDO live feature extractor - pure MediaPipe, multi feature modes.

Target: stable/accurate ~7-10 FPS on Jetson/Orin with ordinary wide RGB camera.
No YOLO, no TensorRT. Focus: shoulders + palms + fingers.

Feature modes:
  84              : compact shoulder + palm/wrist + finger angles.
  179             : shoulder + full global 3D hands + finger angles + meta.
  228             : 179 + selected local palm/finger landmarks + richer geometry/meta.
  268             : shoulder + full hand global + full hand local + meta.
  288             : 268 + per-hand geometry/quality.
  btj_global      : bahu + telapak + jari selected keypoints, global only.
  btj_local       : bahu + telapak + jari selected keypoints, local only.
  btj_global_local: bahu + telapak + jari selected keypoints, global + local.

Keys:
  Q/ESC : quit
  R     : start/stop record sequence to NPZ+CSV
  S     : save one snapshot NPZ
  O     : overlay on/off
  G     : save GIF if --enable-gif-buffer is set
  H     : hold-last-good on/off
  M     : mirror preview on/off (feature extraction remains unmirrored)
  X     : swap left/right handedness labels on/off
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import queue
import signal
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    import mediapipe as mp
except Exception as e:
    print("[FATAL] Failed to import mediapipe:", repr(e))
    print("Install with: pip install mediapipe")
    raise

try:
    import imageio.v2 as imageio
except Exception:
    imageio = None


FEATURE_DIMS = {
    "84": 84, "palm84": 84, "palm_angles": 84,
    "179": 179, "hand179": 179, "compat179": 179,
    "228": 228, "hand228": 228, "rich228": 228,
    "268": 268, "full268": 268, "global_local268": 268,
    "288": 288, "full288": 288, "global_local288": 288,
    "btj_global": 114, "global_btj": 114,
    "btj_local": 114, "local_btj": 114,
    "btj_global_local": 180, "btj_gl": 180, "global_local_btj": 180,
}
EXTRA_PER_HAND_DIM = 10

# MediaPipe hand indices
WRIST = 0
THUMB_CMC = 1
THUMB_MCP = 2
THUMB_IP = 3
THUMB_TIP = 4
INDEX_MCP = 5
INDEX_PIP = 6
INDEX_DIP = 7
INDEX_TIP = 8
MIDDLE_MCP = 9
MIDDLE_PIP = 10
MIDDLE_DIP = 11
MIDDLE_TIP = 12
RING_MCP = 13
RING_PIP = 14
RING_DIP = 15
RING_TIP = 16
PINKY_MCP = 17
PINKY_PIP = 18
PINKY_DIP = 19
PINKY_TIP = 20


@dataclass
class HandState:
    xyz: Optional[np.ndarray] = None          # (21,3) normalized original frame + z
    raw_xyz: Optional[np.ndarray] = None      # before smoothing
    handedness_score: float = 0.0
    last_seen_frame: int = -999999
    detected_now: bool = False
    present: bool = False
    held: bool = False
    geom: np.ndarray = field(default_factory=lambda: np.zeros((EXTRA_PER_HAND_DIM,), dtype=np.float32))


@dataclass
class ShoulderState:
    left: Optional[np.ndarray] = None         # (3,) normalized original frame
    right: Optional[np.ndarray] = None
    ok: bool = False
    visibility_mean: float = 0.0
    scale: float = 0.35
    last_seen_frame: int = -999999


class LatestFrameReader:
    """Camera reader that keeps only the newest frame to avoid latency buildup."""

    def __init__(self, cam: int | str, width: int, height: int, fps: int, fourcc: str = "MJPG", backend: str = "v4l2"):
        self.cam = cam
        self.width = width
        self.height = height
        self.fps = fps
        self.fourcc = fourcc
        self.backend = backend
        self.cap = None
        self.lock = threading.Lock()
        self.frame = None
        self.running = False
        self.thread = None
        self.frames_read = 0

    def open(self):
        if isinstance(self.cam, str) and not self.cam.isdigit():
            source = self.cam
        else:
            source = int(self.cam)
        if self.backend.lower() == "v4l2" and isinstance(source, int):
            self.cap = cv2.VideoCapture(source, cv2.CAP_V4L2)
        else:
            self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera/source: {self.cam}")

        if self.fourcc:
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.fourcc))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        # Tiny buffer if backend honors it.
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return self

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        return self

    def _loop(self):
        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.003)
                continue
            with self.lock:
                self.frame = frame
                self.frames_read += 1

    def read(self):
        if self.thread is None:
            ok, frame = self.cap.read()
            if not ok:
                return None
            return frame
        with self.lock:
            if self.frame is None:
                return None
            return self.frame.copy()

    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        if self.cap is not None:
            self.cap.release()


def now_stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def compute_crop_rect(w: int, h: int, center_crop: float) -> Tuple[int, int, int, int]:
    center_crop = float(np.clip(center_crop, 0.2, 1.0))
    cw = int(round(w * center_crop))
    ch = int(round(h * center_crop))
    x0 = max(0, (w - cw) // 2)
    y0 = max(0, (h - ch) // 2)
    return x0, y0, cw, ch


def resize_keep_aspect(img: np.ndarray, target_width: int) -> np.ndarray:
    h, w = img.shape[:2]
    if target_width <= 0 or w == target_width:
        return img
    scale = target_width / float(w)
    target_h = max(1, int(round(h * scale)))
    return cv2.resize(img, (target_width, target_h), interpolation=cv2.INTER_AREA)


def landmark_to_original_xyz(lm, crop_rect, frame_w: int, frame_h: int, z_scale: float = 1.0) -> np.ndarray:
    x0, y0, cw, ch = crop_rect
    x = (x0 + float(lm.x) * cw) / float(frame_w)
    y = (y0 + float(lm.y) * ch) / float(frame_h)
    z = float(lm.z) * z_scale
    return np.array([x, y, z], dtype=np.float32)


def hand_landmarks_to_xyz(hand_lms, crop_rect, frame_w: int, frame_h: int, z_scale: float = 1.0) -> np.ndarray:
    arr = np.zeros((21, 3), dtype=np.float32)
    for i, lm in enumerate(hand_lms.landmark):
        arr[i] = landmark_to_original_xyz(lm, crop_rect, frame_w, frame_h, z_scale=z_scale)
    return arr


def bbox_from_xyz(xyz: np.ndarray) -> Tuple[float, float, float, float, float, float, float]:
    xs = xyz[:, 0]
    ys = xyz[:, 1]
    x1, x2 = float(xs.min()), float(xs.max())
    y1, y2 = float(ys.min()), float(ys.max())
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    area = bw * bh
    return x1, y1, x2, y2, bw, bh, area


def dist2d(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a[:2] - b[:2]))


def palm_size_2d(xyz: np.ndarray) -> float:
    # Robust hand scale from wrist to MCPs and palm width.
    d1 = dist2d(xyz[WRIST], xyz[MIDDLE_MCP])
    d2 = dist2d(xyz[INDEX_MCP], xyz[PINKY_MCP])
    d3 = dist2d(xyz[WRIST], xyz[INDEX_MCP])
    d4 = dist2d(xyz[WRIST], xyz[PINKY_MCP])
    vals = [v for v in [d1, d2, d3, d4] if np.isfinite(v) and v > 1e-6]
    return float(np.median(vals)) if vals else 1e-3


def pseudo_z_from_scale(scale_vs_shoulder: float, ref: float = 0.38, gain: float = 0.25) -> float:
    # MediaPipe z convention roughly: smaller/more negative means closer to camera.
    s = max(float(scale_vs_shoulder), 1e-4)
    return float(-math.log(s / max(ref, 1e-4)) * gain)


def apply_z_mode(xyz: np.ndarray, shoulder_scale: float, z_mode: str, z_ref_scale: float, z_pseudo_gain: float, z_mp_gain: float) -> Tuple[np.ndarray, float, float, float]:
    out = xyz.copy()
    palm = palm_size_2d(out)
    scale_vs_shoulder = palm / max(float(shoulder_scale), 1e-5)
    pz = pseudo_z_from_scale(scale_vs_shoulder, ref=z_ref_scale, gain=z_pseudo_gain)
    mpz = out[:, 2].copy() * z_mp_gain
    if z_mode == "mp":
        out[:, 2] = mpz
    elif z_mode == "pseudo":
        out[:, 2] = pz
    else:  # blend
        out[:, 2] = mpz + pz
    return out, palm, scale_vs_shoulder, pz


def safe_shoulder_default() -> ShoulderState:
    st = ShoulderState()
    st.left = np.array([0.36, 0.43, 0.0], dtype=np.float32)
    st.right = np.array([0.64, 0.43, 0.0], dtype=np.float32)
    st.ok = False
    st.visibility_mean = 0.0
    st.scale = dist2d(st.left, st.right)
    return st


def shoulder_center_scale(st: ShoulderState) -> Tuple[np.ndarray, float]:
    if st.left is None or st.right is None:
        tmp = safe_shoulder_default()
        return (tmp.left + tmp.right) * 0.5, tmp.scale
    center = (st.left + st.right) * 0.5
    scale = max(dist2d(st.left, st.right), 1e-3)
    return center.astype(np.float32), scale


def normalize_shoulder_features(st: ShoulderState) -> np.ndarray:
    center, scale = shoulder_center_scale(st)
    left = st.left if st.left is not None else np.array([0.36, 0.43, 0.0], dtype=np.float32)
    right = st.right if st.right is not None else np.array([0.64, 0.43, 0.0], dtype=np.float32)
    return np.concatenate([(left - center) / scale, (right - center) / scale]).astype(np.float32)


def hand_global_features(xyz: Optional[np.ndarray], center: np.ndarray, scale: float) -> np.ndarray:
    if xyz is None:
        return np.zeros((63,), dtype=np.float32)
    return ((xyz - center.reshape(1, 3)) / max(scale, 1e-5)).reshape(-1).astype(np.float32)


def hand_local_features(xyz: Optional[np.ndarray], palm_scale: float) -> np.ndarray:
    if xyz is None:
        return np.zeros((63,), dtype=np.float32)
    wrist = xyz[WRIST].reshape(1, 3)
    denom = max(float(palm_scale), 1e-5)
    return ((xyz - wrist) / denom).reshape(-1).astype(np.float32)


def smooth_xyz(prev: Optional[np.ndarray], curr: np.ndarray, alpha: float) -> np.ndarray:
    if prev is None:
        return curr.astype(np.float32)
    return (alpha * curr + (1.0 - alpha) * prev).astype(np.float32)


def parse_handedness(h, swap: bool = False) -> Tuple[str, float]:
    # MediaPipe returns label and score. Label might depend on selfie mirroring, so user can swap.
    label = h.classification[0].label.lower() if h.classification else "unknown"
    score = float(h.classification[0].score) if h.classification else 0.0
    if label.startswith("left"):
        side = "left"
    elif label.startswith("right"):
        side = "right"
    else:
        side = "unknown"
    if swap:
        if side == "left":
            side = "right"
        elif side == "right":
            side = "left"
    return side, score


def assign_hands(dets: List[Dict], hand_states: Dict[str, HandState]) -> Dict[str, Optional[Dict]]:
    """Assign detections to left/right robustly using handedness, confidence, and last wrist proximity."""
    assigned = {"left": None, "right": None}
    if not dets:
        return assigned

    # First pass: unique strong handedness.
    for side in ["left", "right"]:
        candidates = [d for d in dets if d["side"] == side]
        if len(candidates) == 1:
            assigned[side] = candidates[0]
        elif len(candidates) > 1:
            # If duplicate label, choose nearest to previous side if possible, else highest score.
            prev = hand_states[side].xyz
            if prev is not None:
                assigned[side] = min(candidates, key=lambda d: dist2d(d["xyz"][WRIST], prev[WRIST]))
            else:
                assigned[side] = max(candidates, key=lambda d: d["score"])

    used_ids = set(id(v) for v in assigned.values() if v is not None)
    remaining = [d for d in dets if id(d) not in used_ids]

    # Fill missing side by nearest previous wrist.
    for side in ["left", "right"]:
        if assigned[side] is None and remaining:
            prev = hand_states[side].xyz
            if prev is not None:
                chosen = min(remaining, key=lambda d: dist2d(d["xyz"][WRIST], prev[WRIST]))
            else:
                # Fallback by x-position in image; this is image-left/image-right, not anatomical perfect.
                # For non-mirrored view: user's right often appears on image-left. But this is only fallback.
                if side == "left":
                    chosen = max(remaining, key=lambda d: float(d["xyz"][WRIST, 0]))
                else:
                    chosen = min(remaining, key=lambda d: float(d["xyz"][WRIST, 0]))
            assigned[side] = chosen
            remaining.remove(chosen)

    return assigned


def draw_hand_overlay(frame: np.ndarray, xyz: np.ndarray, color: Tuple[int, int, int], label: str = ""):
    h, w = frame.shape[:2]
    pts = [(int(np.clip(p[0] * w, 0, w - 1)), int(np.clip(p[1] * h, 0, h - 1))) for p in xyz]
    # Use a small connection list. Full MediaPipe connections converted to tuples.
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (5, 9), (9, 10), (10, 11), (11, 12),
        (9, 13), (13, 14), (14, 15), (15, 16),
        (13, 17), (17, 18), (18, 19), (19, 20),
        (0, 17)
    ]
    for a, b in connections:
        cv2.line(frame, pts[a], pts[b], color, 1, cv2.LINE_AA)
    for i, p in enumerate(pts):
        r = 3 if i in [0, 4, 8, 12, 16, 20] else 2
        cv2.circle(frame, p, r, color, -1, cv2.LINE_AA)
    if label:
        cv2.putText(frame, label, (pts[0][0] + 5, pts[0][1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def draw_shoulders(frame: np.ndarray, st: ShoulderState):
    if st.left is None or st.right is None:
        return
    h, w = frame.shape[:2]
    l = (int(st.left[0] * w), int(st.left[1] * h))
    r = (int(st.right[0] * w), int(st.right[1] * h))
    cv2.circle(frame, l, 5, (255, 255, 0), -1, cv2.LINE_AA)
    cv2.circle(frame, r, 5, (255, 255, 0), -1, cv2.LINE_AA)
    cv2.line(frame, l, r, (255, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "L shoulder", (l[0] + 5, l[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(frame, "R shoulder", (r[0] + 5, r[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1, cv2.LINE_AA)



FINGER_ANGLE_TRIPLETS = [
    (5, 0, 1), (0, 1, 2), (1, 2, 3), (2, 3, 4),          # thumb
    (0, 5, 6), (5, 6, 7), (6, 7, 8),                     # index
    (0, 9, 10), (9, 10, 11), (10, 11, 12),               # middle
    (0, 13, 14), (13, 14, 15), (14, 15, 16),             # ring
    (0, 17, 18), (17, 18, 19), (18, 19, 20),             # pinky
]

PALM_SELECTED_IDX = [
    WRIST, THUMB_TIP, INDEX_MCP, INDEX_TIP, MIDDLE_MCP, MIDDLE_TIP, PINKY_MCP
]

# "Bahu + telapak tangan + jari" selected points.
# This avoids all 21 raw joints while keeping palm center, MCPs, and fingertips.
BTJ_POINT_NAMES = [
    "wrist", "palm",
    "thumb_tip",
    "index_mcp", "index_tip",
    "middle_mcp", "middle_tip",
    "ring_mcp", "ring_tip",
    "pinky_mcp", "pinky_tip",
]


def canonical_mode(mode: str) -> str:
    m = str(mode).lower().strip().replace("-", "_")
    aliases = {
        "palm84": "84", "palm_angles": "84", "palmangles": "84",
        "hand179": "179", "compat": "179", "compat179": "179",
        "hand228": "228", "rich228": "228",
        "full268": "268", "global_local268": "268", "gl268": "268",
        "full288": "288", "global_local288": "288", "gl288": "288",
        "global_btj": "btj_global",
        "local_btj": "btj_local",
        "btj_gl": "btj_global_local",
        "global_local_btj": "btj_global_local",
        "btj_global_local": "btj_global_local",
        "btj_global": "btj_global",
        "btj_local": "btj_local",
    }
    if m in ("84", "179", "228", "268", "288"):
        return m
    if m in aliases:
        return aliases[m]
    raise ValueError(f"Unknown feature mode: {mode}")


def feature_dim(mode: str) -> int:
    return {
        "84": 84,
        "179": 179,
        "228": 228,
        "268": 268,
        "288": 288,
        "btj_global": 114,
        "btj_local": 114,
        "btj_global_local": 180,
    }[canonical_mode(mode)]


def safe_norm(v: np.ndarray, eps: float = 1e-6) -> float:
    return float(max(np.linalg.norm(v), eps))


def angle_at_b(pa: np.ndarray, pb: np.ndarray, pc: np.ndarray) -> float:
    ba = pa - pb
    bc = pc - pb
    denom = safe_norm(ba) * safe_norm(bc)
    c = float(np.dot(ba, bc) / denom)
    return float(np.arccos(np.clip(c, -1.0, 1.0)))


def finger_angles(xyz: Optional[np.ndarray]) -> np.ndarray:
    if xyz is None:
        return np.zeros((16,), dtype=np.float32)
    out = np.zeros((16,), dtype=np.float32)
    for i, (a, b, c) in enumerate(FINGER_ANGLE_TRIPLETS):
        out[i] = angle_at_b(xyz[a], xyz[b], xyz[c])
    return out


def palm_center(xyz: Optional[np.ndarray]) -> np.ndarray:
    if xyz is None:
        return np.zeros((3,), dtype=np.float32)
    return np.mean(xyz[[WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]], axis=0).astype(np.float32)


def shoulder_core_features(shoulder: ShoulderState, fps_smooth: float) -> Tuple[np.ndarray, np.ndarray, float]:
    center, scale = shoulder_center_scale(shoulder)
    sh_rel = normalize_shoulder_features(shoulder)  # 6
    core10 = np.concatenate([
        sh_rel,
        np.array([
            float(scale),
            1.0 if shoulder.ok else 0.0,
            float(shoulder.visibility_mean),
            min(float(fps_smooth), 60.0) / 60.0,
        ], dtype=np.float32),
    ]).astype(np.float32)
    return core10, center, scale


def palm84_base(hand_states: Dict[str, HandState], shoulder: ShoulderState, fps_smooth: float) -> np.ndarray:
    """52-dim palm/body features: shoulder 12 + per-hand 16*2 + inter-hand 8."""
    center, scale = shoulder_center_scale(shoulder)
    inv = 1.0 / max(scale, 1e-6)
    left_sh = shoulder.left if shoulder.left is not None else np.array([0.36, 0.43, 0.0], dtype=np.float32)
    right_sh = shoulder.right if shoulder.right is not None else np.array([0.64, 0.43, 0.0], dtype=np.float32)
    sh_vec = (right_sh - left_sh).astype(np.float32)
    angle = float(np.arctan2(sh_vec[1], sh_vec[0])) if safe_norm(sh_vec[:2]) > 1e-6 else 0.0

    out: List[float] = []
    out.extend(left_sh.tolist())
    out.extend(right_sh.tolist())
    out.extend(center.tolist())
    out.extend([float(scale), float(np.sin(angle)), float(np.cos(angle))])

    palms = {}
    wrists = {}
    for side, own_sh in (("left", left_sh), ("right", right_sh)):
        st = hand_states[side]
        if st.present and st.xyz is not None:
            lm = st.xyz
            palm = palm_center(lm)
            wrist = lm[WRIST]
            middle = lm[MIDDLE_MCP]
            palm_rel_mid = (palm - center) * inv
            wrist_rel_mid = (wrist - center) * inv
            palm_rel_sh = (palm - own_sh) * inv
            wrist_to_palm = (palm - wrist) * inv
            palm_to_middle = (middle - palm) * inv
            palms[side] = palm
            wrists[side] = wrist
            out.extend(palm_rel_mid.tolist())      # 3
            out.extend(wrist_rel_mid.tolist())     # 3
            out.extend(palm_rel_sh.tolist())       # 3
            out.extend(wrist_to_palm.tolist())     # 3
            out.extend([
                safe_norm(palm_to_middle[:2]),
                float(st.handedness_score),
                1.0 if st.detected_now else 0.0,
                1.0 if st.held else 0.0,
            ])                                     # 4, total per hand 16
        else:
            palms[side] = np.zeros((3,), dtype=np.float32)
            wrists[side] = np.zeros((3,), dtype=np.float32)
            out.extend([0.0] * 16)

    lh, rh = hand_states["left"], hand_states["right"]
    if lh.present and rh.present and lh.xyz is not None and rh.xyz is not None:
        pd = (palms["left"] - palms["right"]) * inv
        wd = (wrists["left"] - wrists["right"]) * inv
        out.extend(pd.tolist())
        out.extend([
            safe_norm(pd[:2]),
            safe_norm(wd[:2]),
            float(pd[2]),
            min(float(lh.handedness_score), float(rh.handedness_score)),
            1.0 if shoulder.ok else 0.0,
        ])
    else:
        out.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, min(float(lh.handedness_score), float(rh.handedness_score)), 1.0 if shoulder.ok else 0.0])

    arr = np.asarray(out, dtype=np.float32)
    if arr.shape[0] != 52:
        raise RuntimeError(f"palm84 base bug: {arr.shape[0]} != 52")
    return arr


def hand179_meta(hand_states: Dict[str, HandState], shoulder: ShoulderState, shoulder_scale: float, fps_smooth: float) -> np.ndarray:
    vals: List[float] = []
    for side in ("left", "right"):
        st = hand_states[side]
        vals.extend([
            1.0 if st.present else 0.0,
            1.0 if st.detected_now else 0.0,
            1.0 if st.held else 0.0,
            float(st.handedness_score),
            float(st.geom[8]) if st.geom is not None and len(st.geom) >= 9 else 0.0,  # scale_vs_shoulder
            float(st.geom[6]) if st.geom is not None and len(st.geom) >= 7 else 0.0,  # bbox area
        ])
    vals.extend([
        1.0 if shoulder.ok else 0.0,
        float(shoulder.visibility_mean),
        min(float(fps_smooth), 60.0) / 60.0,
    ])
    arr = np.asarray(vals, dtype=np.float32)
    if arr.shape[0] != 15:
        raise RuntimeError(f"179 meta bug: {arr.shape[0]} != 15")
    return arr


def selected_local_features(xyz: Optional[np.ndarray], palm_scale: float) -> np.ndarray:
    """7 important palm/finger points * xyz = 21 dims per hand."""
    if xyz is None:
        return np.zeros((21,), dtype=np.float32)
    wrist = xyz[WRIST].reshape(1, 3)
    denom = max(float(palm_scale), 1e-5)
    sel = xyz[PALM_SELECTED_IDX]
    return ((sel - wrist) / denom).reshape(-1).astype(np.float32)


def btj_points(xyz: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Return 11 selected points: wrist, palm center, MCPs, and fingertips."""
    if xyz is None:
        return None
    palm = palm_center(xyz).reshape(1, 3)
    pts = np.vstack([
        xyz[[WRIST]],
        palm,
        xyz[[THUMB_TIP]],
        xyz[[INDEX_MCP]], xyz[[INDEX_TIP]],
        xyz[[MIDDLE_MCP]], xyz[[MIDDLE_TIP]],
        xyz[[RING_MCP]], xyz[[RING_TIP]],
        xyz[[PINKY_MCP]], xyz[[PINKY_TIP]],
    ]).astype(np.float32)
    return pts


def btj_global_features(xyz: Optional[np.ndarray], center: np.ndarray, shoulder_scale: float) -> np.ndarray:
    """11 selected points * xyz = 33 dims per hand, relative to shoulder midpoint."""
    pts = btj_points(xyz)
    if pts is None:
        return np.zeros((33,), dtype=np.float32)
    return ((pts - center.reshape(1, 3)) / max(float(shoulder_scale), 1e-5)).reshape(-1).astype(np.float32)


def btj_local_features(xyz: Optional[np.ndarray], palm_scale: float) -> np.ndarray:
    """11 selected points * xyz = 33 dims per hand, relative to wrist / palm scale."""
    pts = btj_points(xyz)
    if pts is None:
        return np.zeros((33,), dtype=np.float32)
    wrist = xyz[WRIST].reshape(1, 3)
    return ((pts - wrist) / max(float(palm_scale), 1e-5)).reshape(-1).astype(np.float32)


def meta10(hand_states: Dict[str, HandState], shoulder: ShoulderState, shoulder_scale: float, fps_smooth: float) -> np.ndarray:
    l, r = hand_states["left"], hand_states["right"]
    return np.array([
        1.0 if l.present else 0.0,
        1.0 if r.present else 0.0,
        1.0 if l.detected_now else 0.0,
        1.0 if r.detected_now else 0.0,
        1.0 if l.held else 0.0,
        1.0 if r.held else 0.0,
        1.0 if shoulder.ok else 0.0,
        float(shoulder.visibility_mean),
        float(shoulder_scale),
        min(float(fps_smooth), 60.0) / 60.0,
    ], dtype=np.float32)


def rich228_meta(hand_states: Dict[str, HandState], shoulder: ShoulderState, shoulder_scale: float, fps_smooth: float) -> np.ndarray:
    vals: List[float] = []
    for side in ("left", "right"):
        st = hand_states[side]
        geom = st.geom if st.geom is not None else np.zeros((10,), dtype=np.float32)
        vals.extend([
            1.0 if st.present else 0.0,
            1.0 if st.detected_now else 0.0,
            1.0 if st.held else 0.0,
            float(st.handedness_score),
            float(geom[4]),  # bbox_w
            float(geom[5]),  # bbox_h
            float(geom[6]),  # bbox_area
            float(geom[8]),  # scale_vs_shoulder
        ])

    l, r = hand_states["left"], hand_states["right"]
    if l.present and r.present and l.xyz is not None and r.xyz is not None:
        lp, rp = palm_center(l.xyz), palm_center(r.xyz)
        vals.extend([
            dist2d(l.xyz[WRIST], r.xyz[WRIST]) / max(shoulder_scale, 1e-5),
            dist2d(lp, rp) / max(shoulder_scale, 1e-5),
            dist2d(l.xyz[INDEX_TIP], r.xyz[INDEX_TIP]) / max(shoulder_scale, 1e-5),
            dist2d(l.xyz[THUMB_TIP], r.xyz[THUMB_TIP]) / max(shoulder_scale, 1e-5),
            float(lp[2] - rp[2]),
            1.0 if shoulder.ok else 0.0,
        ])
    else:
        vals.extend([0.0, 0.0, 0.0, 0.0, 0.0, 1.0 if shoulder.ok else 0.0])
    arr = np.asarray(vals, dtype=np.float32)
    if arr.shape[0] != 22:
        raise RuntimeError(f"228 meta bug: {arr.shape[0]} != 22")
    return arr


def build_feature_vector(hand_states: Dict[str, HandState], shoulder: ShoulderState, fps_smooth: float, mode: str) -> np.ndarray:
    mode = canonical_mode(mode)
    core10, center, shoulder_scale = shoulder_core_features(shoulder, fps_smooth)
    sh_feat = core10[:6]

    left = hand_states["left"]
    right = hand_states["right"]

    left_palm = max(float(left.geom[7]) if left.geom is not None and len(left.geom) >= 8 else 1e-3, 1e-5)
    right_palm = max(float(right.geom[7]) if right.geom is not None and len(right.geom) >= 8 else 1e-3, 1e-5)

    left_xyz = left.xyz if left.present else None
    right_xyz = right.xyz if right.present else None

    # Full 21-point hand features.
    left_global = hand_global_features(left_xyz, center, shoulder_scale)
    right_global = hand_global_features(right_xyz, center, shoulder_scale)
    left_local = hand_local_features(left_xyz, left_palm)
    right_local = hand_local_features(right_xyz, right_palm)

    # Compact bahu + telapak + jari selected-point features.
    left_btj_global = btj_global_features(left_xyz, center, shoulder_scale)
    right_btj_global = btj_global_features(right_xyz, center, shoulder_scale)
    left_btj_local = btj_local_features(left_xyz, left_palm)
    right_btj_local = btj_local_features(right_xyz, right_palm)

    left_angles = finger_angles(left_xyz)
    right_angles = finger_angles(right_xyz)
    m10 = meta10(hand_states, shoulder, shoulder_scale, fps_smooth)

    if mode == "84":
        feat = np.concatenate([
            palm84_base(hand_states, shoulder, fps_smooth),  # 52
            left_angles,                                     # 16
            right_angles,                                    # 16
        ]).astype(np.float32)
    elif mode == "179":
        feat = np.concatenate([
            sh_feat,             # 6
            left_global,         # 63
            right_global,        # 63
            left_angles,         # 16
            right_angles,        # 16
            hand179_meta(hand_states, shoulder, shoulder_scale, fps_smooth),  # 15
        ]).astype(np.float32)
    elif mode == "228":
        feat = np.concatenate([
            sh_feat,             # 6
            left_global,         # 63
            right_global,        # 63
            selected_local_features(left_xyz, left_palm),    # 21
            selected_local_features(right_xyz, right_palm),  # 21
            left_angles,         # 16
            right_angles,        # 16
            rich228_meta(hand_states, shoulder, shoulder_scale, fps_smooth),  # 22
        ]).astype(np.float32)
    elif mode == "268":
        feat = np.concatenate([
            sh_feat,       # 6
            left_global,   # 63
            right_global,  # 63
            left_local,    # 63
            right_local,   # 63
            m10,           # 10
        ]).astype(np.float32)
    elif mode == "288":
        feat = np.concatenate([
            sh_feat,       # 6
            left_global,   # 63
            right_global,  # 63
            left_local,    # 63
            right_local,   # 63
            m10,           # 10
            left.geom.astype(np.float32),   # 10
            right.geom.astype(np.float32),  # 10
        ]).astype(np.float32)
    elif mode == "btj_global":
        feat = np.concatenate([
            sh_feat,            # 6
            left_btj_global,    # 33
            right_btj_global,   # 33
            left_angles,        # 16
            right_angles,       # 16
            m10,                # 10
        ]).astype(np.float32)
    elif mode == "btj_local":
        feat = np.concatenate([
            sh_feat,           # 6
            left_btj_local,    # 33
            right_btj_local,   # 33
            left_angles,       # 16
            right_angles,      # 16
            m10,               # 10
        ]).astype(np.float32)
    elif mode == "btj_global_local":
        feat = np.concatenate([
            sh_feat,            # 6
            left_btj_global,    # 33
            right_btj_global,   # 33
            left_btj_local,     # 33
            right_btj_local,    # 33
            left_angles,        # 16
            right_angles,       # 16
            m10,                # 10
        ]).astype(np.float32)
    else:
        raise ValueError(mode)

    expected = feature_dim(mode)
    if feat.shape[0] != expected:
        raise RuntimeError(f"Feature dim mismatch for {mode}: got {feat.shape[0]}, expected {expected}")
    return np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

def feature_header(mode: str) -> List[str]:
    mode = canonical_mode(mode)
    axes = ["x", "y", "z"]
    names: List[str] = []

    def add_shoulder():
        names.extend([f"shoulder_{side}_{a}" for side in ["left", "right"] for a in axes])

    def add_full(prefix: str):
        names.extend([f"{prefix}_lm{i}_{a}" for i in range(21) for a in axes])

    def add_btj(prefix: str):
        for point in BTJ_POINT_NAMES:
            names.extend([f"{prefix}_{point}_{a}" for a in axes])

    def add_angles():
        names.extend([f"left_angle_{i:02d}" for i in range(16)])
        names.extend([f"right_angle_{i:02d}" for i in range(16)])

    def add_meta10():
        names.extend(["left_present", "right_present", "left_detected", "right_detected", "left_held", "right_held", "shoulder_ok", "shoulder_visibility", "shoulder_scale", "fps_norm"])

    if mode == "84":
        names += [f"body_left_shoulder_{a}" for a in axes]
        names += [f"body_right_shoulder_{a}" for a in axes]
        names += [f"body_mid_shoulder_{a}" for a in axes]
        names += ["body_shoulder_scale", "body_shoulder_sin", "body_shoulder_cos"]
        for side in ["left", "right"]:
            names += [f"{side}_palm_rel_mid_{a}" for a in axes]
            names += [f"{side}_wrist_rel_mid_{a}" for a in axes]
            names += [f"{side}_palm_rel_shoulder_{a}" for a in axes]
            names += [f"{side}_wrist_to_palm_{a}" for a in axes]
            names += [f"{side}_palm_to_middle_len", f"{side}_score", f"{side}_detected", f"{side}_held"]
        names += [f"inter_palm_delta_{a}" for a in axes]
        names += ["inter_palm_dist", "inter_wrist_dist", "inter_palm_z_delta", "inter_min_score", "shoulder_ok"]
        add_angles()
    elif mode == "179":
        add_shoulder(); add_full("left_global"); add_full("right_global"); add_angles()
        for side in ["left", "right"]:
            names += [f"{side}_present", f"{side}_detected", f"{side}_held", f"{side}_score", f"{side}_scale_vs_shoulder", f"{side}_bbox_area"]
        names += ["shoulder_ok", "shoulder_visibility", "fps_norm"]
    elif mode == "228":
        add_shoulder(); add_full("left_global"); add_full("right_global")
        for side in ["left", "right"]:
            for idx in PALM_SELECTED_IDX:
                names += [f"{side}_local_sel_lm{idx}_{a}" for a in axes]
        add_angles()
        for side in ["left", "right"]:
            names += [f"{side}_present", f"{side}_detected", f"{side}_held", f"{side}_score", f"{side}_bbox_w", f"{side}_bbox_h", f"{side}_bbox_area", f"{side}_scale_vs_shoulder"]
        names += ["inter_wrist_dist", "inter_palm_dist", "inter_index_tip_dist", "inter_thumb_tip_dist", "inter_palm_z_delta", "shoulder_ok"]
    elif mode == "268":
        add_shoulder(); add_full("left_global"); add_full("right_global"); add_full("left_local"); add_full("right_local"); add_meta10()
    elif mode == "288":
        add_shoulder(); add_full("left_global"); add_full("right_global"); add_full("left_local"); add_full("right_local"); add_meta10()
        for side in ["left", "right"]:
            names += [f"{side}_geom_present", f"{side}_geom_detected", f"{side}_geom_held", f"{side}_geom_score", f"{side}_bbox_w", f"{side}_bbox_h", f"{side}_bbox_area", f"{side}_palm_size", f"{side}_scale_vs_shoulder", f"{side}_pseudo_z"]
    elif mode == "btj_global":
        add_shoulder(); add_btj("left_global"); add_btj("right_global"); add_angles(); add_meta10()
    elif mode == "btj_local":
        add_shoulder(); add_btj("left_local"); add_btj("right_local"); add_angles(); add_meta10()
    elif mode == "btj_global_local":
        add_shoulder(); add_btj("left_global"); add_btj("right_global"); add_btj("left_local"); add_btj("right_local"); add_angles(); add_meta10()

    if len(names) != feature_dim(mode):
        raise RuntimeError(f"Header dim mismatch for {mode}: {len(names)} != {feature_dim(mode)}")
    return names

def feature_layout_text(mode: str) -> str:
    mode = canonical_mode(mode)
    if mode == "84":
        return "52 shoulder/palm/wrist/inter-hand features + 32 finger-angle features = 84"
    if mode == "179":
        return "6 shoulders + 126 full hand global xyz + 32 finger angles + 15 status/quality = 179"
    if mode == "228":
        return "179-style global full hand + 42 selected local palm/finger xyz + 22 geometry/status = 228"
    if mode == "268":
        return "6 shoulders + 126 full hand global xyz + 126 full hand local xyz + 10 meta = 268"
    if mode == "288":
        return "268 + 20 per-hand geometry/quality/pseudo-depth = 288"
    if mode == "btj_global":
        return "6 shoulders + 66 selected bahu-telapak-jari global xyz + 32 angles + 10 meta = 114"
    if mode == "btj_local":
        return "6 shoulders + 66 selected bahu-telapak-jari local xyz + 32 angles + 10 meta = 114"
    if mode == "btj_global_local":
        return "6 shoulders + 66 selected global + 66 selected local + 32 angles + 10 meta = 180"
    return mode

def save_sequence(out_dir: Path, features: List[np.ndarray], timestamps: List[float], meta: Dict, mode: str):
    if not features:
        print("[REC] No frames to save.")
        return
    stamp = now_stamp()
    arr = np.stack(features, axis=0).astype(np.float32)
    ts = np.array(timestamps, dtype=np.float64)
    mode = canonical_mode(mode)
    npz_path = out_dir / f"seq_mp_{mode}_{stamp}.npz"
    csv_path = out_dir / f"seq_mp_{mode}_{stamp}.csv"
    meta_path = out_dir / f"seq_mp_{mode}_{stamp}_meta.json"
    np.savez_compressed(npz_path, features=arr, timestamps=ts, feature_dim=feature_dim(mode), feature_mode=mode, header=np.array(feature_header(mode)))
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp"] + feature_header(mode))
        for t, row in zip(ts, arr):
            writer.writerow([f"{t:.6f}"] + [f"{float(v):.7g}" for v in row])
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[REC] Saved: {npz_path}")
    print(f"[REC] Saved: {csv_path}")


def save_snapshot(out_dir: Path, feature: np.ndarray, meta: Dict, mode: str):
    stamp = now_stamp()
    mode = canonical_mode(mode)
    path = out_dir / f"snapshot_mp_{mode}_{stamp}.npz"
    np.savez_compressed(path, feature=feature.astype(np.float32), feature_dim=feature_dim(mode), feature_mode=mode, header=np.array(feature_header(mode)), meta=json.dumps(meta))
    print(f"[SNAP] Saved: {path}")


def save_gif(out_dir: Path, gif_frames: List[np.ndarray], fps: int = 10):
    if imageio is None:
        print("[GIF] imageio not installed. pip install imageio")
        return
    if not gif_frames:
        print("[GIF] Buffer empty. Run with --enable-gif-buffer.")
        return
    stamp = now_stamp()
    path = out_dir / f"live_mp_3modes_{stamp}.gif"
    # Convert BGR frames to RGB for imageio.
    rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in gif_frames]
    imageio.mimsave(path, rgb, fps=fps)
    print(f"[GIF] Saved: {path}")


def update_geometry(st: HandState, shoulder_scale: float, present: bool, detected: bool, held: bool, palm: float, scale_vs_shoulder: float, pseudo_z: float):
    if st.xyz is not None and present:
        _, _, _, _, bw, bh, area = bbox_from_xyz(st.xyz)
    else:
        bw = bh = area = 0.0
        palm = 0.0
        scale_vs_shoulder = 0.0
        pseudo_z = 0.0
    st.geom = np.array([
        1.0 if present else 0.0,
        1.0 if detected else 0.0,
        1.0 if held else 0.0,
        float(st.handedness_score),
        float(bw),
        float(bh),
        float(area),
        float(palm),
        float(scale_vs_shoulder),
        float(pseudo_z),
    ], dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description="Pure MediaPipe BISINDO extractor with 84/179/228/268/288 and BTJ local/global modes.")
    parser.add_argument("--cam", default="0", help="Camera index or video path")
    parser.add_argument("--feature-mode", default="179", choices=["84", "palm84", "179", "hand179", "228", "hand228", "268", "288", "btj_global", "btj_local", "btj_global_local"], help="Feature vector type. Recommended: 179 first; use btj_global/local/global_local for bahu+telapak+jari selected features.")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--fourcc", default="MJPG", help="Camera FourCC, e.g. MJPG, YUYV, or empty")
    parser.add_argument("--threaded-cam", action="store_true", help="Use latest-frame-only camera thread")
    parser.add_argument("--backend", default="v4l2", choices=["v4l2", "default"])

    parser.add_argument("--proc-width", type=int, default=384, help="Hand processing crop width. Accurate: 384/416. Faster: 288/320.")
    parser.add_argument("--center-crop", type=float, default=0.92, help="Crop center fraction for wide camera. 1.0 disables crop.")
    parser.add_argument("--hand-model-complexity", type=int, default=1, choices=[0, 1])
    parser.add_argument("--min-det-conf", type=float, default=0.60)
    parser.add_argument("--min-track-conf", type=float, default=0.60)
    parser.add_argument("--hand-every", type=int, default=1, help="Process hands every N frames. For accuracy use 1.")
    parser.add_argument("--hold-frames", type=int, default=8, help="Hold last good hand when temporarily missing.")
    parser.add_argument("--smooth-alpha", type=float, default=0.62, help="EMA alpha. Higher = less smoothing, lower = smoother.")

    parser.add_argument("--shoulder-backend", default="mp-pose", choices=["mp-pose", "none"], help="Shoulder source")
    parser.add_argument("--shoulder-every", type=int, default=8, help="Run pose every N frames")
    parser.add_argument("--shoulder-proc-width", type=int, default=224)
    parser.add_argument("--pose-model-complexity", type=int, default=1, choices=[0, 1, 2])
    parser.add_argument("--pose-min-det-conf", type=float, default=0.55)
    parser.add_argument("--pose-min-track-conf", type=float, default=0.55)

    parser.add_argument("--z-mode", default="blend", choices=["mp", "pseudo", "blend"])
    parser.add_argument("--z-ref-scale", type=float, default=0.38)
    parser.add_argument("--z-pseudo-gain", type=float, default=0.25)
    parser.add_argument("--z-mp-gain", type=float, default=1.0)

    parser.add_argument("--swap-handedness", action="store_true", help="Swap MediaPipe left/right labels")
    parser.add_argument("--preview-width", type=int, default=640)
    parser.add_argument("--no-overlay", action="store_true")
    parser.add_argument("--draw-every", type=int, default=1, help="Draw skeleton every N frames, preview still updates every frame")
    parser.add_argument("--mirror-preview", action="store_true", help="Mirror preview only")
    parser.add_argument("--enable-gif-buffer", action="store_true")
    parser.add_argument("--gif-seconds", type=float, default=4.0)
    parser.add_argument("--gif-fps", type=int, default=10)
    parser.add_argument("--out-dir", default="runs_mp_3feature_modes")
    parser.add_argument("--perf-log-every", type=int, default=30)
    parser.add_argument("--window-name", default="BISINDO MP Feature Modes")

    args = parser.parse_args()
    out_dir = ensure_dir(args.out_dir)

    mp_hands = mp.solutions.hands
    mp_pose = mp.solutions.pose

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=args.hand_model_complexity,
        min_detection_confidence=args.min_det_conf,
        min_tracking_confidence=args.min_track_conf,
    )

    pose = None
    if args.shoulder_backend == "mp-pose":
        pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=args.pose_model_complexity,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=args.pose_min_det_conf,
            min_tracking_confidence=args.pose_min_track_conf,
        )

    reader = LatestFrameReader(args.cam, args.width, args.height, args.fps, fourcc=args.fourcc, backend=args.backend).open()
    if args.threaded_cam:
        reader.start()

    print("=" * 72)
    print("BISINDO Pure MediaPipe Multi Feature Modes")
    print(f"camera={args.cam} size={args.width}x{args.height} fps={args.fps} fourcc={args.fourcc}")
    print(f"proc_width={args.proc_width} hand_complexity={args.hand_model_complexity} det={args.min_det_conf} track={args.min_track_conf}")
    print(f"shoulder={args.shoulder_backend} shoulder_every={args.shoulder_every} shoulder_proc_width={args.shoulder_proc_width}")
    feature_mode = canonical_mode(args.feature_mode)
    print(f"feature_mode={feature_mode} feature_dim={feature_dim(feature_mode)} z_mode={args.z_mode} center_crop={args.center_crop}")
    print("Modes: 84, 179, 228, 268, 288, btj_global(114), btj_local(114), btj_global_local(180)")
    print("Keys: Q/ESC quit | R record | S snapshot | O overlay | G GIF | H hold | M mirror | X swap labels")
    print("=" * 72)

    hand_states = {"left": HandState(), "right": HandState()}
    shoulder = safe_shoulder_default()
    hold_enabled = True
    overlay_enabled = not args.no_overlay
    mirror_preview = bool(args.mirror_preview)
    swap_handedness = bool(args.swap_handedness)

    frame_idx = 0
    last_time = time.perf_counter()
    fps_smooth = 0.0
    last_feature = np.zeros((feature_dim(feature_mode),), dtype=np.float32)

    recording = False
    rec_features: List[np.ndarray] = []
    rec_ts: List[float] = []
    rec_meta = {}

    gif_frames: List[np.ndarray] = []
    gif_max_frames = max(1, int(args.gif_seconds * args.gif_fps))

    should_stop = False

    def handle_sigint(sig, frame):
        nonlocal should_stop
        should_stop = True

    signal.signal(signal.SIGINT, handle_sigint)

    try:
        while not should_stop:
            frame = reader.read()
            if frame is None:
                time.sleep(0.005)
                continue
            frame_idx += 1
            h, w = frame.shape[:2]
            crop_rect = compute_crop_rect(w, h, args.center_crop)
            x0, y0, cw, ch = crop_rect
            crop = frame[y0:y0 + ch, x0:x0 + cw]

            # Shoulder update, intentionally less frequent than hands.
            if args.shoulder_backend == "none":
                if shoulder.left is None or shoulder.right is None:
                    shoulder = safe_shoulder_default()
            elif pose is not None and (frame_idx % max(1, args.shoulder_every) == 1 or not shoulder.ok):
                pose_img = resize_keep_aspect(crop, args.shoulder_proc_width)
                pose_rgb = cv2.cvtColor(pose_img, cv2.COLOR_BGR2RGB)
                pose_rgb.flags.writeable = False
                pose_res = pose.process(pose_rgb)
                if pose_res.pose_landmarks:
                    lms = pose_res.pose_landmarks.landmark
                    # MediaPipe Pose: 11 left shoulder, 12 right shoulder
                    l_sh = lms[11]
                    r_sh = lms[12]
                    lv = float(getattr(l_sh, "visibility", 0.0))
                    rv = float(getattr(r_sh, "visibility", 0.0))
                    left = landmark_to_original_xyz(l_sh, crop_rect, w, h, z_scale=1.0)
                    right = landmark_to_original_xyz(r_sh, crop_rect, w, h, z_scale=1.0)
                    vis = (lv + rv) * 0.5
                    valid = vis >= 0.35 and dist2d(left, right) > 0.06
                    if valid:
                        if shoulder.left is None or shoulder.right is None:
                            shoulder.left, shoulder.right = left, right
                        else:
                            # Shoulder smoothing stronger to avoid jitter.
                            shoulder.left = (0.35 * left + 0.65 * shoulder.left).astype(np.float32)
                            shoulder.right = (0.35 * right + 0.65 * shoulder.right).astype(np.float32)
                        shoulder.ok = True
                        shoulder.visibility_mean = vis
                        shoulder.scale = dist2d(shoulder.left, shoulder.right)
                        shoulder.last_seen_frame = frame_idx
                # If pose failed, keep last shoulder. If very old, default is still usable.
                if frame_idx - shoulder.last_seen_frame > 240 and shoulder.ok:
                    shoulder.ok = False

            _, shoulder_scale = shoulder_center_scale(shoulder)

            # Reset per-frame hand flags.
            for side in ["left", "right"]:
                hand_states[side].detected_now = False
                hand_states[side].held = False
                hand_states[side].present = False

            # Hand processing.
            should_process_hand = (frame_idx % max(1, args.hand_every) == 1) or (args.hand_every == 1)
            hand_result_dets: List[Dict] = []
            if should_process_hand:
                hand_img = resize_keep_aspect(crop, args.proc_width)
                rgb = cv2.cvtColor(hand_img, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                res = hands.process(rgb)
                if res.multi_hand_landmarks:
                    handed = res.multi_handedness or []
                    for i, h_lms in enumerate(res.multi_hand_landmarks[:2]):
                        hinfo = handed[i] if i < len(handed) else None
                        if hinfo is not None:
                            side, score = parse_handedness(hinfo, swap=swap_handedness)
                        else:
                            side, score = "unknown", 0.0
                        xyz_raw = hand_landmarks_to_xyz(h_lms, crop_rect, w, h, z_scale=1.0)
                        xyz_z, palm, scale_vs_sh, pz = apply_z_mode(
                            xyz_raw, shoulder_scale, args.z_mode, args.z_ref_scale, args.z_pseudo_gain, args.z_mp_gain
                        )
                        hand_result_dets.append({
                            "side": side,
                            "score": score,
                            "xyz": xyz_z,
                            "raw_xyz": xyz_raw,
                            "palm": palm,
                            "scale_vs_shoulder": scale_vs_sh,
                            "pseudo_z": pz,
                        })

                assigned = assign_hands(hand_result_dets, hand_states)
                for side in ["left", "right"]:
                    det = assigned[side]
                    st = hand_states[side]
                    if det is not None:
                        st.detected_now = True
                        st.present = True
                        st.held = False
                        st.handedness_score = float(det["score"])
                        st.raw_xyz = det["raw_xyz"]
                        st.xyz = smooth_xyz(st.xyz, det["xyz"], alpha=args.smooth_alpha)
                        st.last_seen_frame = frame_idx
                        # recompute geometry from smoothed xyz for bbox stability.
                        palm = palm_size_2d(st.xyz)
                        scale_vs_sh = palm / max(shoulder_scale, 1e-5)
                        pz = pseudo_z_from_scale(scale_vs_sh, args.z_ref_scale, args.z_pseudo_gain)
                        update_geometry(st, shoulder_scale, True, True, False, palm, scale_vs_sh, pz)

            # Hold missing hands for short occlusion/self-handshake.
            for side in ["left", "right"]:
                st = hand_states[side]
                if not st.detected_now:
                    age = frame_idx - st.last_seen_frame
                    if hold_enabled and st.xyz is not None and age <= args.hold_frames:
                        st.present = True
                        st.held = True
                        palm = palm_size_2d(st.xyz)
                        scale_vs_sh = palm / max(shoulder_scale, 1e-5)
                        pz = pseudo_z_from_scale(scale_vs_sh, args.z_ref_scale, args.z_pseudo_gain)
                        update_geometry(st, shoulder_scale, True, False, True, palm, scale_vs_sh, pz)
                    else:
                        st.present = False
                        st.held = False
                        update_geometry(st, shoulder_scale, False, False, False, 0.0, 0.0, 0.0)

            # FPS calculation.
            t = time.perf_counter()
            dt = max(t - last_time, 1e-6)
            inst_fps = 1.0 / dt
            last_time = t
            fps_smooth = inst_fps if fps_smooth <= 0 else (0.08 * inst_fps + 0.92 * fps_smooth)

            feature = build_feature_vector(hand_states, shoulder, fps_smooth, feature_mode)
            last_feature = feature

            if recording:
                rec_features.append(feature.copy())
                rec_ts.append(time.time())

            # Preview. Always update with latest camera frame; overlay can be sparse.
            preview = frame.copy()
            if overlay_enabled and (frame_idx % max(1, args.draw_every) == 0):
                draw_shoulders(preview, shoulder)
                if hand_states["left"].present and hand_states["left"].xyz is not None:
                    draw_hand_overlay(preview, hand_states["left"].xyz, (0, 255, 0), "LEFT" + (" HOLD" if hand_states["left"].held else ""))
                if hand_states["right"].present and hand_states["right"].xyz is not None:
                    draw_hand_overlay(preview, hand_states["right"].xyz, (0, 128, 255), "RIGHT" + (" HOLD" if hand_states["right"].held else ""))

            # Draw status text every frame; cheap and useful.
            status = f"FPS {fps_smooth:4.1f} | L:{int(hand_states['left'].present)} R:{int(hand_states['right'].present)} | dim {feature_dim(feature_mode)} | rec:{'ON' if recording else 'off'}"
            cv2.putText(preview, status, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(preview, status, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 1, cv2.LINE_AA)
            status2 = f"hold:{'on' if hold_enabled else 'off'} shoulder:{args.shoulder_backend}/{int(shoulder.ok)} z:{args.z_mode} swap:{int(swap_handedness)}"
            cv2.putText(preview, status2, (8, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(preview, status2, (8, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

            if mirror_preview:
                preview = cv2.flip(preview, 1)
            if args.preview_width > 0 and preview.shape[1] != args.preview_width:
                preview = resize_keep_aspect(preview, args.preview_width)

            if args.enable_gif_buffer:
                # Downsample GIF buffer to reduce memory/CPU.
                gif_frame = resize_keep_aspect(preview, 360)
                gif_frames.append(gif_frame.copy())
                if len(gif_frames) > gif_max_frames:
                    del gif_frames[0:len(gif_frames) - gif_max_frames]

            cv2.imshow(args.window_name, preview)
            key = cv2.waitKey(1) & 0xFF
            if key in [ord('q'), ord('Q'), 27]:
                break
            elif key in [ord('o'), ord('O')]:
                overlay_enabled = not overlay_enabled
                print(f"[UI] overlay={overlay_enabled}")
            elif key in [ord('h'), ord('H')]:
                hold_enabled = not hold_enabled
                print(f"[UI] hold={hold_enabled}")
            elif key in [ord('m'), ord('M')]:
                mirror_preview = not mirror_preview
                print(f"[UI] mirror_preview={mirror_preview}")
            elif key in [ord('x'), ord('X')]:
                swap_handedness = not swap_handedness
                print(f"[UI] swap_handedness={swap_handedness}")
            elif key in [ord('s'), ord('S')]:
                save_snapshot(out_dir, last_feature, vars(args), feature_mode)
            elif key in [ord('g'), ord('G')]:
                save_gif(out_dir, gif_frames, fps=args.gif_fps)
            elif key in [ord('r'), ord('R')]:
                if not recording:
                    recording = True
                    rec_features = []
                    rec_ts = []
                    rec_meta = vars(args).copy()
                    rec_meta.update({
                        "feature_dim": feature_dim(feature_mode),
                        "feature_mode": feature_mode,
                        "feature_layout": feature_layout_text(feature_mode),
                        "started_at": now_stamp(),
                    })
                    print("[REC] started")
                else:
                    recording = False
                    rec_meta["stopped_at"] = now_stamp()
                    save_sequence(out_dir, rec_features, rec_ts, rec_meta, feature_mode)

            if args.perf_log_every > 0 and frame_idx % args.perf_log_every == 0:
                print(
                    f"[PERF] frame={frame_idx} fps={fps_smooth:.2f} "
                    f"L(p/d/h)={int(hand_states['left'].present)}/{int(hand_states['left'].detected_now)}/{int(hand_states['left'].held)} "
                    f"R(p/d/h)={int(hand_states['right'].present)}/{int(hand_states['right'].detected_now)}/{int(hand_states['right'].held)} "
                    f"shoulder_ok={int(shoulder.ok)} scale={shoulder.scale:.3f}"
                )

    finally:
        if recording and rec_features:
            rec_meta["stopped_at"] = now_stamp()
            save_sequence(out_dir, rec_features, rec_ts, rec_meta, feature_mode)
        reader.stop()
        hands.close()
        if pose is not None:
            pose.close()
        cv2.destroyAllWindows()
        print("[DONE]")


if __name__ == "__main__":
    main()
