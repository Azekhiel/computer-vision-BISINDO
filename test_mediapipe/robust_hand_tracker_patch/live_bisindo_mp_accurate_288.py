#!/usr/bin/env python3
"""
BISINDO live feature extractor - pure MediaPipe accurate mode, 288 dims/frame.

Target: stable/accurate ~10 FPS on Jetson/Orin with ordinary wide RGB camera.
No YOLO, no TensorRT. Focus: MediaPipe Hands accuracy + shoulder anchor + low latency.

Feature layout, 288 dim:
  0:6       shoulders L/R normalized to shoulder center/scale, 2*xyz
  6:69      left hand global, 21*xyz, relative to shoulder center/scale
  69:132    right hand global, 21*xyz, relative to shoulder center/scale
  132:195   left hand local, 21*xyz, relative to wrist/palm scale
  195:258   right hand local, 21*xyz, relative to wrist/palm scale
  258:268   metadata 10 dim
  268:278   left geometry/quality 10 dim
  278:288   right geometry/quality 10 dim

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


FEATURE_DIM = 288
BASE_FEATURE_DIM = 268
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


def build_feature_vector(hand_states: Dict[str, HandState], shoulder: ShoulderState, fps_smooth: float) -> np.ndarray:
    center, shoulder_scale = shoulder_center_scale(shoulder)
    sh_feat = normalize_shoulder_features(shoulder)

    left = hand_states["left"]
    right = hand_states["right"]

    left_palm = max(float(left.geom[7]) if left.geom is not None and len(left.geom) >= 8 else 1e-3, 1e-5)
    right_palm = max(float(right.geom[7]) if right.geom is not None and len(right.geom) >= 8 else 1e-3, 1e-5)

    left_global = hand_global_features(left.xyz if left.present else None, center, shoulder_scale)
    right_global = hand_global_features(right.xyz if right.present else None, center, shoulder_scale)
    left_local = hand_local_features(left.xyz if left.present else None, left_palm)
    right_local = hand_local_features(right.xyz if right.present else None, right_palm)

    meta = np.array([
        1.0 if left.present else 0.0,
        1.0 if right.present else 0.0,
        1.0 if left.detected_now else 0.0,
        1.0 if right.detected_now else 0.0,
        1.0 if left.held else 0.0,
        1.0 if right.held else 0.0,
        1.0 if shoulder.ok else 0.0,
        float(shoulder.visibility_mean),
        float(shoulder_scale),
        float(fps_smooth),
    ], dtype=np.float32)

    feat = np.concatenate([
        sh_feat,
        left_global,
        right_global,
        left_local,
        right_local,
        meta,
        left.geom.astype(np.float32),
        right.geom.astype(np.float32),
    ]).astype(np.float32)
    if feat.shape[0] != FEATURE_DIM:
        raise RuntimeError(f"Feature dim mismatch: got {feat.shape[0]}, expected {FEATURE_DIM}")
    # Clean NaN/Inf.
    feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
    return feat


def feature_header() -> List[str]:
    names = []
    names += [f"shoulder_{side}_{axis}" for side in ["left", "right"] for axis in ["x", "y", "z"]]
    names += [f"left_global_lm{i}_{axis}" for i in range(21) for axis in ["x", "y", "z"]]
    names += [f"right_global_lm{i}_{axis}" for i in range(21) for axis in ["x", "y", "z"]]
    names += [f"left_local_lm{i}_{axis}" for i in range(21) for axis in ["x", "y", "z"]]
    names += [f"right_local_lm{i}_{axis}" for i in range(21) for axis in ["x", "y", "z"]]
    names += [
        "left_present", "right_present", "left_detected", "right_detected",
        "left_held", "right_held", "shoulder_ok", "shoulder_visibility",
        "shoulder_scale", "fps_smooth"
    ]
    extra = ["present", "detected", "held", "handedness_score", "bbox_w", "bbox_h", "bbox_area", "palm_size", "scale_vs_shoulder", "pseudo_z"]
    names += [f"left_geom_{x}" for x in extra]
    names += [f"right_geom_{x}" for x in extra]
    assert len(names) == FEATURE_DIM
    return names


def save_sequence(out_dir: Path, features: List[np.ndarray], timestamps: List[float], meta: Dict):
    if not features:
        print("[REC] No frames to save.")
        return
    stamp = now_stamp()
    arr = np.stack(features, axis=0).astype(np.float32)
    ts = np.array(timestamps, dtype=np.float64)
    npz_path = out_dir / f"seq_mp_accurate_288_{stamp}.npz"
    csv_path = out_dir / f"seq_mp_accurate_288_{stamp}.csv"
    meta_path = out_dir / f"seq_mp_accurate_288_{stamp}_meta.json"
    np.savez_compressed(npz_path, features=arr, timestamps=ts, feature_dim=FEATURE_DIM, header=np.array(feature_header()))
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp"] + feature_header())
        for t, row in zip(ts, arr):
            writer.writerow([f"{t:.6f}"] + [f"{float(v):.7g}" for v in row])
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[REC] Saved: {npz_path}")
    print(f"[REC] Saved: {csv_path}")


def save_snapshot(out_dir: Path, feature: np.ndarray, meta: Dict):
    stamp = now_stamp()
    path = out_dir / f"snapshot_mp_accurate_288_{stamp}.npz"
    np.savez_compressed(path, feature=feature.astype(np.float32), feature_dim=FEATURE_DIM, header=np.array(feature_header()), meta=json.dumps(meta))
    print(f"[SNAP] Saved: {path}")


def save_gif(out_dir: Path, gif_frames: List[np.ndarray], fps: int = 10):
    if imageio is None:
        print("[GIF] imageio not installed. pip install imageio")
        return
    if not gif_frames:
        print("[GIF] Buffer empty. Run with --enable-gif-buffer.")
        return
    stamp = now_stamp()
    path = out_dir / f"live_mp_accurate_288_{stamp}.gif"
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
    parser = argparse.ArgumentParser(description="Pure MediaPipe accurate BISINDO extractor, 288 dims.")
    parser.add_argument("--cam", default="0", help="Camera index or video path")
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
    parser.add_argument("--out-dir", default="runs_mp_accurate_288")
    parser.add_argument("--perf-log-every", type=int, default=30)
    parser.add_argument("--window-name", default="BISINDO MP Accurate 288")

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
    print("BISINDO Pure MediaPipe Accurate 288")
    print(f"camera={args.cam} size={args.width}x{args.height} fps={args.fps} fourcc={args.fourcc}")
    print(f"proc_width={args.proc_width} hand_complexity={args.hand_model_complexity} det={args.min_det_conf} track={args.min_track_conf}")
    print(f"shoulder={args.shoulder_backend} shoulder_every={args.shoulder_every} shoulder_proc_width={args.shoulder_proc_width}")
    print(f"feature_dim={FEATURE_DIM} z_mode={args.z_mode} center_crop={args.center_crop}")
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
    last_feature = np.zeros((FEATURE_DIM,), dtype=np.float32)

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

            feature = build_feature_vector(hand_states, shoulder, fps_smooth)
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
            status = f"FPS {fps_smooth:4.1f} | L:{int(hand_states['left'].present)} R:{int(hand_states['right'].present)} | dim {FEATURE_DIM} | rec:{'ON' if recording else 'off'}"
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
                save_snapshot(out_dir, last_feature, vars(args))
            elif key in [ord('g'), ord('G')]:
                save_gif(out_dir, gif_frames, fps=args.gif_fps)
            elif key in [ord('r'), ord('R')]:
                if not recording:
                    recording = True
                    rec_features = []
                    rec_ts = []
                    rec_meta = vars(args).copy()
                    rec_meta.update({
                        "feature_dim": FEATURE_DIM,
                        "feature_layout": "6 shoulder + 126 global hands + 126 local hands + 10 metadata + 20 geom_quality",
                        "started_at": now_stamp(),
                    })
                    print("[REC] started")
                else:
                    recording = False
                    rec_meta["stopped_at"] = now_stamp()
                    save_sequence(out_dir, rec_features, rec_ts, rec_meta)

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
            save_sequence(out_dir, rec_features, rec_ts, rec_meta)
        reader.stop()
        hands.close()
        if pose is not None:
            pose.close()
        cv2.destroyAllWindows()
        print("[DONE]")


if __name__ == "__main__":
    main()
