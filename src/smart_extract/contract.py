"""Shared Smart Extract V8 best-mode feature contract."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from typing import Iterable

import numpy as np

FEATURE_SCHEMA = "bisindo_smart_v8_btj_global_local_180_best"
FEATURE_MODE = "btj_global_local"
FEATURE_DIM = 180
TARGET_FPS = 10.0
EXTRACT_PROFILE = "smart_v8_best"

SLICE_SHOULDERS = slice(0, 6)
SLICE_LEFT_GLOBAL = slice(6, 39)
SLICE_RIGHT_GLOBAL = slice(39, 72)
SLICE_LEFT_LOCAL = slice(72, 105)
SLICE_RIGHT_LOCAL = slice(105, 138)
SLICE_LEFT_ANGLES = slice(138, 154)
SLICE_RIGHT_ANGLES = slice(154, 170)
SLICE_META = slice(170, 180)

IDX_META_LEFT_PRESENT = 0
IDX_META_RIGHT_PRESENT = 1
IDX_META_LEFT_DETECTED = 2
IDX_META_RIGHT_DETECTED = 3
IDX_META_LEFT_HELD = 4
IDX_META_RIGHT_HELD = 5
IDX_META_SHOULDER_OK = 6
IDX_META_SHOULDER_SCALE = 7
IDX_META_LEFT_SCORE = 8
IDX_META_RIGHT_SCORE = 9

BEST_FALLBACK_VARIANTS = "auto,clahe_sharp,gamma_bright,sharp,denoise_clahe_sharp,none"

BEST_EXTRACT_SETTINGS = {
    "feature_mode": FEATURE_MODE,
    "target_fps": TARGET_FPS,
    "width": 640,
    "height": 480,
    "center_crop": 1.0,
    "proc_width": 384,
    "shoulder_backend": "mp-pose",
    "pose_every": 3,
    "pose_proc_width": 256,
    "hand_model_complexity": 0,
    "pose_model_complexity": 0,
    "det_conf": 0.40,
    "track_conf": 0.45,
    "smooth_alpha": 0.78,
    "shoulder_smooth_alpha": 0.35,
    "hold_frames": 5,
    "smart_mode": "best",
    "search_radius": 2,
    "enhance": "auto",
    "fallback_variants": BEST_FALLBACK_VARIANTS,
    "gif_width": 420,
}


def make_best_args(**overrides) -> Namespace:
    """Return an argparse-compatible namespace for the best extract profile."""
    values = {
        "video": None,
        "batch_dir": None,
        "out_dir": None,
        "save_gif": True,
        "no_gif": False,
        "save_mp4": False,
        "skeleton_bg": "black",
        "quiet": False,
        "mirror_input": False,
        "no_mirror_handedness": False,
        "weak_hand_threshold": 1.5,
    }
    values.update(BEST_EXTRACT_SETTINGS)
    values.update(overrides)
    return Namespace(**values)


def parse_feature_value(value) -> np.ndarray:
    if isinstance(value, str):
        return np.fromstring(value, sep=",", dtype=np.float32)
    return np.asarray(value, dtype=np.float32).reshape(-1)


def format_feature_value(features: Iterable[float]) -> str:
    return ",".join(map(str, np.asarray(features, dtype=np.float32).reshape(-1).tolist()))


def ensure_feature_dim(sequence, expected_dim: int = FEATURE_DIM) -> np.ndarray:
    arr = np.asarray(sequence, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2 or arr.shape[1] != int(expected_dim):
        raise ValueError(f"Expected feature shape (T, {expected_dim}), got {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError("Feature sequence contains non-finite values")
    return arr.astype(np.float32, copy=False)


def filter_current_feature_rows(df):
    if "feature_version" not in df.columns:
        return df.iloc[0:0].copy()
    if "feature_dim" in df.columns:
        dims = df["feature_dim"]
        try:
            dims = dims.astype(int)
        except Exception:
            def _dim(value):
                try:
                    return int(float(value))
                except Exception:
                    return -1
            dims = dims.apply(_dim)
        return df[(df["feature_version"] == FEATURE_SCHEMA) & (dims == FEATURE_DIM)].copy()
    return df[df["feature_version"] == FEATURE_SCHEMA].copy()


def sample_gif_paths(vocab: str, video_id: str, root_dir: str | Path, modes: Iterable[str] = ("overlay", "skeleton")) -> dict[str, str]:
    safe_video_id = "".join(c if c.isalnum() or c in "._-" else "_" for c in str(video_id))
    base = Path(root_dir) / "assets" / "gifs" / "samples" / str(vocab)
    return {mode: str(base / f"{safe_video_id}_{mode}.gif") for mode in modes}


def motion_score(prev: np.ndarray | None, curr: np.ndarray | None) -> tuple[float, bool]:
    if curr is None:
        return 0.0, False
    curr = np.asarray(curr, dtype=np.float32).reshape(-1)
    if curr.shape[0] < FEATURE_DIM:
        return 0.0, False
    meta = curr[SLICE_META]
    visible = bool(meta[IDX_META_LEFT_PRESENT] >= 0.5 or meta[IDX_META_RIGHT_PRESENT] >= 0.5)
    if prev is None:
        return 0.0, visible
    prev = np.asarray(prev, dtype=np.float32).reshape(-1)
    if prev.shape[0] < FEATURE_DIM:
        return 0.0, visible
    chunks = [
        (SLICE_LEFT_GLOBAL, IDX_META_LEFT_PRESENT),
        (SLICE_RIGHT_GLOBAL, IDX_META_RIGHT_PRESENT),
        (SLICE_LEFT_LOCAL, IDX_META_LEFT_PRESENT),
        (SLICE_RIGHT_LOCAL, IDX_META_RIGHT_PRESENT),
    ]
    scores = []
    for sl, meta_idx in chunks:
        if meta[meta_idx] >= 0.5:
            scores.append(float(np.linalg.norm(curr[sl] - prev[sl])))
    return (float(max(scores)) if scores else 0.0), visible
