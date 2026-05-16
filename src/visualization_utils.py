import os

import matplotlib
matplotlib.use("Agg")

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import feature_engine as fe

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATABASE_DIR = os.path.join(ROOT_DIR, "dataset_parquets")
GIF_OUTPUT_DIR = os.path.join(ROOT_DIR, "assets", "gifs")

POSE_CONNECTIONS = [(0, 1), (0, 2), (2, 4), (1, 3), (3, 5)]


def _parse_features(feature_value):
    if isinstance(feature_value, str):
        return np.fromstring(feature_value, sep=",", dtype=np.float32)
    return np.asarray(feature_value, dtype=np.float32)


def _select_preview_sample(df_vocab):
    original = df_vocab[~df_vocab["video_id"].astype(str).str.contains("_aug_")]
    if "feature_version" in original.columns:
        current_original = original[original["feature_version"] == fe.FEATURE_SCHEMA]
        if not current_original.empty:
            original = current_original
        else:
            return original.iloc[0:0]
    if original.empty:
        original = df_vocab
    target_video_id = original["video_id"].iloc[0]
    return original[original["video_id"] == target_video_id].sort_values("frame_num")


def _block_detected_points(block: np.ndarray, detected: np.ndarray | None) -> np.ndarray:
    flat = block[:, :, :2].reshape(block.shape[0], -1, 2)
    non_zero = np.max(np.linalg.norm(flat, axis=2), axis=1) > 1e-6
    if detected is not None:
        non_zero &= detected.astype(bool)
    if not np.any(non_zero):
        non_zero = np.max(np.linalg.norm(flat, axis=2), axis=1) > 1e-6
    return flat[non_zero].reshape(-1, 2) if np.any(non_zero) else np.empty((0, 2), dtype=np.float32)


def sequence_axis_limits(sequence: np.ndarray, flags: np.ndarray | None = None):
    seq = np.asarray(sequence, dtype=np.float32)
    pose = seq[:, 0:18].reshape(-1, 6, 3)
    left = seq[:, 18:81].reshape(-1, 21, 3)
    right = seq[:, 81:144].reshape(-1, 21, 3)

    pose_ok = left_ok = right_ok = None
    if flags is not None and len(flags) == len(seq):
        pose_ok = flags[:, fe.IDX_POSE] >= 0.5
        left_ok = flags[:, fe.IDX_LH] >= 0.5
        right_ok = flags[:, fe.IDX_RH] >= 0.5

    chunks = [
        _block_detected_points(pose, pose_ok),
        _block_detected_points(left, left_ok),
        _block_detected_points(right, right_ok),
    ]
    points = [chunk for chunk in chunks if len(chunk) > 0]
    if not points:
        return (-3.0, 3.0), (3.5, -1.5)

    pts = np.vstack(points)
    lo = np.percentile(pts, 2, axis=0)
    hi = np.percentile(pts, 98, axis=0)
    center = (lo + hi) * 0.5
    span = np.maximum(hi - lo, 0.75)
    radius = float(max(span[0], span[1]) * 0.68 + 0.20)
    radius = max(radius, 1.0)
    xlim = (center[0] - radius, center[0] + radius)
    ylim = (center[1] + radius, center[1] - radius)
    return xlim, ylim


def _split_spatial(vector):
    pose = vector[0:18].reshape(6, 3)
    left = vector[18:81].reshape(21, 3)
    right = vector[81:144].reshape(21, 3)
    return pose, left, right


def _legacy_reconstruct(pose, left, right):
    if not np.allclose(pose, 0.0):
        left = left + pose[4]
        right = right + pose[5]
    return pose, left, right


def draw_pose(ax, pose_points, detected=True):
    if np.allclose(pose_points, 0.0):
        return
    alpha = 0.85 if detected else 0.35
    for start, end in POSE_CONNECTIONS:
        ax.plot(
            [pose_points[start, 0], pose_points[end, 0]],
            [pose_points[start, 1], pose_points[end, 1]],
            color="gray",
            linewidth=1.5,
            alpha=alpha,
        )
    ax.scatter(pose_points[:, 0], pose_points[:, 1], color="black", s=5, alpha=alpha)


def draw_hand(ax, hand_points, color, detected=True):
    if np.allclose(hand_points, 0.0):
        return
    alpha = 1.0 if detected else 0.32
    for start, end in fe.HAND_DRAW_CONNECTIONS:
        ax.plot(
            [hand_points[start, 0], hand_points[end, 0]],
            [hand_points[start, 1], hand_points[end, 1]],
            color=color,
            linewidth=2,
            alpha=alpha,
        )
    ax.scatter(hand_points[:, 0], hand_points[:, 1], color="black", s=5, alpha=alpha)


def render_sequence_gif(
    sequence,
    gif_path,
    title,
    feature_version: str | None = None,
    interval: int = 50,
) -> str:
    seq = np.asarray(sequence, dtype=np.float32)
    if seq.ndim != 2 or seq.shape[1] < 144:
        raise ValueError(f"Expected sequence shape (T, >=144), got {seq.shape}")

    os.makedirs(os.path.dirname(gif_path), exist_ok=True)
    is_current = feature_version == fe.FEATURE_SCHEMA
    flags = seq[:, 176:179] if seq.shape[1] >= 179 else np.ones((len(seq), 3), dtype=np.float32)
    xlim, ylim = sequence_axis_limits(seq, flags)

    fig, ax = plt.subplots(figsize=(3, 3))

    def update(frame_idx):
        ax.clear()
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_title(title, fontweight="bold", fontsize=10)
        ax.axis("off")

        pose, left, right = _split_spatial(seq[frame_idx][:144])
        if not is_current:
            pose, left, right = _legacy_reconstruct(pose, left, right)

        draw_pose(ax, pose, detected=bool(flags[frame_idx, fe.IDX_POSE] >= 0.5))
        draw_hand(ax, left, "red", detected=bool(flags[frame_idx, fe.IDX_LH] >= 0.5))
        draw_hand(ax, right, "blue", detected=bool(flags[frame_idx, fe.IDX_RH] >= 0.5))

    anim = animation.FuncAnimation(fig, update, frames=len(seq), interval=interval)
    anim.save(gif_path, writer="pillow")
    plt.close(fig)
    return gif_path


def generate_vocab_gif(vocab_name, force_regenerate=False):
    os.makedirs(GIF_OUTPUT_DIR, exist_ok=True)
    gif_path = os.path.join(GIF_OUTPUT_DIR, f"{vocab_name}.gif")
    vocab_file = os.path.join(DATABASE_DIR, f"{vocab_name}.parquet")

    if not os.path.exists(vocab_file):
        return None

    try:
        df_vocab = pd.read_parquet(vocab_file)
        if df_vocab.empty:
            return None

        video_data = _select_preview_sample(df_vocab)
        if video_data.empty:
            return None
        sequence = np.array([_parse_features(f) for f in video_data["features"]], dtype=np.float32)
        feature_version = None
        if "feature_version" in video_data.columns and not video_data.empty:
            versions = video_data["feature_version"].astype(str)
            if (versions == fe.FEATURE_SCHEMA).all():
                feature_version = fe.FEATURE_SCHEMA
        if os.path.exists(gif_path) and not force_regenerate:
            if os.path.getmtime(gif_path) >= os.path.getmtime(vocab_file):
                return gif_path

        return render_sequence_gif(
            sequence=sequence,
            gif_path=gif_path,
            title=f"Vocab: {vocab_name.upper()}",
            feature_version=feature_version,
        )

    except Exception as e:
        print(f"Error rendering GIF untuk {vocab_name}: {e}")
        return None
