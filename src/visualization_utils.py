import os
import hashlib
import json
import subprocess

import matplotlib
matplotlib.use("Agg")

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import cv2
except Exception:  # pragma: no cover - GIF rendering can run without video probing.
    cv2 = None

import feature_engine as fe

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATABASE_DIR = os.path.join(ROOT_DIR, "dataset_parquets")
GIF_OUTPUT_DIR = os.path.join(ROOT_DIR, "assets", "gifs")
GIF_GENERATION_LOG = os.path.join(GIF_OUTPUT_DIR, "generation_log.jsonl")

POSE_CONNECTIONS = [(0, 1), (0, 2), (2, 4), (1, 3), (3, 5)]


def _parse_features(feature_value):
    if isinstance(feature_value, str):
        return np.fromstring(feature_value, sep=",", dtype=np.float32)
    return np.asarray(feature_value, dtype=np.float32)


def _safe_json_load(value, default=None):
    if default is None:
        default = {}
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return default
    if isinstance(value, dict):
        return value
    try:
        return json.loads(str(value))
    except Exception:
        return default


def _git_sha():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT_DIR,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2,
        ).strip()
    except Exception:
        return None


def _file_sha1(path: str | None):
    if not path or not os.path.exists(path):
        return None
    digest = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _video_probe(path: str | None):
    if not path or not os.path.exists(path) or cv2 is None:
        return None
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    info = {
        "fps": float(cap.get(cv2.CAP_PROP_FPS) or 0.0),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
    }
    cap.release()
    return info


def _resolve_raw_video_path(video_id: str | None, vocab_name: str | None, split: str | None = None):
    if not video_id or not vocab_name:
        return None

    media_name = None
    video_id = str(video_id)
    if "_manual_" in video_id:
        media_name = video_id.split("_manual_", 1)[1]
    elif "_generate_" in video_id:
        media_name = video_id.split("_generate_", 1)[1].split("_aug_", 1)[0]

    if not media_name or "." not in media_name:
        return None

    candidates = []
    split_candidates = [split] if split else []
    split_candidates += ["train", "val", "test"]
    seen = set()
    for split_name in split_candidates:
        if not split_name or split_name in seen:
            continue
        seen.add(split_name)
        candidates.append(os.path.join(ROOT_DIR, "record", "video", split_name, vocab_name, media_name))

    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[0] if candidates else None


def _metadata_from_rows(rows: pd.DataFrame, flags: np.ndarray) -> list[dict]:
    def value_or_default(row, key, default):
        value = row.get(key, default)
        try:
            if pd.isna(value):
                return default
        except Exception:
            pass
        return value

    metadata = []
    for idx, (_, row) in enumerate(rows.iterrows()):
        parsed = _safe_json_load(row.get("tracking_metadata"), default=None)
        if parsed:
            metadata.append(parsed)
            continue

        left_source = str(value_or_default(row, "tracking_left_source", "holistic" if flags[idx, fe.IDX_LH] >= 0.5 else "missing"))
        right_source = str(value_or_default(row, "tracking_right_source", "holistic" if flags[idx, fe.IDX_RH] >= 0.5 else "missing"))
        metadata.append(
            {
                "pose_detected": bool(flags[idx, fe.IDX_POSE] >= 0.5),
                "left": {
                    "source": left_source,
                    "confidence": float(value_or_default(row, "tracking_left_confidence", 1.0 if flags[idx, fe.IDX_LH] >= 0.5 else 0.0)),
                    "gap_age": int(value_or_default(row, "tracking_left_gap_age", 0 if flags[idx, fe.IDX_LH] >= 0.5 else 999)),
                    "quality_reason": "parquet_columns",
                    "original_detected": bool(flags[idx, fe.IDX_LH] >= 0.5),
                    "rendered": bool(flags[idx, fe.IDX_LH] >= 0.5),
                    "roi": None,
                },
                "right": {
                    "source": right_source,
                    "confidence": float(value_or_default(row, "tracking_right_confidence", 1.0 if flags[idx, fe.IDX_RH] >= 0.5 else 0.0)),
                    "gap_age": int(value_or_default(row, "tracking_right_gap_age", 0 if flags[idx, fe.IDX_RH] >= 0.5 else 999)),
                    "quality_reason": "parquet_columns",
                    "original_detected": bool(flags[idx, fe.IDX_RH] >= 0.5),
                    "rendered": bool(flags[idx, fe.IDX_RH] >= 0.5),
                    "roi": None,
                },
            }
        )
    return metadata


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


def _block_rendered(block: np.ndarray) -> np.ndarray:
    pts = np.asarray(block, dtype=np.float32).reshape(len(block), -1, 3)
    return np.max(np.linalg.norm(pts[:, :, :2], axis=2), axis=1) > 1e-6


def _default_tracking_metadata(sequence: np.ndarray, flags: np.ndarray) -> list[dict]:
    seq = np.asarray(sequence, dtype=np.float32)
    left_rendered = _block_rendered(seq[:, fe.SLICE_LH].reshape(len(seq), 21, 3))
    right_rendered = _block_rendered(seq[:, fe.SLICE_RH].reshape(len(seq), 21, 3))
    metadata = []
    for i in range(len(seq)):
        left_detected = bool(flags[i, fe.IDX_LH] >= 0.5)
        right_detected = bool(flags[i, fe.IDX_RH] >= 0.5)
        metadata.append(
            {
                "pose_detected": bool(flags[i, fe.IDX_POSE] >= 0.5),
                "left": {
                    "source": "holistic" if left_detected else ("imputed" if left_rendered[i] else "missing"),
                    "confidence": 1.0 if left_detected else (0.35 if left_rendered[i] else 0.0),
                    "gap_age": 0 if left_detected else 999,
                    "quality_reason": "derived_from_tensor",
                    "original_detected": left_detected,
                    "rendered": bool(left_rendered[i]),
                    "roi": None,
                },
                "right": {
                    "source": "holistic" if right_detected else ("imputed" if right_rendered[i] else "missing"),
                    "confidence": 1.0 if right_detected else (0.35 if right_rendered[i] else 0.0),
                    "gap_age": 0 if right_detected else 999,
                    "quality_reason": "derived_from_tensor",
                    "original_detected": right_detected,
                    "rendered": bool(right_rendered[i]),
                    "roi": None,
                },
            }
        )
    return metadata


def _write_generation_log(
    gif_path: str,
    title: str,
    sequence: np.ndarray,
    flags: np.ndarray,
    tracking_metadata: list[dict],
    feature_version: str | None,
    interval: int,
    video_id: str | None,
    vocab_name: str | None,
    raw_video_path: str | None,
    source_frame_indices: list[int] | None,
    parquet_path: str | None,
    xlim,
    ylim,
    cached: bool,
):
    if source_frame_indices is None or len(source_frame_indices) != len(sequence):
        source_frame_indices = list(range(len(sequence)))

    parquet_stat = None
    if parquet_path and os.path.exists(parquet_path):
        parquet_stat = {
            "mtime": os.path.getmtime(parquet_path),
            "sha1": _file_sha1(parquet_path),
        }

    record = {
        "event": "gif_generation",
        "cached": bool(cached),
        "gif_path": os.path.abspath(gif_path),
        "title": title,
        "feature_schema": feature_version or fe.LEGACY_SCHEMA,
        "code_schema": fe.FEATURE_SCHEMA,
        "git_sha": _git_sha(),
        "parquet_path": os.path.abspath(parquet_path) if parquet_path else None,
        "parquet": parquet_stat,
        "raw_video_path": os.path.abspath(raw_video_path) if raw_video_path else None,
        "raw_video_exists": bool(raw_video_path and os.path.exists(raw_video_path)),
        "raw_video": _video_probe(raw_video_path),
        "video_id": video_id,
        "vocab_name": vocab_name,
        "sequence_length": int(len(sequence)),
        "gif_frame_count": int(len(sequence)),
        "source_frame_indices": [int(x) for x in source_frame_indices],
        "interval_ms": int(interval),
        "axis": {
            "xlim": [float(xlim[0]), float(xlim[1])],
            "ylim": [float(ylim[0]), float(ylim[1])],
        },
        "flags": flags.astype(float).tolist(),
        "tracking": tracking_metadata,
    }
    fe.TrackingRegistryWriter(GIF_GENERATION_LOG).write(record)


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


def _hand_alpha(metadata: dict | None, detected: bool):
    if metadata is None:
        return 1.0 if detected else 0.32
    source = str(metadata.get("source", "missing"))
    confidence = float(metadata.get("confidence", 0.0))
    if source == "missing" or confidence <= 0.0:
        return 0.0
    base = {
        "holistic": 1.0,
        "hands": 0.82,
        "flow": 0.55,
        "prediction": 0.32,
    }.get(source, 0.32 if not detected else 0.75)
    return float(np.clip(base * max(confidence, 0.25), 0.18, 1.0))


def draw_hand(ax, hand_points, color, detected=True, alpha=None):
    if np.allclose(hand_points, 0.0):
        return
    alpha = (1.0 if detected else 0.32) if alpha is None else float(alpha)
    if alpha <= 0.0:
        return
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
    tracking_metadata: list[dict] | None = None,
    video_id: str | None = None,
    vocab_name: str | None = None,
    raw_video_path: str | None = None,
    source_frame_indices: list[int] | None = None,
    parquet_path: str | None = None,
    cached: bool = False,
) -> str:
    seq = np.asarray(sequence, dtype=np.float32)
    if seq.ndim != 2 or seq.shape[1] < 144:
        raise ValueError(f"Expected sequence shape (T, >=144), got {seq.shape}")

    os.makedirs(os.path.dirname(gif_path), exist_ok=True)
    is_current = feature_version == fe.FEATURE_SCHEMA
    flags = seq[:, 176:179] if seq.shape[1] >= 179 else np.ones((len(seq), 3), dtype=np.float32)
    if tracking_metadata is None or len(tracking_metadata) != len(seq):
        tracking_metadata = _default_tracking_metadata(seq, flags)
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

        meta = tracking_metadata[frame_idx]
        left_meta = meta.get("left", {})
        right_meta = meta.get("right", {})
        left_detected = bool(flags[frame_idx, fe.IDX_LH] >= 0.5)
        right_detected = bool(flags[frame_idx, fe.IDX_RH] >= 0.5)

        draw_pose(ax, pose, detected=bool(flags[frame_idx, fe.IDX_POSE] >= 0.5))
        draw_hand(ax, left, "red", detected=left_detected, alpha=_hand_alpha(left_meta, left_detected))
        draw_hand(ax, right, "blue", detected=right_detected, alpha=_hand_alpha(right_meta, right_detected))

    if not cached:
        anim = animation.FuncAnimation(fig, update, frames=len(seq), interval=interval)
        anim.save(gif_path, writer="pillow")
    plt.close(fig)
    _write_generation_log(
        gif_path=gif_path,
        title=title,
        sequence=seq,
        flags=flags,
        tracking_metadata=tracking_metadata,
        feature_version=feature_version,
        interval=interval,
        video_id=video_id,
        vocab_name=vocab_name,
        raw_video_path=raw_video_path,
        source_frame_indices=source_frame_indices,
        parquet_path=parquet_path,
        xlim=xlim,
        ylim=ylim,
        cached=cached,
    )
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
        flags = sequence[:, 176:179] if sequence.shape[1] >= 179 else np.ones((len(sequence), 3), dtype=np.float32)
        tracking_metadata = _metadata_from_rows(video_data, flags)
        video_id = str(video_data["video_id"].iloc[0]) if "video_id" in video_data.columns else None
        split = str(video_data["split"].iloc[0]) if "split" in video_data.columns else None
        raw_video_path = _resolve_raw_video_path(video_id, vocab_name, split)
        source_frame_indices = (
            [int(x) for x in video_data["source_frame_num"].tolist()]
            if "source_frame_num" in video_data.columns
            else [int(x) for x in video_data["frame_num"].tolist()]
        )
        feature_version = None
        if "feature_version" in video_data.columns and not video_data.empty:
            versions = video_data["feature_version"].astype(str)
            if (versions == fe.FEATURE_SCHEMA).all():
                feature_version = fe.FEATURE_SCHEMA
        if os.path.exists(gif_path) and not force_regenerate:
            if os.path.getmtime(gif_path) >= os.path.getmtime(vocab_file):
                render_sequence_gif(
                    sequence=sequence,
                    gif_path=gif_path,
                    title=f"Vocab: {vocab_name.upper()}",
                    feature_version=feature_version,
                    tracking_metadata=tracking_metadata,
                    video_id=video_id,
                    vocab_name=vocab_name,
                    raw_video_path=raw_video_path,
                    source_frame_indices=source_frame_indices,
                    parquet_path=vocab_file,
                    cached=True,
                )
                return gif_path
        if os.path.exists(gif_path) and force_regenerate:
            os.remove(gif_path)

        return render_sequence_gif(
            sequence=sequence,
            gif_path=gif_path,
            title=f"Vocab: {vocab_name.upper()}",
            feature_version=feature_version,
            tracking_metadata=tracking_metadata,
            video_id=video_id,
            vocab_name=vocab_name,
            raw_video_path=raw_video_path,
            source_frame_indices=source_frame_indices,
            parquet_path=vocab_file,
        )

    except Exception as e:
        print(f"Error rendering GIF untuk {vocab_name}: {e}")
        return None
