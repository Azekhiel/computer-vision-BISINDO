import os
import hashlib
import json
import subprocess

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

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
        left_accepted = bool(value_or_default(row, "tracking_left_accepted", flags[idx, fe.IDX_LH] >= 0.5))
        right_accepted = bool(value_or_default(row, "tracking_right_accepted", flags[idx, fe.IDX_RH] >= 0.5))
        left_original = bool(value_or_default(row, "tracking_left_original_detected", flags[idx, fe.IDX_LH] >= 0.5))
        right_original = bool(value_or_default(row, "tracking_right_original_detected", flags[idx, fe.IDX_RH] >= 0.5))
        metadata.append(
            {
                "pose_detected": bool(flags[idx, fe.IDX_POSE] >= 0.5),
                "left": {
                    "source": left_source,
                    "confidence": float(value_or_default(row, "tracking_left_confidence", 1.0 if flags[idx, fe.IDX_LH] >= 0.5 else 0.0)),
                    "gap_age": int(value_or_default(row, "tracking_left_gap_age", 0 if flags[idx, fe.IDX_LH] >= 0.5 else 999)),
                    "quality_reason": "parquet_columns",
                    "original_detected": left_original,
                    "accepted": left_accepted,
                    "track_id": int(value_or_default(row, "tracking_left_track_id", -1)),
                    "rendered": left_accepted,
                    "roi": None,
                },
                "right": {
                    "source": right_source,
                    "confidence": float(value_or_default(row, "tracking_right_confidence", 1.0 if flags[idx, fe.IDX_RH] >= 0.5 else 0.0)),
                    "gap_age": int(value_or_default(row, "tracking_right_gap_age", 0 if flags[idx, fe.IDX_RH] >= 0.5 else 999)),
                    "quality_reason": "parquet_columns",
                    "original_detected": right_original,
                    "accepted": right_accepted,
                    "track_id": int(value_or_default(row, "tracking_right_track_id", -1)),
                    "rendered": right_accepted,
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
                    "accepted": bool(left_detected and left_rendered[i]),
                    "track_id": -1,
                    "rendered": bool(left_rendered[i]),
                    "roi": None,
                },
                "right": {
                    "source": "holistic" if right_detected else ("imputed" if right_rendered[i] else "missing"),
                    "confidence": 1.0 if right_detected else (0.35 if right_rendered[i] else 0.0),
                    "gap_age": 0 if right_detected else 999,
                    "quality_reason": "derived_from_tensor",
                    "original_detected": right_detected,
                    "accepted": bool(right_detected and right_rendered[i]),
                    "track_id": -1,
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
    render_mode: str = "bodyframe",
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
        "render_mode": render_mode,
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
        if np.allclose(pose_points[start], 0.0) or np.allclose(pose_points[end], 0.0):
            continue
        ax.plot(
            [pose_points[start, 0], pose_points[end, 0]],
            [pose_points[start, 1], pose_points[end, 1]],
            color="gray",
            linewidth=1.5,
            alpha=alpha,
        )
    visible = np.max(np.abs(pose_points[:, :2]), axis=1) > 1e-6
    if np.any(visible):
        ax.scatter(pose_points[visible, 0], pose_points[visible, 1], color="black", s=5, alpha=alpha)


def _hand_alpha(metadata: dict | None, detected: bool):
    if metadata is None:
        return 1.0 if detected else 0.32
    if not bool(metadata.get("accepted", detected)):
        return 0.0
    source = str(metadata.get("source", "missing"))
    confidence = float(metadata.get("confidence", 0.0))
    if source == "missing" or confidence <= 0.0:
        return 0.0
    base = {
        "holistic": 1.0,
        "hands": 0.82,
        "hands_enhanced": 0.82,
        "hands_raw": 0.72,
        "contact_imputed": 0.45,
        "flow": 0.55,
        "imputed": 0.42,
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


def _blend_rgb(color: tuple[int, int, int], alpha: float, bg: tuple[int, int, int] = (255, 255, 255)):
    alpha = float(np.clip(alpha, 0.0, 1.0))
    return tuple(int(round(bg[i] * (1.0 - alpha) + color[i] * alpha)) for i in range(3))


def _body_point_to_pixel(point, xlim, ylim, width: int, height: int):
    x = float(point[0])
    y = float(point[1])
    px = int(round((x - float(xlim[0])) / max(float(xlim[1] - xlim[0]), 1e-6) * (width - 1)))
    py = int(round((float(ylim[0]) - y) / max(float(ylim[0] - ylim[1]), 1e-6) * (height - 1)))
    return px, py


def _draw_body_pose_pil(draw: ImageDraw.ImageDraw, pose_points, xlim, ylim, width: int, height: int, detected=True):
    if np.allclose(pose_points, 0.0):
        return
    alpha = 0.85 if detected else 0.35
    color = _blend_rgb((120, 120, 120), alpha)
    for start, end in POSE_CONNECTIONS:
        if np.allclose(pose_points[start], 0.0) or np.allclose(pose_points[end], 0.0):
            continue
        draw.line(
            [
                _body_point_to_pixel(pose_points[start], xlim, ylim, width, height),
                _body_point_to_pixel(pose_points[end], xlim, ylim, width, height),
            ],
            fill=color,
            width=2,
        )
    visible = np.max(np.abs(pose_points[:, :2]), axis=1) > 1e-6
    for point in pose_points[visible]:
        px, py = _body_point_to_pixel(point, xlim, ylim, width, height)
        draw.ellipse((px - 2, py - 2, px + 2, py + 2), fill=_blend_rgb((0, 0, 0), alpha))


def _draw_body_hand_pil(draw: ImageDraw.ImageDraw, hand_points, color, xlim, ylim, width: int, height: int, alpha: float):
    if np.allclose(hand_points, 0.0) or alpha <= 0.0:
        return
    line_color = _blend_rgb(color, alpha)
    point_color = _blend_rgb((0, 0, 0), alpha)
    for start, end in fe.HAND_DRAW_CONNECTIONS:
        draw.line(
            [
                _body_point_to_pixel(hand_points[start], xlim, ylim, width, height),
                _body_point_to_pixel(hand_points[end], xlim, ylim, width, height),
            ],
            fill=line_color,
            width=2,
        )
    for point in hand_points:
        px, py = _body_point_to_pixel(point, xlim, ylim, width, height)
        draw.ellipse((px - 2, py - 2, px + 2, py + 2), fill=point_color)


def _render_body_frame_pil(seq, frame_idx: int, title: str, flags, tracking_metadata, is_current: bool, xlim, ylim):
    width = height = 300
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((82, 8), title, fill=(0, 0, 0))

    pose, left, right = _split_spatial(seq[frame_idx][:144])
    if not is_current:
        pose, left, right = _legacy_reconstruct(pose, left, right)

    meta = tracking_metadata[frame_idx]
    left_meta = meta.get("left", {})
    right_meta = meta.get("right", {})
    left_detected = bool(flags[frame_idx, fe.IDX_LH] >= 0.5)
    right_detected = bool(flags[frame_idx, fe.IDX_RH] >= 0.5)
    if is_current and not left_detected:
        left = np.zeros_like(left)
    if is_current and not right_detected:
        right = np.zeros_like(right)

    _draw_body_pose_pil(draw, pose, xlim, ylim, width, height, detected=bool(flags[frame_idx, fe.IDX_POSE] >= 0.5))
    _draw_body_hand_pil(draw, left, (220, 30, 30), xlim, ylim, width, height, _hand_alpha(left_meta, left_detected))
    _draw_body_hand_pil(draw, right, (35, 70, 220), xlim, ylim, width, height, _hand_alpha(right_meta, right_detected))
    return image


def _save_gif_frames(frames: list[Image.Image], gif_path: str, interval: int):
    if not frames:
        return
    os.makedirs(os.path.dirname(gif_path), exist_ok=True)
    first, rest = frames[0], frames[1:]
    first.save(
        gif_path,
        save_all=True,
        append_images=rest,
        duration=int(interval),
        loop=0,
        disposal=2,
        optimize=False,
    )


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

    if not cached:
        frames = [
            _render_body_frame_pil(seq, frame_idx, title, flags, tracking_metadata, is_current, xlim, ylim)
            for frame_idx in range(len(seq))
        ]
        _save_gif_frames(frames, gif_path, interval)
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
        render_mode="bodyframe",
    )
    return gif_path


def _roi_to_pixels(roi, width: int, height: int):
    if not roi or len(roi) != 4:
        return None
    x0, y0, x1, y1 = [float(v) for v in roi]
    return (
        int(np.clip(x0, -0.25, 1.25) * width),
        int(np.clip(y0, -0.25, 1.25) * height),
        int(np.clip(x1, -0.25, 1.25) * width),
        int(np.clip(y1, -0.25, 1.25) * height),
    )


def _draw_tracking_roi(frame: np.ndarray, metadata: dict, side: str):
    hand = metadata.get(side, {}) if isinstance(metadata, dict) else {}
    roi = _roi_to_pixels(hand.get("roi"), frame.shape[1], frame.shape[0])
    if roi is None:
        return
    accepted = bool(hand.get("accepted", hand.get("rendered", False)))
    original = bool(hand.get("original_detected", False))
    source = str(hand.get("source", "missing"))
    confidence = float(hand.get("confidence", 0.0))
    color = (40, 40, 220) if side == "left" else (220, 80, 40)
    if not accepted:
        color = (120, 120, 120)
    x0, y0, x1, y1 = roi
    cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2 if accepted else 1)
    label = f"{side[0].upper()} {source} {confidence:.2f}"
    if original:
        label += " orig"
    if accepted:
        label += " acc"
    cv2.putText(frame, label, (x0, max(18, y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def _draw_candidate_rois(frame: np.ndarray, metadata: dict):
    accepted_rois = []
    for side in ("left", "right"):
        hand = metadata.get(side, {}) if isinstance(metadata, dict) else {}
        if bool(hand.get("accepted", False)) and hand.get("roi"):
            accepted_rois.append(tuple(float(v) for v in hand.get("roi")))

    for candidate in metadata.get("candidates", []) if isinstance(metadata, dict) else []:
        roi = _roi_to_pixels(candidate.get("roi"), frame.shape[1], frame.shape[0])
        if roi is None:
            continue
        norm_roi = candidate.get("roi")
        is_accepted = any(fe._roi_iou(norm_roi, accepted) > 0.72 for accepted in accepted_rois) if norm_roi else False
        color = (0, 220, 220) if is_accepted else (80, 80, 80)
        x0, y0, x1, y1 = roi
        cv2.rectangle(frame, (x0, y0), (x1, y1), color, 1)
        label = f"cand {candidate.get('source', '?')} {float(candidate.get('score', 0.0)):.2f}"
        cv2.putText(frame, label, (x0, min(frame.shape[0] - 6, y1 + 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)


def render_pixel_overlay_gif(
    raw_video_path: str,
    gif_path: str,
    title: str,
    tracking_metadata: list[dict],
    flags: np.ndarray,
    source_frame_indices: list[int],
    feature_version: str | None = None,
    interval: int = 50,
    video_id: str | None = None,
    vocab_name: str | None = None,
    parquet_path: str | None = None,
    cached: bool = False,
) -> str | None:
    if cv2 is None or not raw_video_path or not os.path.exists(raw_video_path):
        return None
    os.makedirs(os.path.dirname(gif_path), exist_ok=True)
    cap = cv2.VideoCapture(raw_video_path)
    if not cap.isOpened():
        return None

    frames: list[Image.Image] = []
    for idx, source_idx in enumerate(source_frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(source_idx))
        ok, frame = cap.read()
        if not ok:
            continue
        meta = tracking_metadata[idx] if idx < len(tracking_metadata) else {}
        _draw_candidate_rois(frame, meta)
        _draw_tracking_roi(frame, meta, "left")
        _draw_tracking_roi(frame, meta, "right")
        cv2.putText(
            frame,
            f"{title} f{source_idx}",
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        image.thumbnail((480, 360))
        canvas = Image.new("RGB", (480, 360), "#111")
        canvas.paste(image, ((480 - image.width) // 2, (360 - image.height) // 2))
        frames.append(canvas)
    cap.release()
    if not frames:
        return None

    if not cached:
        _save_gif_frames(frames, gif_path, interval)

    dummy_seq = np.zeros((len(frames), fe.N_TOTAL_WITH_FLAGS), dtype=np.float32)
    log_flags = np.asarray(flags, dtype=np.float32)
    if len(log_flags) != len(frames):
        log_flags = np.ones((len(frames), 3), dtype=np.float32)
    _write_generation_log(
        gif_path=gif_path,
        title=title,
        sequence=dummy_seq,
        flags=log_flags,
        tracking_metadata=tracking_metadata[: len(frames)],
        feature_version=feature_version,
        interval=interval,
        video_id=video_id,
        vocab_name=vocab_name,
        raw_video_path=raw_video_path,
        source_frame_indices=source_frame_indices[: len(frames)],
        parquet_path=parquet_path,
        xlim=(0, frames[0].size[0]),
        ylim=(frames[0].size[1], 0),
        cached=cached,
        render_mode="pixel_overlay",
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


def generate_vocab_pixel_overlay_gif(vocab_name, force_regenerate=False):
    overlay_dir = os.path.join(GIF_OUTPUT_DIR, "pixel_overlay")
    os.makedirs(overlay_dir, exist_ok=True)
    gif_path = os.path.join(overlay_dir, f"{vocab_name}.gif")
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
        flags = sequence[:, fe.SLICE_FLAGS] if sequence.shape[1] >= fe.N_TOTAL_WITH_FLAGS else np.ones((len(sequence), 3), dtype=np.float32)
        tracking_metadata = _metadata_from_rows(video_data, flags)
        video_id = str(video_data["video_id"].iloc[0]) if "video_id" in video_data.columns else None
        split = str(video_data["split"].iloc[0]) if "split" in video_data.columns else None
        raw_video_path = _resolve_raw_video_path(video_id, vocab_name, split)
        if not raw_video_path or not os.path.exists(raw_video_path):
            return None
        source_frame_indices = (
            [int(x) for x in video_data["source_frame_num"].tolist()]
            if "source_frame_num" in video_data.columns
            else [int(x) for x in video_data["frame_num"].tolist()]
        )
        if os.path.exists(gif_path) and force_regenerate:
            os.remove(gif_path)
        cached = bool(os.path.exists(gif_path) and not force_regenerate)
        feature_version = None
        if "feature_version" in video_data.columns and (video_data["feature_version"].astype(str) == fe.FEATURE_SCHEMA).all():
            feature_version = fe.FEATURE_SCHEMA
        return render_pixel_overlay_gif(
            raw_video_path=raw_video_path,
            gif_path=gif_path,
            title=f"Overlay: {vocab_name.upper()}",
            tracking_metadata=tracking_metadata,
            flags=flags,
            source_frame_indices=source_frame_indices,
            feature_version=feature_version,
            video_id=video_id,
            vocab_name=vocab_name,
            parquet_path=vocab_file,
            cached=cached,
        )
    except Exception as e:
        print(f"Error rendering overlay GIF untuk {vocab_name}: {e}")
        return None
