"""Terminal-first BISINDO workflow CLI.

This module intentionally does not import Tkinter or the live camera stack at
module import time. Heavy pieces are loaded only by the commands that need them.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
import os
from pathlib import Path
import queue
import shutil
import subprocess
import sys
import time
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd

import feature_schemas as fs
import gru_manager as gm
import jetson_runtime as jr
from smart_extract import contract as sc


ROOT_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT_DIR / "dataset_parquets"
MODEL_DIR = ROOT_DIR / "models"
GIF_DIR = ROOT_DIR / "assets" / "gifs"
BACKUP_ROOT = ROOT_DIR / "backups"
SPLITS = {"train", "val", "test"}
MEDIA_EXTS = {".mkv", ".mp4", ".avi", ".mov", ".webm", ".m4v"}
DEFAULT_LIVE_PROFILE = "lossless1080_10"
LIVE_PROFILE_CHOICES = ["accurate10", "fast10", "jetson10", "lite", "ultra", "fast", "quality", "lossless1080_10"]
SCHEMA_CHOICES = [*fs.SCHEMA_NAMES, "all", "base", "original", "face", "faceref", "full", "extra"]


def _split_values(value: str | None, *, allow_all: bool = True) -> list[str]:
    raw = str(value or ("all" if allow_all else "train")).strip().lower().replace(";", ",")
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    if not parts:
        parts = ["all" if allow_all else "train"]
    if allow_all and "all" in parts:
        return sorted(SPLITS)
    invalid = [part for part in parts if part not in SPLITS]
    if invalid:
        raise ValueError("split harus salah satu: train, val, test, all, atau comma list split")
    out: list[str] = []
    for part in parts:
        if part not in out:
            out.append(part)
    return out


@dataclass(frozen=True)
class ImportItem:
    video_path: Path
    label: str
    split: str
    video_id: str


@dataclass
class ImportResult:
    scanned: int = 0
    imported: int = 0
    skipped: int = 0
    failed: int = 0
    rows: int = 0
    backup_dir: Path | None = None
    gif_paths: list[Path] | None = None


@dataclass
class AugmentResult:
    generated: int = 0
    skipped: int = 0
    rows: int = 0
    backup_dir: Path | None = None


@dataclass
class DeleteAugmentResult:
    deleted_sequences: int = 0
    deleted_rows: int = 0
    files_updated: int = 0
    scanned_files: int = 0
    skipped: int = 0
    backup_dir: Path | None = None
    dry_run: bool = False


def clean_label(value: str) -> str:
    return str(value or "").strip().replace(" ", "_").lower()


def is_media_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in MEDIA_EXTS


def _media_files(folder: Path) -> list[Path]:
    return sorted(path for path in folder.iterdir() if is_media_file(path)) if folder.exists() else []


def _video_id(split: str, video_path: Path) -> str:
    return f"{split}_manual_{video_path.name}"


def scan_import_items(
    paths: Sequence[str | Path],
    default_split: str = "train",
    label: str | None = None,
) -> list[ImportItem]:
    """Scan supported import layouts into concrete video import tasks."""

    default_split = str(default_split or "train").lower()
    if default_split not in SPLITS:
        raise ValueError("split harus salah satu: train, val, test")
    fixed_label = clean_label(label) if label else None
    items: list[ImportItem] = []
    seen: set[tuple[str, str, str]] = set()

    def add(video_path: Path, item_label: str, item_split: str) -> None:
        item_label = clean_label(item_label)
        item_split = str(item_split or default_split).lower()
        if item_split not in SPLITS:
            item_split = default_split
        if not item_label:
            raise ValueError(f"Label kosong untuk {video_path}")
        key = (str(video_path.resolve()), item_label, item_split)
        if key in seen:
            return
        seen.add(key)
        items.append(ImportItem(video_path=video_path, label=item_label, split=item_split, video_id=_video_id(item_split, video_path)))

    for raw_path in paths:
        path = Path(raw_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Path tidak ditemukan: {path}")
        if is_media_file(path):
            if not fixed_label:
                raise ValueError("Import single video butuh --label.")
            add(path, fixed_label, default_split)
            continue
        if not path.is_dir():
            continue

        direct_media = _media_files(path)
        if direct_media:
            split = path.parent.name.lower() if path.parent.name.lower() in SPLITS else default_split
            item_label = fixed_label or path.name
            for video_path in direct_media:
                add(video_path, item_label, split)
            continue

        if fixed_label:
            for video_path in sorted(path.rglob("*")):
                if is_media_file(video_path):
                    split = video_path.parent.name.lower() if video_path.parent.name.lower() in SPLITS else default_split
                    add(video_path, fixed_label, split)
            continue

        split_dirs = [child for child in sorted(path.iterdir()) if child.is_dir() and child.name.lower() in SPLITS]
        if split_dirs:
            for split_dir in split_dirs:
                split = split_dir.name.lower()
                for vocab_dir in sorted(child for child in split_dir.iterdir() if child.is_dir()):
                    for video_path in _media_files(vocab_dir):
                        add(video_path, vocab_dir.name, split)
            continue

        for vocab_dir in sorted(child for child in path.iterdir() if child.is_dir()):
            for video_path in _media_files(vocab_dir):
                add(video_path, vocab_dir.name, default_split)

    return items


def _existing_video_ids(parquet_path: Path) -> set[str]:
    if not parquet_path.exists():
        return set()
    try:
        df = pd.read_parquet(parquet_path, columns=["video_id"])
    except TypeError:
        df = pd.read_parquet(parquet_path)
    except Exception:
        return set()
    if "video_id" not in df.columns:
        return set()
    return {str(value) for value in df["video_id"].dropna().unique()}


def _existing_video_ids_for_label(dataset_root: Path, label: str) -> set[str]:
    label = clean_label(label)
    if not dataset_root.exists():
        return set()
    ids: set[str] = set()
    for parquet_path in sorted(dataset_root.glob("*.parquet")):
        try:
            df = pd.read_parquet(parquet_path, columns=["label", "video_id"])
        except Exception:
            try:
                df = pd.read_parquet(parquet_path)
            except Exception:
                continue
        if "video_id" not in df.columns:
            continue
        if "label" in df.columns:
            df = df[df["label"].astype(str).map(clean_label) == label]
        elif clean_label(parquet_path.stem) != label:
            continue
        ids.update(str(value) for value in df["video_id"].dropna().unique())
    return ids


def _backup_existing(parquet_path: Path, backup_dir: Path, copied: set[Path]) -> None:
    if not parquet_path.exists() or parquet_path in copied:
        return
    backup_dir.mkdir(parents=True, exist_ok=True)
    try:
        rel = parquet_path.resolve().relative_to(ROOT_DIR.resolve())
    except Exception:
        rel = Path(parquet_path.parent.name) / parquet_path.name
    dest = backup_dir / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(parquet_path, dest)
    copied.add(parquet_path)
    print(f"[BACKUP] {parquet_path} -> {dest}", flush=True)


def _rows_from_extract_result(item: ImportItem, result: dict, schema: str = fs.DEFAULT_SCHEMA) -> list[dict]:
    schema_spec = fs.get_schema(schema)
    features = fs.ensure_feature_dim(result["features"], schema_spec)
    frames = list(result.get("frames", []))
    rows: list[dict] = []
    for frame_num, vec in enumerate(features):
        meta = frames[frame_num] if frame_num < len(frames) else {}
        rows.append(
            {
                "video_id": item.video_id,
                "label": item.label,
                "frame_num": int(frame_num),
                "split": item.split,
                "schema": schema_spec.name,
                "feature_version": schema_spec.feature_schema,
                "feature_mode": str(result.get("feature_mode", schema_spec.feature_mode)),
                "feature_dim": schema_spec.feature_dim,
                "target_fps": schema_spec.target_fps,
                "extract_profile": str(result.get("extract_profile", schema_spec.extractor)),
                "source_media_path": str(item.video_path),
                "source_frame_num": int(meta.get("target_source_frame", frame_num)),
                "chosen_source_frame": int(meta.get("chosen_source_frame", meta.get("target_source_frame", frame_num))),
                "time_sec": float(meta.get("time_sec", frame_num / sc.TARGET_FPS)),
                "smart_mode": str(result.get("smart_mode", "best")),
                "enhance_mode": str(meta.get("enhance_mode", "")),
                "left_present": float(meta.get("left_present", 0.0)),
                "right_present": float(meta.get("right_present", 0.0)),
                "left_detected": float(meta.get("left_detected", 0.0)),
                "right_detected": float(meta.get("right_detected", 0.0)),
                "left_held": float(meta.get("left_held", 0.0)),
                "right_held": float(meta.get("right_held", 0.0)),
                "left_score": float(meta.get("left_score", 0.0)),
                "right_score": float(meta.get("right_score", 0.0)),
                "features": sc.format_feature_value(vec),
            }
        )
    return rows


def _save_import_gifs(item: ImportItem, result: dict, schema: str = fs.DEFAULT_SCHEMA, gif_width: int = 420) -> list[Path]:
    from smart_extract.extract_video_smart_v8 import save_gif

    paths = fs.sample_gif_paths(schema, item.label, item.video_id, ROOT_DIR, modes=("overlay", "skeleton"))
    overlay_path = Path(paths["overlay"])
    skeleton_path = Path(paths["skeleton"])
    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    fps = float(result.get("target_fps", sc.TARGET_FPS))
    save_gif(result.get("overlay_frames", []), overlay_path, fps, int(gif_width))
    save_gif(result.get("skeleton_frames", []), skeleton_path, fps, int(gif_width))
    return [overlay_path, skeleton_path]


def _make_extract_args_for_schema(schema: str, quiet: bool = False, save_gif: bool = False):
    schema_spec = fs.get_schema(schema)
    if schema_spec.name == "smart180":
        from smart_extract.extract_video_smart_v8 import make_best_extract_args

        return make_best_extract_args(quiet=quiet, save_gif=save_gif, no_gif=not save_gif)

    import holistic_features

    return holistic_features.make_holistic_args(
        quiet=quiet,
        target_fps=schema_spec.target_fps,
        width=sc.BEST_EXTRACT_SETTINGS["width"],
        height=sc.BEST_EXTRACT_SETTINGS["height"],
        proc_width=sc.BEST_EXTRACT_SETTINGS["proc_width"],
        det_conf=sc.BEST_EXTRACT_SETTINGS["det_conf"],
        track_conf=sc.BEST_EXTRACT_SETTINGS["track_conf"],
        gif_width=sc.BEST_EXTRACT_SETTINGS["gif_width"],
    )


def _extract_video_for_schema(video_path: Path, schema: str, args, include_frames: bool) -> dict | None:
    schema_spec = fs.get_schema(schema)
    if schema_spec.name == "smart180":
        from smart_extract.extract_video_smart_v8 import extract_video_arrays

        return extract_video_arrays(video_path, args=args, include_frames=include_frames)

    import holistic_features

    return holistic_features.extract_video_arrays(video_path, schema=schema_spec.name, args=args, include_frames=include_frames)


def append_import_items(
    items: Sequence[ImportItem],
    dataset_dir: str | Path = DATASET_DIR,
    backup_root: str | Path = BACKUP_ROOT,
    schema: str = fs.DEFAULT_SCHEMA,
    save_gif: bool = False,
    quiet: bool = False,
    extractor: Callable[[Path, object, bool], dict | None] | None = None,
    overwrite_existing: bool = False,
) -> ImportResult:
    """Append schema rows safely, skipping duplicate video_id per vocab."""

    schema_spec = fs.get_schema(schema)
    dataset_root = fs.dataset_dir_for(schema_spec, dataset_dir)
    dataset_root.mkdir(parents=True, exist_ok=True)
    backup_dir = Path(backup_root) / f"bisindo_cli_import_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    copied: set[Path] = set()
    rows_by_label: dict[str, list[dict]] = {}
    replace_ids_by_label: dict[str, set[str]] = {}
    known_ids_by_label: dict[str, set[str]] = {}
    gif_paths: list[Path] = []
    result = ImportResult(scanned=len(items), gif_paths=gif_paths)

    for idx, item in enumerate(items, 1):
        parquet_path = dataset_root / f"{item.label}.parquet"
        known_ids = known_ids_by_label.setdefault(item.label, _existing_video_ids_for_label(dataset_root, item.label))
        if item.video_id in known_ids and not overwrite_existing:
            result.skipped += 1
            if not quiet:
                print(f"[SKIP existing video_id] {schema_spec.name}/{item.label}/{item.video_id}", flush=True)
            continue
        if item.video_id in known_ids:
            replace_ids_by_label.setdefault(item.label, set()).add(item.video_id)
            if not quiet:
                print(f"[OVERWRITE existing video_id] {schema_spec.name}/{item.label}/{item.video_id}", flush=True)
        if not quiet:
            print(f"[IMPORT] {schema_spec.name} {idx}/{len(items)} {item.split}/{item.label}/{item.video_path.name}")
        args = _make_extract_args_for_schema(schema_spec.name, quiet=quiet, save_gif=save_gif)
        if extractor is not None:
            extracted = extractor(item.video_path, args=args, include_frames=save_gif)
        else:
            extracted = _extract_video_for_schema(item.video_path, schema_spec.name, args=args, include_frames=save_gif)
        if extracted is None:
            result.failed += 1
            continue
        try:
            rows = _rows_from_extract_result(item, extracted, schema=schema_spec.name)
        except Exception as exc:
            result.failed += 1
            print(f"[FAIL] {item.video_path}: {exc}")
            continue
        rows_by_label.setdefault(item.label, []).extend(rows)
        known_ids.add(item.video_id)
        result.imported += 1
        result.rows += len(rows)
        if save_gif:
            gif_paths.extend(_save_import_gifs(item, extracted, schema=schema_spec.name, gif_width=int(args.gif_width)))

    for label, rows in rows_by_label.items():
        parquet_path = dataset_root / f"{label}.parquet"
        _backup_existing(parquet_path, backup_dir, copied)
        df_new = pd.DataFrame(rows)
        if parquet_path.exists():
            df_old = pd.read_parquet(parquet_path)
            replace_ids = replace_ids_by_label.get(label, set())
            if replace_ids and "video_id" in df_old.columns:
                mask = df_old["video_id"].astype(str).isin(replace_ids)
                if "label" in df_old.columns:
                    mask = mask & df_old["label"].astype(str).map(clean_label).eq(label)
                df_old = df_old.loc[~mask].copy()
            df_out = pd.concat([df_old, df_new], ignore_index=True)
        else:
            df_out = df_new
        df_out.to_parquet(parquet_path, index=False)
        print(f"[APPEND parquet +{len(rows)} rows] {parquet_path}", flush=True)

    result.backup_dir = backup_dir if copied else None
    return result


def _is_augmented_group(group: pd.DataFrame, video_id: str) -> bool:
    """Return True when a sequence is synthetic augmentation output."""
    vid = str(video_id).lower()
    if "_augmentation" in vid or "_aug_" in vid or "_augmented" in vid or "_generate_" in vid:
        return True
    return gm.sample_is_augmented(group, video_id)


def _resample_feature_sequence(seq: np.ndarray, new_len: int, schema: str | fs.FeatureSchema) -> np.ndarray:
    spec = fs.get_schema(schema)
    seq = fs.ensure_feature_dim(seq, spec)
    new_len = max(1, int(new_len))
    if len(seq) == new_len:
        return seq.astype(np.float32, copy=True)
    if len(seq) <= 1:
        return np.repeat(seq, new_len, axis=0).astype(np.float32, copy=False)

    old_x = np.linspace(0.0, 1.0, num=len(seq), dtype=np.float32)
    new_x = np.linspace(0.0, 1.0, num=new_len, dtype=np.float32)
    out = np.empty((new_len, spec.feature_dim), dtype=np.float32)
    if spec.name in {"smart180", "smart180_face1584"}:
        for col in range(sc.SLICE_META.start):
            out[:, col] = np.interp(new_x, old_x, seq[:, col]).astype(np.float32)
        if spec.feature_dim > sc.SLICE_META.stop:
            for col in range(sc.SLICE_META.stop, spec.feature_dim):
                out[:, col] = np.interp(new_x, old_x, seq[:, col]).astype(np.float32)
        nearest = np.clip(np.rint(new_x * (len(seq) - 1)).astype(int), 0, len(seq) - 1)
        out[:, sc.SLICE_META] = seq[nearest, sc.SLICE_META]
    elif spec.name == "smart268":
        for col in range(fs.SMART268_SLICE_META.start):
            out[:, col] = np.interp(new_x, old_x, seq[:, col]).astype(np.float32)
        nearest = np.clip(np.rint(new_x * (len(seq) - 1)).astype(int), 0, len(seq) - 1)
        out[:, fs.SMART268_SLICE_META] = seq[nearest, fs.SMART268_SLICE_META]
    else:
        for col in range(spec.feature_dim):
            out[:, col] = np.interp(new_x, old_x, seq[:, col]).astype(np.float32)
    return out


def _augment_sequence(
    sequence: np.ndarray,
    rng: np.random.Generator,
    schema: str = fs.DEFAULT_SCHEMA,
    intensity: float = 1.0,
) -> np.ndarray:
    """Lightweight sequence-level augmentation for already-extracted features.

    This intentionally avoids mirror/left-right swap because many BISINDO signs are
    handedness-sensitive. The safe transforms are temporal speed changes, tiny
    coordinate noise, tiny scale jitter, angle jitter for smart180, and occasional
    frame thinning while preserving chronological order.
    """

    spec = fs.get_schema(schema)
    intensity = max(0.0, float(intensity))
    seq = fs.ensure_feature_dim(sequence, spec).copy().astype(np.float32, copy=False)
    if len(seq) <= 0:
        return seq

    # 1) Temporal stretch/compress: same gesture, slightly faster/slower performer.
    if len(seq) >= 4 and rng.random() < 0.70:
        scale = float(rng.uniform(0.88, 1.16))
        scale = 1.0 + (scale - 1.0) * min(1.5, intensity)
        new_len = max(3, int(round(len(seq) * scale)))
        seq = _resample_feature_sequence(seq, new_len, spec)

    # 2) Small coordinate/landmark jitter. Keep smart180 metadata untouched.
    if spec.name in {"smart180", "smart180_face1584"}:
        coord_stop = sc.SLICE_META.start
        if rng.random() < 0.90:
            sigma = 0.0045 * intensity
            seq[:, :coord_stop] += rng.normal(0.0, sigma, size=(len(seq), coord_stop)).astype(np.float32)
            if spec.feature_dim > sc.SLICE_META.stop:
                seq[:, sc.SLICE_META.stop:] += rng.normal(0.0, sigma * 0.55, size=(len(seq), spec.feature_dim - sc.SLICE_META.stop)).astype(np.float32)
        if rng.random() < 0.65:
            angle_sigma = 0.016 * intensity
            seq[:, sc.SLICE_LEFT_ANGLES] += rng.normal(0.0, angle_sigma, size=(len(seq), sc.SLICE_LEFT_ANGLES.stop - sc.SLICE_LEFT_ANGLES.start)).astype(np.float32)
            seq[:, sc.SLICE_RIGHT_ANGLES] += rng.normal(0.0, angle_sigma, size=(len(seq), sc.SLICE_RIGHT_ANGLES.stop - sc.SLICE_RIGHT_ANGLES.start)).astype(np.float32)
        if rng.random() < 0.45:
            scale = float(rng.uniform(0.97, 1.03))
            seq[:, :coord_stop] *= scale
            if spec.feature_dim > sc.SLICE_META.stop:
                seq[:, sc.SLICE_META.stop:] *= scale
        # Clamp presence/score metadata back to sane range.
        seq[:, sc.SLICE_META] = np.clip(seq[:, sc.SLICE_META], 0.0, np.inf)
        for idx in (
            sc.IDX_META_LEFT_PRESENT,
            sc.IDX_META_RIGHT_PRESENT,
            sc.IDX_META_LEFT_DETECTED,
            sc.IDX_META_RIGHT_DETECTED,
            sc.IDX_META_LEFT_HELD,
            sc.IDX_META_RIGHT_HELD,
            sc.IDX_META_SHOULDER_OK,
        ):
            seq[:, sc.SLICE_META.start + idx] = np.clip(seq[:, sc.SLICE_META.start + idx], 0.0, 1.0)
    elif spec.name == "smart268":
        coord_stop = fs.SMART268_SLICE_META.start
        if rng.random() < 0.90:
            sigma = 0.0040 * intensity
            seq[:, :coord_stop] += rng.normal(0.0, sigma, size=(len(seq), coord_stop)).astype(np.float32)
        if rng.random() < 0.45:
            scale = float(rng.uniform(0.97, 1.03))
            seq[:, :coord_stop] *= scale
        seq[:, fs.SMART268_SLICE_META] = np.clip(seq[:, fs.SMART268_SLICE_META], 0.0, np.inf)
        for idx in (
            fs.IDX_SMART268_META_LEFT_PRESENT,
            fs.IDX_SMART268_META_RIGHT_PRESENT,
            fs.IDX_SMART268_META_LEFT_DETECTED,
            fs.IDX_SMART268_META_RIGHT_DETECTED,
            fs.IDX_SMART268_META_LEFT_HELD,
            fs.IDX_SMART268_META_RIGHT_HELD,
            fs.IDX_SMART268_META_SHOULDER_OK,
        ):
            seq[:, fs.SMART268_SLICE_META.start + idx] = np.clip(seq[:, fs.SMART268_SLICE_META.start + idx], 0.0, 1.0)
    else:
        if rng.random() < 0.90:
            sigma = 0.0025 * intensity
            seq += rng.normal(0.0, sigma, size=seq.shape).astype(np.float32)
        if rng.random() < 0.40:
            scale = float(rng.uniform(0.985, 1.015))
            seq *= scale
        seq = np.nan_to_num(seq, nan=0.0, posinf=0.0, neginf=0.0)

    # 3) Drop a few frames then keep order. GRU trainer will resample to target_frames.
    if len(seq) >= 8 and rng.random() < 0.25:
        drop_prob = min(0.08, 0.04 * intensity)
        keep = rng.random(len(seq)) > drop_prob
        if int(keep.sum()) >= 4:
            seq = seq[keep]

    return fs.ensure_feature_dim(seq, spec)


def _rows_from_augmented_sequence(
    *,
    label: str,
    split: str,
    video_id: str,
    base_video_id: str,
    aug_seq: np.ndarray,
    schema: str | fs.FeatureSchema,
    seed: int | None,
    aug_index: int,
) -> list[dict]:
    spec = fs.get_schema(schema)
    rows: list[dict] = []
    for frame_num, vec in enumerate(fs.ensure_feature_dim(aug_seq, spec)):
        presence = fs.presence_from_vector(spec, vec)
        if spec.name in {"smart180", "smart180_face1584"}:
            meta = vec[sc.SLICE_META]
            left_held = float(meta[sc.IDX_META_LEFT_HELD])
            right_held = float(meta[sc.IDX_META_RIGHT_HELD])
            left_score = float(meta[sc.IDX_META_LEFT_SCORE])
            right_score = float(meta[sc.IDX_META_RIGHT_SCORE])
        elif spec.name == "smart268":
            meta = vec[fs.SMART268_SLICE_META]
            left_held = float(meta[fs.IDX_SMART268_META_LEFT_HELD])
            right_held = float(meta[fs.IDX_SMART268_META_RIGHT_HELD])
            left_score = float(presence["left_present"])
            right_score = float(presence["right_present"])
        else:
            left_held = 0.0
            right_held = 0.0
            left_score = float(presence["left_present"])
            right_score = float(presence["right_present"])
        rows.append(
            {
                "video_id": video_id,
                "label": label,
                "frame_num": int(frame_num),
                "split": split,
                "schema": spec.name,
                "feature_version": spec.feature_schema,
                "feature_mode": spec.feature_mode,
                "feature_dim": spec.feature_dim,
                "target_fps": spec.target_fps,
                "extract_profile": "augment",
                "source_media_path": "",
                "source_frame_num": int(frame_num),
                "chosen_source_frame": int(frame_num),
                "time_sec": float(frame_num / spec.target_fps),
                "smart_mode": "augment",
                "enhance_mode": "augment",
                "left_present": float(presence["left_present"]),
                "right_present": float(presence["right_present"]),
                "left_detected": float(presence["left_present"]),
                "right_detected": float(presence["right_present"]),
                "left_held": left_held,
                "right_held": right_held,
                "left_score": left_score,
                "right_score": right_score,
                "is_augmented": True,
                "augmented_from": str(base_video_id),
                "augmentation_seed": "" if seed is None else int(seed),
                "augmentation_index": int(aug_index),
                "features": sc.format_feature_value(vec),
            }
        )
    return rows


def augment_dataset(
    dataset_dir: str | Path = DATASET_DIR,
    backup_root: str | Path = BACKUP_ROOT,
    schema: str = fs.DEFAULT_SCHEMA,
    split: str = "train",
    target_per_class: int = 0,
    copies_per_sample: int = 2,
    min_source_samples: int = 5,
    vocab: str | None = None,
    include_idle: bool = False,
    include_augmented_source: bool = False,
    seed: int | None = None,
    intensity: float = 1.0,
    overwrite_existing: bool = False,
) -> AugmentResult:
    split = str(split or "train").lower()
    split_values = _split_values(split, allow_all=True)

    schema_spec = fs.get_schema(schema)
    wanted_vocab = clean_label(vocab) if vocab else ""
    rng = np.random.default_rng(seed)
    backup_dir = Path(backup_root) / f"bisindo_cli_augment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    copied: set[Path] = set()
    result = AugmentResult()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if any(value in {"val", "test"} for value in split_values):
        print("[WARN] Augmentasi val/test sebaiknya hanya untuk eksperimen robustness, bukan final evaluation.", flush=True)

    for parquet_path in fs.dataset_parquet_paths(schema_spec, dataset_dir):
        label = clean_label(parquet_path.stem)
        if wanted_vocab and label != wanted_vocab:
            continue
        if not include_idle and label.lower() == "idle":
            continue

        df = pd.read_parquet(parquet_path)
        if "split" not in df.columns:
            df = df.copy()
            df["split"] = "train"
        if "label" not in df.columns:
            df = df.copy()
            df["label"] = label
        current = fs.filter_feature_rows(df, schema_spec)
        if current.empty:
            continue

        parquet_new_rows: list[dict] = []
        existing_ids = {str(value) for value in current.get("video_id", pd.Series(dtype=str)).dropna().astype(str).unique()}

        for split_name in split_values:
            split_df = current[current["split"].astype(str).str.lower() == split_name]
            if split_df.empty:
                print(f"[AUG-SKIP] {schema_spec.name}/{split_name}/{label}: tidak ada data", flush=True)
                result.skipped += 1
                continue

            all_groups = [(str(video_id), group.sort_values("frame_num")) for video_id, group in split_df.groupby("video_id", sort=False)]
            source_groups = [(video_id, group) for video_id, group in all_groups if include_augmented_source or not _is_augmented_group(group, video_id)]
            source_count = len(source_groups)
            existing_count = len(all_groups)

            if source_count < int(min_source_samples):
                print(
                    f"[AUG-SKIP] {schema_spec.name}/{split_name}/{label}: source={source_count} < min={int(min_source_samples)}",
                    flush=True,
                )
                result.skipped += 1
                continue

            by_target = max(0, int(target_per_class) - existing_count) if int(target_per_class) > 0 else 0
            by_copies = max(0, int(copies_per_sample)) * source_count
            needed = max(by_target, by_copies)
            if needed <= 0:
                print(
                    f"[AUG-SKIP] {schema_spec.name}/{split_name}/{label}: existing={existing_count}, target={int(target_per_class)}, copies={int(copies_per_sample)}",
                    flush=True,
                )
                result.skipped += 1
                continue

            made = 0
            for aug_idx in range(needed):
                base_video_id, group = source_groups[int(rng.integers(0, len(source_groups)))]
                base_seq = fs.ensure_feature_dim([sc.parse_feature_value(value) for value in group["features"].tolist()], schema_spec)
                if len(base_seq) <= 0:
                    continue
                aug_seq = _augment_sequence(base_seq, rng, schema=schema_spec.name, intensity=float(intensity))
                seq_idx = result.generated + made
                new_video_id = f"{split_name}_{label}_augmentation_{stamp}_{seq_idx:05d}"
                dedup = 1
                while new_video_id in existing_ids:
                    new_video_id = f"{split_name}_{label}_augmentation_{stamp}_{seq_idx:05d}_{dedup}"
                    dedup += 1
                existing_ids.add(new_video_id)
                rows = _rows_from_augmented_sequence(
                    label=label,
                    split=split_name,
                    video_id=new_video_id,
                    base_video_id=base_video_id,
                    aug_seq=aug_seq,
                    schema=schema_spec,
                    seed=seed,
                    aug_index=seq_idx,
                )
                parquet_new_rows.extend(rows)
                made += 1
                result.rows += len(rows)

            result.generated += made
            print(
                f"[AUG] {schema_spec.name}/{split_name}/{label}: source={source_count} existing={existing_count} +{made} sequences rows+={sum(1 for row in parquet_new_rows if row['split'] == split_name)}",
                flush=True,
            )

        if parquet_new_rows:
            _backup_existing(parquet_path, backup_dir, copied)
            df_out = pd.concat([df, pd.DataFrame(parquet_new_rows)], ignore_index=True)
            df_out.to_parquet(parquet_path, index=False)
            print(f"[APPEND parquet +{len(parquet_new_rows)} rows] {parquet_path}", flush=True)

    result.backup_dir = backup_dir if copied else None
    return result


def delete_augmented_dataset(
    dataset_dir: str | Path = DATASET_DIR,
    backup_root: str | Path = BACKUP_ROOT,
    schema: str = fs.DEFAULT_SCHEMA,
    split: str = "all",
    vocab: str | None = None,
    include_idle: bool = False,
    dry_run: bool = False,
) -> DeleteAugmentResult:
    """Remove only synthetic augmentation sequences from parquet datasets.

    Augmented samples are detected at sequence/group level using the explicit
    metadata columns written by `augment_dataset` plus the `video_id` marker
    `_augmentation`. The older `_aug_` marker is also supported so previous
    augmentation outputs can be cleaned safely.
    """

    split = str(split or "all").lower()
    split_values = _split_values(split, allow_all=True)

    schema_spec = fs.get_schema(schema)
    wanted_vocab = clean_label(vocab) if vocab else ""
    backup_dir = Path(backup_root) / f"bisindo_cli_delete_augmentation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    copied: set[Path] = set()
    result = DeleteAugmentResult(dry_run=bool(dry_run))

    for parquet_path in fs.dataset_parquet_paths(schema_spec, dataset_dir):
        result.scanned_files += 1
        label = clean_label(parquet_path.stem)
        if wanted_vocab and label != wanted_vocab:
            continue
        if not include_idle and label.lower() == "idle":
            continue

        try:
            df = pd.read_parquet(parquet_path)
        except Exception as exc:
            print(f"[DEL-SKIP] {parquet_path}: gagal baca parquet: {exc}", flush=True)
            result.skipped += 1
            continue

        if df.empty or "video_id" not in df.columns:
            print(f"[DEL-SKIP] {schema_spec.name}/{label}: parquet kosong atau tanpa video_id", flush=True)
            result.skipped += 1
            continue

        work = df.copy()
        if "split" not in work.columns:
            work["split"] = "train"
        if "label" not in work.columns:
            work["label"] = label

        current = fs.filter_feature_rows(work, schema_spec)
        if current.empty:
            print(f"[DEL-SKIP] {schema_spec.name}/{label}: tidak ada row schema aktif", flush=True)
            result.skipped += 1
            continue

        delete_ids: set[str] = set()
        for video_id, group in current.groupby("video_id", sort=False):
            group = group.sort_values("frame_num") if "frame_num" in group.columns else group
            split_name = str(group["split"].iloc[0]).lower() if "split" in group.columns and not group.empty else "train"
            if split_name not in split_values:
                continue
            if _is_augmented_group(group, str(video_id)):
                delete_ids.add(str(video_id))

        if not delete_ids:
            print(f"[DEL-SKIP] {schema_spec.name}/{label}: tidak ada hasil augmentasi untuk split={split}", flush=True)
            result.skipped += 1
            continue

        delete_mask = work["video_id"].astype(str).isin(delete_ids)
        deleted_rows = int(delete_mask.sum())
        result.deleted_sequences += len(delete_ids)
        result.deleted_rows += deleted_rows

        print(
            f"[DEL{'-DRY' if dry_run else ''}] {schema_spec.name}/{label}: "
            f"hapus {len(delete_ids)} sequence augmentasi, rows={deleted_rows}, split={split}",
            flush=True,
        )

        if dry_run:
            continue

        _backup_existing(parquet_path, backup_dir, copied)
        out = work.loc[~delete_mask].copy()
        out.to_parquet(parquet_path, index=False)
        result.files_updated += 1
        print(f"[WRITE] {parquet_path} rows-={deleted_rows}", flush=True)

    result.backup_dir = backup_dir if copied else None
    return result


def count_gifs(root: str | Path = GIF_DIR) -> int:
    path = Path(root)
    return len(list(path.rglob("*.gif"))) if path.exists() else 0


def dataset_quick_summary(dataset_dir: str | Path = DATASET_DIR, schema: str = fs.DEFAULT_SCHEMA) -> dict[str, object]:
    schema_spec = fs.get_schema(schema)
    dataset_root = Path(dataset_dir)
    counts: dict[str, dict[str, int]] = {}
    labels: set[str] = set()
    total_samples = 0
    if not dataset_root.exists():
        return {"total_samples": 0, "num_classifier_classes": 0, "labels": [], "counts": counts}

    wanted = ["label", "video_id", "split", "feature_version", "feature_dim"]
    for parquet_path in fs.dataset_parquet_paths(schema_spec, dataset_root):
        try:
            df = pd.read_parquet(parquet_path, columns=wanted)
        except Exception:
            try:
                df = pd.read_parquet(parquet_path)
            except Exception:
                continue
        if "label" not in df.columns:
            df = df.copy()
            df["label"] = parquet_path.stem
        if "video_id" not in df.columns:
            df = df.copy()
            df["video_id"] = parquet_path.stem
        if "split" not in df.columns:
            df = df.copy()
            df["split"] = "train"
        current = fs.filter_feature_rows(df, schema_spec)
        if current.empty:
            continue
        for (label, split), group in current.groupby(["label", "split"], sort=False):
            label = str(label)
            split = str(split).lower()
            labels.add(label)
            count = int(group["video_id"].astype(str).nunique())
            counts.setdefault(label, {})
            counts[label][split] = counts[label].get(split, 0) + count
            total_samples += count
    return {
        "total_samples": total_samples,
        "num_classifier_classes": len([label for label in labels if label.lower() not in gm.EXCLUDED_LABELS]),
        "labels": sorted(labels),
        "counts": counts,
    }


def cmd_status(args: argparse.Namespace) -> int:
    print("== BISINDO Status ==")
    print(jr.diagnostics_text(jr.diagnostics()))
    for schema_name in fs.expand_schema_names(args.schema):
        spec = fs.get_schema(schema_name)
        try:
            summary = dataset_quick_summary(args.dataset_dir, schema=schema_name)
            print(f"dataset[{schema_name}:{spec.feature_dim}D]: {summary['total_samples']} samples, {summary['num_classifier_classes']} classifier classes")
        except Exception as exc:
            print(f"dataset[{schema_name}]: unavailable ({exc})")
        for variant, value in gm.list_model_status(args.model_dir, schema=schema_name).items():
            print(f"gru_{schema_name}_{variant}: {value}")
    print(f"gifs: {count_gifs(args.gif_dir)}")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    exit_code = 0
    for schema_name in fs.expand_schema_names(args.schema if args.schema else fs.DEFAULT_SCHEMA):
        argv = [
            "train",
            "--variant",
            args.variant,
            "--schema",
            schema_name,
            "--dataset-dir",
            args.dataset_dir,
            "--model-dir",
            args.model_dir,
            "--device",
            args.device,
            *([] if args.epochs is None else ["--epochs", str(args.epochs)]),
            *([] if args.batch_size is None else ["--batch-size", str(args.batch_size)]),
            *([] if args.lr is None else ["--lr", str(args.lr)]),
            *([] if args.patience is None else ["--patience", str(args.patience)]),
            *([] if args.limit_per_class is None else ["--limit-per-class", str(args.limit_per_class)]),
            "--train-data",
            getattr(args, "train_data", "original"),
        ]
        if args.overwrite_existing:
            argv += ["--overwrite-existing", "--backup-root", args.backup_root]
        code = gm.main(argv)
        exit_code = max(exit_code, int(code or 0))
    return exit_code


def cmd_eval(args: argparse.Namespace) -> int:
    return gm.main(
        [
            "eval",
            "--variant",
            args.variant,
            "--schema",
            args.schema,
            "--dataset-dir",
            args.dataset_dir,
            "--model-dir",
            args.model_dir,
            "--device",
            args.device,
            "--split",
            args.split,
            "--suite",
            args.suite,
        ]
    )


def cmd_benchmark(args: argparse.Namespace) -> int:
    argv = [
        "benchmark",
        "--variant",
        args.variant,
        "--schema",
        args.schema,
        "--dataset-dir",
        args.dataset_dir,
        "--model-dir",
        args.model_dir,
        "--device",
        args.device,
        "--warmup",
        str(args.warmup),
        "--runs",
        str(args.runs),
        "--threads",
        str(args.threads),
    ]
    if args.no_jit:
        argv.append("--no-jit")
    return gm.main(argv)


def cmd_import(args: argparse.Namespace) -> int:
    items = scan_import_items(args.path, default_split=args.split, label=args.label)
    if not items:
        print("Tidak ada video yang bisa diimport.")
        return 1
    exit_code = 0
    for schema_name in fs.expand_schema_names(args.schema if args.schema else fs.DEFAULT_SCHEMA):
        result = append_import_items(
            items,
            dataset_dir=args.dataset_dir,
            backup_root=args.backup_root,
            schema=schema_name,
            save_gif=args.save_gif,
            quiet=args.quiet,
            overwrite_existing=bool(args.overwrite_existing),
        )
        print(
            f"import[{schema_name}]: scanned={result.scanned} imported={result.imported} skipped={result.skipped} "
            f"failed={result.failed} rows={result.rows}"
        )
        if result.backup_dir:
            print(f"backup[{schema_name}]: {result.backup_dir}")
        if result.gif_paths:
            print(f"gifs[{schema_name}]: {len(result.gif_paths)} files")
        if result.failed:
            exit_code = 1
    return exit_code


def cmd_extract_full(args: argparse.Namespace) -> int:
    import full_mediapipe_converter as fmc

    result = fmc.extract_full_mediapipe_dataset(
        source=args.source,
        dataset_dir=args.dataset_dir,
        model_dir=args.model_dir,
        backup_root=args.backup_root,
        schema=args.schema if args.schema else "full",
        clean=args.clean,
        limit_per_class=args.limit_per_class,
        vocab=args.vocab,
    )
    print(
        f"extract-full: scanned={result.scanned} converted={result.converted} failed={result.failed} "
        f"rows={result.rows} schemas={','.join(result.schemas)}",
        flush=True,
    )
    if result.backup_dir:
        print(f"backup: {result.backup_dir}", flush=True)
    if result.manifest_path:
        print(f"manifest: {result.manifest_path}", flush=True)
    return 0 if result.failed == 0 and result.converted > 0 else 1


def cmd_train_suite(args: argparse.Namespace) -> int:
    import gru_experts as ge

    variants = gm.expand_variant_request(args.variant, getattr(args, "train_data", "original"))
    exit_code = 0
    for schema_name in fs.expand_schema_names(args.schema):
        for variant in variants:
            try:
                result = ge.train_suite(
                    variant=variant,
                    schema=schema_name,
                    suites=args.suite,
                    dataset_dir=args.dataset_dir,
                    model_dir=args.model_dir,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    patience=args.patience,
                    device=args.device,
                    threshold=args.threshold_distance,
                    limit_per_class=args.limit_per_class,
                    overwrite_existing=bool(args.overwrite_existing),
                    backup_root=args.backup_root,
                    train_data=gm.variant_train_data_mode(variant),
                )
                print(f"train-suite[{schema_name}/{variant}]: {json.dumps(result, default=str)[:1200]}", flush=True)
            except Exception as exc:
                print(f"train-suite[{schema_name}/{variant}]: error ({exc})", flush=True)
                exit_code = 1
    return exit_code


def make_gif_from_video(
    video: str | Path,
    label: str,
    video_id: str | None = None,
    out_dir: str | Path | None = None,
    schema: str = fs.DEFAULT_SCHEMA,
) -> list[Path]:
    from smart_extract.extract_video_smart_v8 import save_gif

    schema_spec = fs.get_schema(schema)
    label = clean_label(label)
    if not label:
        raise ValueError("--label wajib untuk gif make")
    video_path = Path(video).expanduser()
    item = ImportItem(video_path=video_path, label=label, split="preview", video_id=video_id or video_path.stem)
    args = _make_extract_args_for_schema(schema_spec.name, quiet=False, save_gif=True)
    result = _extract_video_for_schema(video_path, schema=schema_spec.name, args=args, include_frames=True)
    if result is None:
        raise RuntimeError(f"Gagal extract GIF: {video_path}")
    if out_dir is None:
        return _save_import_gifs(item, result, schema=schema_spec.name, gif_width=int(args.gif_width))

    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    overlay = root / f"{video_path.stem}_overlay.gif"
    skeleton = root / f"{video_path.stem}_skeleton.gif"
    save_gif(result.get("overlay_frames", []), overlay, float(result.get("target_fps", sc.TARGET_FPS)), int(args.gif_width))
    save_gif(result.get("skeleton_frames", []), skeleton, float(result.get("target_fps", sc.TARGET_FPS)), int(args.gif_width))
    return [overlay, skeleton]



# Lightweight GIF-from-feature renderer.  This deliberately does not touch the
# original video/camera pipeline, so it stays fast even on Jetson.  The output is
# a skeleton visualization reconstructed from rows already stored in parquet.
SMART_BTJ_CONNECTIONS = (
    (0, 1),  # wrist -> palm center
    (1, 2),  # palm -> thumb tip
    (1, 3), (3, 4),
    (1, 5), (5, 6),
    (1, 7), (7, 8),
    (1, 9), (9, 10),
)

HAND21_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
)


def _safe_name(value: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in str(value))


def _feature_gif_path(
    *,
    schema: str,
    vocab: str,
    split: str,
    video_id: str,
    gif_dir: str | Path = GIF_DIR,
) -> Path:
    spec = fs.get_schema(schema)
    return Path(gif_dir) / "samples" / spec.name / clean_label(vocab) / f"{_safe_name(split)}_{_safe_name(video_id)}_feature.gif"


def _draw_points(
    img,
    pts: np.ndarray,
    connections: Sequence[tuple[int, int]],
    color: tuple[int, int, int],
    radius: int = 4,
    thickness: int = 2,
) -> None:
    import cv2

    if pts is None:
        return
    arr = np.asarray(pts, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] < 2:
        return
    valid = np.isfinite(arr[:, :2]).all(axis=1)
    if not np.any(valid):
        return
    h, w = img.shape[:2]
    xy = arr[:, :2].copy()
    xy[:, 0] = np.clip(xy[:, 0], 0, w - 1)
    xy[:, 1] = np.clip(xy[:, 1], 0, h - 1)
    pix = np.round(xy).astype(int)
    for a, b in connections:
        if a < len(pix) and b < len(pix) and valid[a] and valid[b]:
            cv2.line(img, tuple(pix[a]), tuple(pix[b]), color, thickness, cv2.LINE_AA)
    for idx, pt in enumerate(pix):
        if valid[idx]:
            cv2.circle(img, tuple(pt), radius, color, -1, cv2.LINE_AA)


def _norm_points_to_pixels(points: np.ndarray, width: int, height: int) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] < 2:
        return np.zeros((0, 2), dtype=np.float32)
    out = pts[:, :2].copy()
    out[:, 0] *= float(width)
    out[:, 1] *= float(height)
    return out


def _smart_points_from_vector(vec: np.ndarray, width: int, height: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = fs.ensure_feature_dim(vec, "smart180")[0]
    anchor_px = np.array([width * 0.50, height * 0.36], dtype=np.float32)
    scale_px = min(width, height) * 0.34

    shoulders_rel = arr[sc.SLICE_SHOULDERS].reshape(2, 3)
    if float(np.linalg.norm(shoulders_rel[:, :2])) < 1e-6:
        shoulders_rel = np.array([[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=np.float32)
    shoulders = anchor_px + shoulders_rel[:, :2] * scale_px

    left_global = arr[sc.SLICE_LEFT_GLOBAL].reshape(11, 3)
    right_global = arr[sc.SLICE_RIGHT_GLOBAL].reshape(11, 3)
    left_local = arr[sc.SLICE_LEFT_LOCAL].reshape(11, 3)
    right_local = arr[sc.SLICE_RIGHT_LOCAL].reshape(11, 3)
    meta = arr[sc.SLICE_META]

    def hand_pixels(global_pts: np.ndarray, local_pts: np.ndarray, is_left: bool) -> np.ndarray:
        if float(np.linalg.norm(global_pts[:, :2])) > 1e-6:
            return anchor_px + global_pts[:, :2] * scale_px
        # Fallback when a schema row only has local points.  Put the wrist near
        # the shoulder side, then draw the local hand shape from there.
        xoff = -0.45 if is_left else 0.45
        base = anchor_px + np.array([xoff * scale_px, 0.80 * scale_px], dtype=np.float32)
        return base + local_pts[:, :2] * (scale_px * 0.38)

    left_present = float(meta[sc.IDX_META_LEFT_PRESENT]) >= 0.5 if len(meta) > sc.IDX_META_LEFT_PRESENT else True
    right_present = float(meta[sc.IDX_META_RIGHT_PRESENT]) >= 0.5 if len(meta) > sc.IDX_META_RIGHT_PRESENT else True
    left = hand_pixels(left_global, left_local, True) if left_present or np.linalg.norm(left_global) > 1e-6 else np.zeros((0, 2), dtype=np.float32)
    right = hand_pixels(right_global, right_local, False) if right_present or np.linalg.norm(right_global) > 1e-6 else np.zeros((0, 2), dtype=np.float32)
    return shoulders.astype(np.float32), left.astype(np.float32), right.astype(np.float32)


def _smart268_points_from_vector(vec: np.ndarray, width: int, height: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = fs.ensure_feature_dim(vec, "smart268")[0]
    anchor_px = np.array([width * 0.50, height * 0.36], dtype=np.float32)
    scale_px = min(width, height) * 0.34

    shoulders_rel = arr[fs.SMART268_SLICE_SHOULDERS].reshape(2, 3)
    if float(np.linalg.norm(shoulders_rel[:, :2])) < 1e-6:
        shoulders_rel = np.array([[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=np.float32)
    shoulders = anchor_px + shoulders_rel[:, :2] * scale_px

    left_global = arr[fs.SMART268_SLICE_LEFT_GLOBAL].reshape(21, 3)
    right_global = arr[fs.SMART268_SLICE_RIGHT_GLOBAL].reshape(21, 3)
    left_local = arr[fs.SMART268_SLICE_LEFT_LOCAL].reshape(21, 3)
    right_local = arr[fs.SMART268_SLICE_RIGHT_LOCAL].reshape(21, 3)
    meta = arr[fs.SMART268_SLICE_META]

    def hand_pixels(global_pts: np.ndarray, local_pts: np.ndarray, is_left: bool) -> np.ndarray:
        if float(np.linalg.norm(global_pts[:, :2])) > 1e-6:
            return anchor_px + global_pts[:, :2] * scale_px
        xoff = -0.45 if is_left else 0.45
        base = anchor_px + np.array([xoff * scale_px, 0.80 * scale_px], dtype=np.float32)
        return base + local_pts[:, :2] * (scale_px * 0.38)

    left_present = float(meta[fs.IDX_SMART268_META_LEFT_PRESENT]) >= 0.5
    right_present = float(meta[fs.IDX_SMART268_META_RIGHT_PRESENT]) >= 0.5
    left = hand_pixels(left_global, left_local, True) if left_present or np.linalg.norm(left_global) > 1e-6 else np.zeros((0, 2), dtype=np.float32)
    right = hand_pixels(right_global, right_local, False) if right_present or np.linalg.norm(right_global) > 1e-6 else np.zeros((0, 2), dtype=np.float32)
    return shoulders.astype(np.float32), left.astype(np.float32), right.astype(np.float32)


def _holistic_parts_from_vector(schema: str, vec: np.ndarray, width: int, height: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    spec = fs.get_schema(schema)
    arr = fs.ensure_feature_dim(vec, spec)[0]
    if spec.name == "khukuh1629":
        right = arr[0:63].reshape(21, 3)
        left = arr[63:126].reshape(21, 3)
        pose = arr[126:225].reshape(33, 3)
        shoulders = pose[[11, 12], :2] if pose.shape[0] >= 13 else np.zeros((0, 2), dtype=np.float32)
    elif spec.name == "adi1662":
        pose = arr[0:132].reshape(33, 4)
        left = arr[1536:1599].reshape(21, 3)
        right = arr[1599:1662].reshape(21, 3)
        shoulders = pose[[11, 12], :2] if pose.shape[0] >= 13 else np.zeros((0, 2), dtype=np.float32)
    elif spec.name == "smart180_face1584":
        # First 180 dims are exactly the Smart180 compact vector.
        return _smart_points_from_vector(arr[:180], width, height)
    else:
        raise ValueError(f"Renderer holistic tidak mendukung schema {schema}")

    def present(points: np.ndarray) -> bool:
        return bool(np.isfinite(points[:, :2]).all() and np.linalg.norm(points[:, :2]) > 1e-6)

    shoulders_px = _norm_points_to_pixels(shoulders, width, height) if present(shoulders) else np.zeros((0, 2), dtype=np.float32)
    left_px = _norm_points_to_pixels(left, width, height) if present(left) else np.zeros((0, 2), dtype=np.float32)
    right_px = _norm_points_to_pixels(right, width, height) if present(right) else np.zeros((0, 2), dtype=np.float32)
    return shoulders_px.astype(np.float32), left_px.astype(np.float32), right_px.astype(np.float32)


def _face_points_from_vector(schema: str, vec: np.ndarray, width: int, height: int) -> np.ndarray:
    """Return face landmark xy pixels for schemas that contain face xyz."""
    spec = fs.get_schema(schema)
    arr = fs.ensure_feature_dim(vec, spec)[0]
    if spec.name == "khukuh1629":
        face = arr[225:1629].reshape(468, 3)
    elif spec.name == "adi1662":
        face = arr[132:1536].reshape(468, 3)
    elif spec.name == "smart180_face1584":
        face = arr[180:1584].reshape(468, 3)
    else:
        return np.zeros((0, 2), dtype=np.float32)
    if not np.isfinite(face[:, :2]).all() or float(np.linalg.norm(face[:, :2])) <= 1e-6:
        return np.zeros((0, 2), dtype=np.float32)
    return _norm_points_to_pixels(face, width, height).astype(np.float32)


FACE_SPARSE_INDICES = np.array([
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
    33, 133, 362, 263, 61, 291, 1, 4, 199, 168,
], dtype=np.int32)


def _render_feature_gif_frame(
    *,
    schema: str,
    vector: np.ndarray,
    label: str,
    split: str,
    video_id: str,
    frame_idx: int,
    total_frames: int,
    width: int,
    height: int,
    draw_face: str = "auto",
) -> np.ndarray:
    import cv2

    spec = fs.get_schema(schema)
    img = np.zeros((int(height), int(width), 3), dtype=np.uint8)

    if spec.name in {"smart180", "smart180_face1584"}:
        shoulders, left, right = _smart_points_from_vector(vector, width, height)
        hand_connections = SMART_BTJ_CONNECTIONS
    elif spec.name == "smart268":
        shoulders, left, right = _smart268_points_from_vector(vector, width, height)
        hand_connections = HAND21_CONNECTIONS
    else:
        shoulders, left, right = _holistic_parts_from_vector(spec.name, vector, width, height)
        hand_connections = HAND21_CONNECTIONS

    _draw_points(img, shoulders, ((0, 1),), (180, 180, 180), radius=5, thickness=3)
    _draw_points(img, left, hand_connections, (60, 220, 60), radius=4, thickness=2)
    _draw_points(img, right, hand_connections, (0, 190, 255), radius=4, thickness=2)

    face_mode = str(draw_face or "auto").lower()
    if face_mode == "auto":
        face_mode = "sparse" if bool(getattr(spec, "uses_face", False)) else "off"
    if face_mode not in {"off", "none", "0", "false"}:
        face_pts = _face_points_from_vector(spec.name, vector, width, height)
        if face_pts.shape[0] > 0:
            if face_mode == "mesh":
                shown = face_pts
                radius = 1
            else:
                idx = FACE_SPARSE_INDICES[FACE_SPARSE_INDICES < len(face_pts)]
                shown = face_pts[idx]
                radius = 2
            _draw_points(img, shown, tuple(), (160, 160, 255), radius=radius, thickness=1)

    cv2.putText(img, f"{label} | {split} | {spec.name}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (245, 245, 245), 2, cv2.LINE_AA)
    cv2.putText(img, f"{frame_idx + 1}/{total_frames} | {video_id[:42]}", (12, height - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (210, 210, 210), 1, cv2.LINE_AA)
    return img


def _sequence_to_feature_gif(
    *,
    schema: str,
    sequence: np.ndarray,
    label: str,
    split: str,
    video_id: str,
    out_path: str | Path,
    fps: float,
    width: int,
    height: int,
    max_frames: int = 0,
    draw_face: str = "auto",
) -> Path:
    from smart_extract.extract_video_smart_v8 import save_gif

    spec = fs.get_schema(schema)
    seq = fs.ensure_feature_dim(sequence, spec)
    if max_frames and len(seq) > int(max_frames):
        # Uniformly sample long sequences so the GIF stays small and fast.
        idx = np.linspace(0, len(seq) - 1, int(max_frames)).round().astype(int)
        seq = seq[idx]
    frames = [
        _render_feature_gif_frame(
            schema=spec.name,
            vector=vec,
            label=label,
            split=split,
            video_id=video_id,
            frame_idx=i,
            total_frames=len(seq),
            width=int(width),
            height=int(height),
            draw_face=draw_face,
        )
        for i, vec in enumerate(seq)
    ]
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_gif(frames, out, float(fps), int(width))
    return out


def _iter_dataset_feature_sequences(
    *,
    schema: str,
    dataset_dir: str | Path = DATASET_DIR,
    vocab: str | None = None,
    split: str = "all",
    limit_per_split: int = 0,
) -> Iterable[tuple[str, str, str, np.ndarray]]:
    spec = fs.get_schema(schema)
    label_filter = clean_label(vocab) if vocab else None
    split_filter = str(split or "all").lower()
    counts: dict[tuple[str, str], int] = {}
    for parquet_path in fs.dataset_parquet_paths(spec, dataset_dir):
        try:
            df = pd.read_parquet(parquet_path)
        except Exception as exc:
            print(f"[GIF][WARN] skip parquet gagal dibaca: {parquet_path} ({exc})", flush=True)
            continue
        df = fs.filter_feature_rows(df, spec)
        if df.empty:
            continue
        if label_filter:
            df = df[df["label"].astype(str).map(clean_label) == label_filter]
        if split_filter != "all" and "split" in df.columns:
            df = df[df["split"].astype(str).str.lower() == split_filter]
        if df.empty:
            continue
        group_cols = ["label", "split", "video_id"]
        missing = [col for col in group_cols if col not in df.columns]
        if missing:
            print(f"[GIF][WARN] skip {parquet_path}: kolom hilang {missing}", flush=True)
            continue
        for (label, item_split, video_id), group in df.groupby(group_cols, sort=True):
            label = clean_label(label)
            item_split = str(item_split).lower()
            key = (label, item_split)
            if limit_per_split > 0 and counts.get(key, 0) >= int(limit_per_split):
                continue
            group = group.sort_values("frame_num") if "frame_num" in group.columns else group
            try:
                seq = np.stack([fs.parse_feature_value(value) for value in group["features"].tolist()]).astype(np.float32)
                seq = fs.ensure_feature_dim(seq, spec)
            except Exception as exc:
                print(f"[GIF][WARN] skip {label}/{item_split}/{video_id}: parse feature gagal ({exc})", flush=True)
                continue
            counts[key] = counts.get(key, 0) + 1
            yield label, item_split, str(video_id), seq


def dataset_vocab_summary(
    *,
    schema: str | Iterable[str] = "all",
    dataset_dir: str | Path = DATASET_DIR,
) -> list[dict[str, object]]:
    schema_names = fs.expand_schema_names(schema)
    # schema=all usually stores the same sample in three schema folders.  Use
    # max-per-split instead of sum to avoid showing 3x inflated counts.
    merged: dict[str, dict[str, object]] = {}
    for schema_name in schema_names:
        spec = fs.get_schema(schema_name)
        local: dict[str, dict[str, int]] = {}
        for parquet_path in fs.dataset_parquet_paths(spec, dataset_dir):
            try:
                df = pd.read_parquet(parquet_path)
            except Exception:
                continue
            df = fs.filter_feature_rows(df, spec)
            if df.empty or not {"label", "split", "video_id"}.issubset(df.columns):
                continue
            unique_samples = df[["label", "split", "video_id"]].drop_duplicates()
            for (label, split_name), group in unique_samples.groupby(["label", "split"], sort=True):
                label = clean_label(label)
                split_name = str(split_name).lower()
                if split_name not in SPLITS:
                    continue
                local.setdefault(label, {"train": 0, "val": 0, "test": 0})[split_name] += int(len(group))
        for label, counts in local.items():
            row = merged.setdefault(label, {"label": label, "train": 0, "val": 0, "test": 0, "schemas": set()})
            for split_name in ("train", "val", "test"):
                row[split_name] = max(int(row[split_name]), int(counts.get(split_name, 0)))
            row["schemas"].add(schema_name)

    out = []
    for label, row in sorted(merged.items()):
        train = int(row["train"])
        val = int(row["val"])
        test = int(row["test"])
        out.append({
            "label": label,
            "train": train,
            "val": val,
            "test": test,
            "total": train + val + test,
            "schemas": ",".join(sorted(row["schemas"])),
        })
    return out


def cmd_gif_vocab(args: argparse.Namespace) -> int:
    rows = dataset_vocab_summary(schema=args.schema, dataset_dir=args.dataset_dir)
    if args.plain:
        print("label\ttrain\tval\ttest\ttotal\tschemas")
        for row in rows:
            print(f"{row['label']}\t{row['train']}\t{row['val']}\t{row['test']}\t{row['total']}\t{row['schemas']}")
    else:
        print(f"VOCAB DATASET GIF SOURCE | schema={args.schema}")
        print(f"{'vocab':24s} {'train':>6s} {'val':>6s} {'test':>6s} {'total':>6s}  schemas")
        print("-" * 78)
        for row in rows:
            print(f"{row['label'][:24]:24s} {int(row['train']):6d} {int(row['val']):6d} {int(row['test']):6d} {int(row['total']):6d}  {row['schemas']}")
        print(f"total vocab: {len(rows)}")
    return 0


def cmd_gif_dataset(args: argparse.Namespace) -> int:
    vocab = clean_label(args.vocab or "")
    if not vocab and not args.all_vocab:
        raise ValueError("Pilih --vocab NAMA atau pakai --all-vocab.")
    if args.all_vocab:
        selected_vocabs = [str(row["label"]) for row in dataset_vocab_summary(schema=args.schema, dataset_dir=args.dataset_dir)]
        if not selected_vocabs:
            print("[GIF] Tidak ada vocab dataset.")
            return 1
    else:
        selected_vocabs = [vocab]

    generated = 0
    skipped = 0
    failed = 0
    schema_names = fs.expand_schema_names(args.schema)
    split = str(args.split or "all").lower()
    t0 = time.perf_counter()
    for schema_name in schema_names:
        spec = fs.get_schema(schema_name)
        for vocab_name in selected_vocabs:
            print(f"[GIF] scan schema={spec.name} vocab={vocab_name} split={split}", flush=True)
            seq_iter = _iter_dataset_feature_sequences(
                schema=spec.name,
                dataset_dir=args.dataset_dir,
                vocab=vocab_name,
                split=split,
                limit_per_split=int(args.limit),
            )
            any_sample = False
            for idx, (label, item_split, video_id, sequence) in enumerate(seq_iter, start=1):
                any_sample = True
                out_path = _feature_gif_path(
                    schema=spec.name,
                    vocab=label,
                    split=item_split,
                    video_id=video_id,
                    gif_dir=args.gif_dir,
                )
                if out_path.exists() and not args.force:
                    skipped += 1
                    print(f"[GIF] skip existing schema={spec.name} {item_split}/{label}/{video_id} -> {out_path}", flush=True)
                    continue
                try:
                    print(
                        f"[GIF] make schema={spec.name} vocab={label} split={item_split} sample={idx} frames={len(sequence)} fps={float(args.fps):.1f}",
                        flush=True,
                    )
                    made = _sequence_to_feature_gif(
                        schema=spec.name,
                        sequence=sequence,
                        label=label,
                        split=item_split,
                        video_id=video_id,
                        out_path=out_path,
                        fps=float(args.fps),
                        width=int(args.width),
                        height=int(args.height),
                        max_frames=int(args.max_frames),
                        draw_face=str(getattr(args, "draw_face", "auto")),
                    )
                    generated += 1
                    print(f"[GIF] saved {made}", flush=True)
                except Exception as exc:
                    failed += 1
                    print(f"[GIF][ERR] {spec.name} {item_split}/{label}/{video_id}: {exc}", flush=True)
            if not any_sample:
                print(f"[GIF][WARN] tidak ada sample schema={spec.name} vocab={vocab_name} split={split}", flush=True)
    elapsed = time.perf_counter() - t0
    print(f"gif-dataset: generated={generated} skipped={skipped} failed={failed} elapsed={elapsed:.2f}s", flush=True)
    return 0 if failed == 0 and generated + skipped > 0 else 1


def cmd_gif(args: argparse.Namespace) -> int:
    if args.gif_command == "vocab":
        return cmd_gif_vocab(args)
    if args.gif_command == "dataset":
        return cmd_gif_dataset(args)

    if args.gif_command == "list":
        root = Path(args.gif_dir)
        if args.schema == "all":
            root = root / "samples"
            if args.vocab:
                files = sorted(root.glob(f"*/{clean_label(args.vocab)}/*.gif")) if root.exists() else []
            else:
                files = sorted(root.rglob("*.gif")) if root.exists() else []
            for path in files:
                print(path)
            print(f"total: {len(files)}")
            return 0
        else:
            root = root / "samples" / fs.normalize_schema_name(args.schema)
            if args.vocab:
                root = root / clean_label(args.vocab)
            files = sorted(root.rglob("*.gif")) if root.exists() else []
            for path in files:
                print(path)
            print(f"total: {len(files)}")
            return 0

    if args.gif_command == "check":
        paths = fs.sample_gif_paths(fs.normalize_schema_name(args.schema), clean_label(args.vocab), args.video_id, ROOT_DIR, modes=("overlay", "skeleton"))
        ok = True
        for name, path in paths.items():
            exists = Path(path).exists()
            ok = ok and exists
            print(f"{name}: {'OK' if exists else 'missing'} {path}")
        return 0 if ok else 1

    if args.gif_command == "make":
        paths = make_gif_from_video(args.video, args.label, video_id=args.video_id, out_dir=args.out_dir, schema=args.schema)
        for path in paths:
            print(path)
        return 0

    raise ValueError(f"Unknown gif command: {args.gif_command}")


def _photo_paths(args: argparse.Namespace) -> list[str]:
    return list(args.path or [str(ROOT_DIR / "record" / "photo")])


def cmd_photo_scan(args: argparse.Namespace) -> int:
    import photo_extract as pe

    items = pe.scan_photo_items(_photo_paths(args), default_split=args.split, label=args.label)
    if args.plain:
        print("label\tsplit\tvideo_id\tpath")
        for item in items:
            print(f"{item.label}\t{item.split}\t{item.video_id}\t{item.image_path.resolve()}")
        return 0

    counts: dict[tuple[str, str], int] = {}
    for item in items:
        key = (item.split, item.label)
        counts[key] = counts.get(key, 0) + 1
    print(f"photo-scan: images={len(items)} groups={len(counts)}")
    for (split, label), count in sorted(counts.items()):
        print(f"{split:5s} {label:24s} {count:5d}")
    return 0


def cmd_photo_extract(args: argparse.Namespace) -> int:
    import photo_extract as pe

    schema_names = pe.expand_schema_args(args.schema or [fs.DEFAULT_SCHEMA])
    manifest = pe.extract_photo_session(
        paths=_photo_paths(args),
        schemas=schema_names,
        default_split=args.split,
        label=args.label,
        profile=args.profile,
        workers=max(1, int(args.workers)),
        duplicate_frames=int(args.frames),
        dataset_dir=args.dataset_dir,
        session_root=args.session_root,
        overwrite_existing=bool(args.overwrite_existing),
    )
    summary = manifest.get("summary", {})
    print(f"session\t{manifest['session_dir']}")
    print(
        f"photo-summary\tsuccess={summary.get('success', 0)}\tfailed={summary.get('failed', 0)}\t"
        f"skipped_existing={summary.get('skipped_existing', 0)}"
    )
    return 0 if int(summary.get("success", 0)) + int(summary.get("skipped_existing", 0)) > 0 else 1


def _load_entry_ids_file(path: str | None) -> list[str]:
    if not path:
        return []
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("entry_ids", [])
    if not isinstance(payload, list):
        raise ValueError("--accept-file harus JSON list atau object dengan key entry_ids")
    return [str(value) for value in payload]


def cmd_photo_gif(args: argparse.Namespace) -> int:
    import photo_extract as pe

    manifest = pe.read_manifest(args.session)
    selected = set(args.entry_id or [])
    generated = 0
    skipped = 0
    failed = 0
    for entry in manifest.get("entries", []):
        if entry.get("status") != "success":
            continue
        if selected and entry.get("entry_id") not in selected:
            continue
        try:
            sequence, _frames = pe.load_staged_result(entry)
            out_path = (
                Path(args.session)
                / "gifs"
                / str(entry["schema"])
                / pe.clean_label(str(entry["label"]))
                / f"{pe.safe_name(str(entry['split']))}_{pe.safe_name(str(entry['video_id']))}_feature.gif"
            )
            if out_path.exists() and not args.force:
                skipped += 1
            else:
                _sequence_to_feature_gif(
                    schema=str(entry["schema"]),
                    sequence=sequence,
                    label=str(entry["label"]),
                    split=str(entry["split"]),
                    video_id=str(entry["video_id"]),
                    out_path=out_path,
                    fps=float(args.fps),
                    width=int(args.width),
                    height=int(args.height),
                    max_frames=0,
                    draw_face=str(args.draw_face),
                )
                generated += 1
            entry["gif_temp_path"] = str(out_path)
            print(f"[PHOTO-GIF] {entry['entry_id']} -> {out_path}", flush=True)
        except Exception as exc:
            failed += 1
            print(f"[PHOTO-GIF][ERR] {entry.get('entry_id')}: {exc}", flush=True)

    pe.write_manifest(args.session, manifest)
    print(f"photo-gif: generated={generated} skipped={skipped} failed={failed}", flush=True)
    return 0 if failed == 0 and generated + skipped > 0 else 1


def cmd_photo_commit(args: argparse.Namespace) -> int:
    import photo_extract as pe

    accepted_ids = list(args.entry_id or [])
    accepted_ids.extend(_load_entry_ids_file(args.accept_file))
    result = pe.commit_photo_session(
        session_dir=args.session,
        accepted_entry_ids=accepted_ids or None,
        dataset_dir=args.dataset_dir,
        backup_root=args.backup_root,
        gif_dir=args.gif_dir,
        dry_run=bool(args.dry_run),
        overwrite_existing=bool(args.overwrite_existing),
    )
    print(
        f"photo-commit: scanned={result.scanned} committed={result.committed} skipped={result.skipped} "
        f"failed={result.failed} rows={result.rows} dry_run={result.dry_run}",
        flush=True,
    )
    if result.backup_dir:
        print(f"backup: {result.backup_dir}", flush=True)
    if result.gif_paths:
        print("gifs:")
        for path in result.gif_paths:
            print(f"  {path}")
    return 0 if result.failed == 0 else 1


def cmd_photo(args: argparse.Namespace) -> int:
    if args.photo_command == "scan":
        return cmd_photo_scan(args)
    if args.photo_command == "extract":
        return cmd_photo_extract(args)
    if args.photo_command == "gif":
        return cmd_photo_gif(args)
    if args.photo_command == "commit":
        return cmd_photo_commit(args)
    raise ValueError(f"Unknown photo command: {args.photo_command}")



# ---------------------------------------------------------------------------
# Dataset sample maintenance
# ---------------------------------------------------------------------------
# Keep this block deliberately independent from the live recorder/inference path.
# It only reads/writes parquet files and removes matching GIF/archive files.


def _sample_match_mask(df: pd.DataFrame, *, vocab: str | None, split: str | None, video_id: str | None) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    if vocab:
        if "label" not in df.columns:
            return pd.Series(False, index=df.index)
        mask &= df["label"].astype(str).str.lower() == clean_label(vocab)
    if split and split != "all":
        if "split" not in df.columns:
            return pd.Series(False, index=df.index)
        mask &= df["split"].astype(str).str.lower() == str(split).lower()
    if video_id:
        if "video_id" not in df.columns:
            return pd.Series(False, index=df.index)
        mask &= df["video_id"].astype(str) == str(video_id)
    return mask


def _dataset_sample_rows(
    *,
    schema: str = "all",
    vocab: str | None = None,
    split: str = "all",
    dataset_dir: str | Path = DATASET_DIR,
) -> list[dict]:
    """Return unique samples from parquet files.

    For schema=all, the same sample may appear in multiple schema folders.  The
    output is grouped by (label, split, video_id) and includes schema/frame counts
    so the user can choose the exact sample to delete.
    """

    rows: dict[tuple[str, str, str], dict] = {}
    for schema_name in fs.expand_schema_names(schema):
        spec = fs.get_schema(schema_name)
        for parquet_path in fs.dataset_parquet_paths(spec, dataset_dir):
            if not parquet_path.exists():
                continue
            try:
                df = pd.read_parquet(parquet_path)
            except Exception as exc:
                print(f"[SAMPLE][WARN] gagal baca {parquet_path}: {exc}", flush=True)
                continue
            if df.empty or not {"label", "split", "video_id"}.issubset(df.columns):
                continue
            mask = _sample_match_mask(df, vocab=vocab, split=split, video_id=None)
            df = df[mask]
            if df.empty:
                continue
            grouped = df.groupby(["label", "split", "video_id"], sort=True)
            for (label, split_name, sample_id), group in grouped:
                key = (str(label), str(split_name), str(sample_id))
                item = rows.setdefault(
                    key,
                    {
                        "label": str(label),
                        "split": str(split_name),
                        "video_id": str(sample_id),
                        "schemas": [],
                        "frames_by_schema": {},
                        "rows": 0,
                        "is_augmented": False,
                    },
                )
                item["schemas"].append(spec.name)
                item["frames_by_schema"][spec.name] = int(group["frame_num"].nunique() if "frame_num" in group.columns else len(group))
                item["rows"] += int(len(group))
                if "is_augmented" in group.columns:
                    try:
                        item["is_augmented"] = bool(item["is_augmented"] or group["is_augmented"].fillna(False).astype(bool).any())
                    except Exception:
                        pass
                if "augmented_from" in group.columns:
                    try:
                        item["is_augmented"] = bool(item["is_augmented"] or group["augmented_from"].fillna("").astype(str).str.len().gt(0).any())
                    except Exception:
                        pass
                sample_id_text = str(sample_id).lower()
                if "_augmentation" in sample_id_text or "_aug_" in sample_id_text or "_augmented" in sample_id_text:
                    item["is_augmented"] = True
    out = list(rows.values())
    out.sort(key=lambda x: (x["label"], x["split"], x["video_id"]))
    return out


def _print_sample_rows(rows: list[dict], plain: bool = False) -> None:
    if plain:
        print("label\tsplit\tvideo_id\trows\tschemas\tframes_by_schema\tis_augmented")
        for item in rows:
            frames = ",".join(f"{k}:{v}" for k, v in sorted(item["frames_by_schema"].items()))
            print(
                f"{item['label']}\t{item['split']}\t{item['video_id']}\t{item['rows']}\t"
                f"{','.join(item['schemas'])}\t{frames}\t{int(bool(item['is_augmented']))}"
            )
        return

    if not rows:
        print("Tidak ada sample yang cocok.")
        return
    print(f"{'label':18} {'split':6} {'aug':3} {'rows':6} {'schemas':28} video_id")
    print("-" * 110)
    for item in rows:
        schemas = ",".join(item["schemas"])
        print(f"{item['label'][:18]:18} {item['split'][:6]:6} {int(bool(item['is_augmented'])):<3} {int(item['rows']):<6} {schemas[:28]:28} {item['video_id']}")


def _delete_gifs_for_sample(
    *,
    schema_names: Sequence[str],
    vocab: str,
    split: str,
    video_id: str,
    gif_dir: str | Path = GIF_DIR,
    dry_run: bool = False,
) -> list[Path]:
    """Delete generated GIFs for one sample.

    Covers both feature-GIF filenames from the v3 generator and older
    overlay/skeleton sample GIF names.
    """

    deleted: list[Path] = []
    safe_vid = _safe_name(video_id)
    safe_split = _safe_name(split)
    for schema_name in schema_names:
        spec = fs.get_schema(schema_name)
        sample_dir = Path(gif_dir) / "samples" / spec.name / clean_label(vocab)
        if not sample_dir.exists():
            continue
        patterns = [
            f"*{safe_vid}*.gif",
            f"{safe_split}_{safe_vid}_feature.gif",
            f"{safe_vid}_overlay.gif",
            f"{safe_vid}_skeleton.gif",
        ]
        candidates: dict[Path, None] = {}
        for pattern in patterns:
            for path in sample_dir.glob(pattern):
                candidates[path] = None
        for path in sorted(candidates):
            deleted.append(path)
            if not dry_run:
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass
    return deleted


def _delete_full_archive_for_sample(
    *,
    vocab: str,
    split: str,
    video_id: str,
    full_root: str | Path = DATASET_DIR / "full_features",
    dry_run: bool = False,
) -> list[Path]:
    deleted: list[Path] = []
    root = Path(full_root)
    sample_dir = root / str(split) / clean_label(vocab)
    for path in (sample_dir / f"{video_id}.npz", sample_dir / f"{video_id}_meta.json"):
        if path.exists():
            deleted.append(path)
            if not dry_run:
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass

    manifest = root / "manifest.parquet"
    if manifest.exists():
        try:
            df = pd.read_parquet(manifest)
            if "video_id" in df.columns:
                before = len(df)
                df2 = df[df["video_id"].astype(str) != str(video_id)].copy()
                if len(df2) != before:
                    deleted.append(manifest)
                    if not dry_run:
                        df2.to_parquet(manifest, index=False)
        except Exception as exc:
            print(f"[SAMPLE][WARN] gagal update full manifest {manifest}: {exc}", flush=True)
    return deleted


def delete_dataset_sample(
    *,
    vocab: str,
    video_id: str,
    split: str = "all",
    schema: str = "all",
    dataset_dir: str | Path = DATASET_DIR,
    backup_root: str | Path = BACKUP_ROOT,
    gif_dir: str | Path = GIF_DIR,
    full_root: str | Path = DATASET_DIR / "full_features",
    dry_run: bool = False,
) -> dict:
    """Delete one selected sample across requested schemas and its GIF/archive files."""

    vocab = clean_label(vocab)
    if not vocab:
        raise ValueError("--vocab wajib diisi")
    if not video_id:
        raise ValueError("--video-id wajib diisi")
    if split != "all" and split not in SPLITS:
        raise ValueError("--split harus all/train/val/test")

    schema_names = fs.expand_schema_names(schema)
    backup_dir = Path(backup_root) / f"bisindo_cli_delete_sample_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    copied: set[Path] = set()
    report = {
        "matched_rows": 0,
        "deleted_rows": 0,
        "updated_parquets": [],
        "gif_paths": [],
        "archive_paths": [],
        "backup_dir": str(backup_dir),
        "dry_run": bool(dry_run),
    }

    for schema_name in schema_names:
        spec = fs.get_schema(schema_name)
        for parquet_path in fs.dataset_parquet_paths(spec, dataset_dir):
            if not parquet_path.exists():
                continue
            try:
                df = pd.read_parquet(parquet_path)
            except Exception as exc:
                print(f"[SAMPLE][WARN] gagal baca {parquet_path}: {exc}", flush=True)
                continue
            if df.empty:
                continue
            mask = _sample_match_mask(df, vocab=vocab, split=split, video_id=video_id)
            count = int(mask.sum())
            if count <= 0:
                continue
            report["matched_rows"] += count
            report["deleted_rows"] += count
            report["updated_parquets"].append(str(parquet_path))
            print(f"[SAMPLE-DELETE] schema={schema_name} parquet={parquet_path} rows={count}", flush=True)
            if not dry_run:
                backup_target = backup_dir / spec.name / parquet_path.name
                if parquet_path not in copied:
                    backup_target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(parquet_path, backup_target)
                    copied.add(parquet_path)
                df_out = df[~mask].copy()
                df_out.to_parquet(parquet_path, index=False)

    # Remove GIFs/archive after parquet scan. If split=all, discover actual splits
    # from matching rows first so folder paths are exact.
    split_names = [split] if split != "all" else []
    if split == "all":
        matches = _dataset_sample_rows(schema=schema, vocab=vocab, split="all", dataset_dir=dataset_dir)
        split_names = sorted({item["split"] for item in matches if item["video_id"] == video_id}) or ["train", "val", "test"]
    for split_name in split_names:
        gifs = _delete_gifs_for_sample(schema_names=schema_names, vocab=vocab, split=split_name, video_id=video_id, gif_dir=gif_dir, dry_run=dry_run)
        archive = _delete_full_archive_for_sample(vocab=vocab, split=split_name, video_id=video_id, full_root=full_root, dry_run=dry_run)
        report["gif_paths"].extend(str(p) for p in gifs)
        report["archive_paths"].extend(str(p) for p in archive)

    if copied:
        print(f"backup: {backup_dir}", flush=True)
    return report


def cmd_sample(args: argparse.Namespace) -> int:
    if args.sample_command == "list":
        rows = _dataset_sample_rows(schema=args.schema, vocab=args.vocab, split=args.split, dataset_dir=args.dataset_dir)
        _print_sample_rows(rows, plain=bool(args.plain))
        return 0
    if args.sample_command == "delete":
        report = delete_dataset_sample(
            vocab=args.vocab,
            video_id=args.video_id,
            split=args.split,
            schema=args.schema,
            dataset_dir=args.dataset_dir,
            backup_root=args.backup_root,
            gif_dir=args.gif_dir,
            full_root=args.full_root,
            dry_run=bool(args.dry_run),
        )
        print(
            f"sample-delete: matched_rows={report['matched_rows']} deleted_rows={report['deleted_rows']} "
            f"parquets={len(report['updated_parquets'])} gifs={len(report['gif_paths'])} archive_files={len(report['archive_paths'])} dry_run={report['dry_run']}",
            flush=True,
        )
        if report["gif_paths"]:
            print("deleted_gifs:")
            for path in report["gif_paths"]:
                print(f"  {path}")
        if report["archive_paths"]:
            print("deleted_archive:")
            for path in report["archive_paths"]:
                print(f"  {path}")
        if report["matched_rows"] <= 0:
            print("[SAMPLE-DELETE][WARN] tidak ada row dataset yang cocok. GIF/archive mungkin tetap dicek.", flush=True)
            return 1
        return 0
    raise ValueError(f"Unknown sample command: {args.sample_command}")


def _augment_vocab_values(args: argparse.Namespace) -> list[str | None]:
    if bool(getattr(args, "all_vocab", False)):
        schema_names = getattr(args, "_resolved_schema_names", None)
        schema_value = schema_names if schema_names is not None else getattr(args, "schema", None)
        vocabs = [str(row["label"]) for row in dataset_vocab_summary(schema=schema_value, dataset_dir=args.dataset_dir)]
        return vocabs or [None]
    raw = getattr(args, "vocab", None)
    if raw is None:
        return [None]
    if isinstance(raw, str):
        raw_values = [raw]
    else:
        raw_values = list(raw)
    vocabs = []
    for value in raw_values:
        clean = clean_label(value)
        if clean and clean not in vocabs:
            vocabs.append(clean)
    return vocabs or [None]


def _augment_schema_names(args: argparse.Namespace, default_schema: str) -> tuple[str, ...]:
    raw = getattr(args, "schema", None)
    schema_names = fs.expand_schema_names(raw if raw else default_schema)
    setattr(args, "_resolved_schema_names", schema_names)
    return schema_names


def cmd_augment(args: argparse.Namespace) -> int:
    schema_names = _augment_schema_names(args, fs.DEFAULT_SCHEMA)
    vocabs = _augment_vocab_values(args)
    for schema_name in schema_names:
        for vocab in vocabs:
            result = augment_dataset(
                dataset_dir=args.dataset_dir,
                backup_root=args.backup_root,
                schema=schema_name,
                split=args.split,
                target_per_class=args.target_per_class,
                copies_per_sample=args.copies_per_sample,
                min_source_samples=args.min_source_samples,
                vocab=vocab,
                include_idle=args.include_idle,
                include_augmented_source=args.include_augmented_source,
                seed=args.seed,
                intensity=args.intensity,
                overwrite_existing=bool(args.overwrite_existing),
            )
            vocab_text = vocab or "all"
            print(f"augment[{schema_name}/{vocab_text}]: generated={result.generated} skipped={result.skipped} rows={result.rows}", flush=True)
            if result.backup_dir:
                print(f"backup[{schema_name}/{vocab_text}]: {result.backup_dir}", flush=True)
    return 0


def cmd_augment_delete(args: argparse.Namespace) -> int:
    schema_names = _augment_schema_names(args, "all")
    vocabs = _augment_vocab_values(args)
    for schema_name in schema_names:
        for vocab in vocabs:
            result = delete_augmented_dataset(
                dataset_dir=args.dataset_dir,
                backup_root=args.backup_root,
                schema=schema_name,
                split=args.split,
                vocab=vocab,
                include_idle=args.include_idle,
                dry_run=args.dry_run,
            )
            vocab_text = vocab or "all"
            print(
                f"augment-delete[{schema_name}/{vocab_text}]: deleted_sequences={result.deleted_sequences} "
                f"deleted_rows={result.deleted_rows} files_updated={result.files_updated} "
                f"skipped={result.skipped} dry_run={result.dry_run}",
                flush=True,
            )
            if result.backup_dir:
                print(f"backup[{schema_name}/{vocab_text}]: {result.backup_dir}", flush=True)
    return 0


def _format_live_status(item: dict) -> str:
    top = item.get("top") or []
    top_text = ""
    if top:
        top_text = " top=" + "/".join(f"{label}:{float(score):.2f}" for label, score in top[:3])
    capture_text = ""
    if item.get("capture_width") and item.get("capture_height"):
        capture_text = (
            f"profile={item.get('profile', '-')} "
            f"cap={int(item.get('capture_width'))}x{int(item.get('capture_height'))}@{int(item.get('camera_fps', 0))} "
            f"proc={int(item.get('proc_width', 0))}/pose={int(item.get('pose_proc_width', 0))} "
            f"display={int(item.get('display_width', 0))}x{int(item.get('display_height', 0))} "
        )
    return (
        f"schema={item.get('schema', '-')} variant={item.get('variant', '-')} {capture_text}"
        f"pred={item.get('prediction', '-')} conf={float(item.get('confidence', 0.0)):.2f} "
        f"raw={item.get('raw_prediction', '-')} raw_conf={float(item.get('raw_confidence', 0.0)):.2f} "
        f"cam={float(item.get('fps_camera', item.get('fps', 0.0))):.1f}fps "
        f"pred_fps={float(item.get('fps_predict', 0.0)):.1f} "
        f"extract={float(item.get('extract_ms', 0.0)):.1f}ms model={float(item.get('model_ms', 0.0)):.1f}ms "
        f"segment={int(item.get('segment_len', item.get('buffer', 0)))} "
        f"segment_ms={float(item.get('segment_ms', 0.0)):.0f} sample_fps={float(item.get('sample_fps', 0.0)):.1f} "
        f"buffer={int(item.get('buffer', 0))}/{int(item.get('target_frames', 0))} "
        f"shoulder={int(bool(item.get('shoulder_ok', False)))} "
        f"L={float(item.get('left_present', 0.0)):.0f} R={float(item.get('right_present', 0.0)):.0f} "
        f"visible={int(bool(item.get('visible', False)))}{top_text}"
    )


def cmd_live(args: argparse.Namespace, worker_factory: Callable[..., object] | None = None) -> int:
    import live_gru_fast

    status_queue: queue.Queue = queue.Queue()
    factory = worker_factory or live_gru_fast.start_live_inference
    worker = factory(
        args.variant,
        status_queue=status_queue,
        profile=args.profile,
        device=args.device,
        camera_index=args.camera,
        confidence_threshold=args.threshold,
        show_window=bool(args.window),
        use_jit=not args.no_jit,
        segment_mode=args.segment_mode,
        schema=args.schema,
        route=args.route,
        stream_workers=args.stream_workers,
        mp_workers=args.mp_workers,
        inference_workers=args.inference_workers,
        mp_method=getattr(args, "mp_method", "holistic"),
        specialist_enabled=not bool(getattr(args, "no_specialist", False)),
        specialist_name=getattr(args, "specialist", "all"),
    )
    print("Live terminal aktif. Ctrl+C untuk stop.")
    deadline = time.perf_counter() + float(args.duration) if args.duration else None
    last_print = 0.0
    exit_code = 0
    try:
        while getattr(worker, "is_alive")():
            if deadline is not None and time.perf_counter() >= deadline:
                break
            try:
                item = status_queue.get(timeout=0.20)
            except queue.Empty:
                continue
            event = item.get("event")
            if event == "started":
                capture_text = ""
                if item.get("capture_width") and item.get("capture_height"):
                    capture_text = (
                        f" capture={int(item.get('capture_width'))}x{int(item.get('capture_height'))}@{int(item.get('camera_fps', 0))}"
                        f" proc={int(item.get('proc_width', 0))}/pose={int(item.get('pose_proc_width', 0))}"
                        f" display={int(item.get('display_width', 0))}x{int(item.get('display_height', 0))}"
                    )
                print(
                    f"started: schema={item.get('schema', args.schema)} variant={item.get('variant')} route={item.get('route', args.route)} profile={item.get('profile')} "
                    f"device={item.get('device')}{capture_text} reason={item.get('device_reason', '-')}"
                )
                if item.get("warning"):
                    print(f"warning: {item['warning']}")
            elif event == "status":
                now = time.perf_counter()
                if now - last_print >= float(args.print_interval):
                    last_print = now
                    print(_format_live_status(item), flush=True)
            elif event == "error":
                print(f"error: {item.get('message')}")
                exit_code = 1
                break
            elif event == "stopped":
                closed = " window_closed=1" if item.get("window_closed") else ""
                print(f"stopped: {item.get('message')}{closed}")
                break
    except KeyboardInterrupt:
        print("\nstopping...")
    finally:
        if hasattr(worker, "stop"):
            worker.stop()
        if hasattr(worker, "join"):
            worker.join(timeout=3.0)
    if getattr(worker, "last_error", None):
        print(getattr(worker, "last_error"))
        return 1
    return exit_code


def cmd_live_diagnose(args: argparse.Namespace) -> int:
    import live_gru_fast

    result = live_gru_fast.capture_one_gesture(
        variant=args.variant,
        schema=args.schema,
        profile=args.profile,
        device=args.device,
        camera_index=args.camera,
        out_dir=args.out_dir,
        timeout=args.timeout,
        show_window=bool(args.window),
        use_jit=not args.no_jit,
    )
    print(
        f"diagnose: schema={result.get('schema', args.schema)} variant={result['variant']} profile={result['profile']} device={result['device']} "
        f"frames={result['sequence_frames']} segment_ms={result['segment_ms']:.0f}"
    )
    print(f"prediction: {result['label']} {result['confidence']:.3f}")
    if result.get("top"):
        print("top3: " + " / ".join(f"{label}:{score:.3f}" for label, score in result["top"][:3]))
    print(f"npz: {result['npz']}")
    if result.get("gif"):
        print(f"gif: {result['gif']}")
    return 0


def _live_frame_meta(res, frame_num: int, time_sec: float, profile: str) -> dict:
    return {
        "out_index": int(frame_num),
        "time_sec": float(time_sec),
        "target_source_frame": int(frame_num),
        "chosen_source_frame": int(frame_num),
        "enhance_mode": "live",
        "left_present": float(res.present[0]) if getattr(res, "present", None) is not None else 0.0,
        "right_present": float(res.present[1]) if getattr(res, "present", None) is not None else 0.0,
        "left_detected": float(res.detected[0]) if getattr(res, "detected", None) is not None else 0.0,
        "right_detected": float(res.detected[1]) if getattr(res, "detected", None) is not None else 0.0,
        "left_held": float(res.held[0]) if getattr(res, "held", None) is not None else 0.0,
        "right_held": float(res.held[1]) if getattr(res, "held", None) is not None else 0.0,
        "left_score": float(res.scores[0]) if getattr(res, "scores", None) is not None else 0.0,
        "right_score": float(res.scores[1]) if getattr(res, "scores", None) is not None else 0.0,
        "hand_ms": float(getattr(res, "hand_ms", 0.0)),
        "pose_ms": float(getattr(res, "pose_ms", 0.0)),
        "profile": str(profile),
    }


def _append_live_schema_rows(
    *,
    label: str,
    split: str,
    video_id: str,
    schema: str,
    features: np.ndarray,
    frames: list[dict],
    dataset_dir: str | Path = DATASET_DIR,
    backup_root: str | Path = BACKUP_ROOT,
    camera_index: int = 0,
    profile: str = "accurate10",
    overwrite_existing: bool = False,
) -> ImportResult:
    """Append one live-captured feature sequence to one schema parquet."""

    spec = fs.get_schema(schema)
    features = fs.ensure_feature_dim(features, spec)
    dataset_root = fs.dataset_dir_for(spec, dataset_dir)
    dataset_root.mkdir(parents=True, exist_ok=True)
    parquet_path = dataset_root / f"{label}.parquet"
    result = ImportResult(scanned=1)

    replace_existing = video_id in _existing_video_ids_for_label(dataset_root, label)
    if replace_existing and not overwrite_existing:
        result.skipped = 1
        print(f"[SKIP existing video_id] {spec.name}/{label}/{video_id}", flush=True)
        return result
    if replace_existing:
        print(f"[OVERWRITE existing video_id] {spec.name}/{label}/{video_id}", flush=True)

    item = ImportItem(
        video_path=Path(f"live_camera_{camera_index}"),
        label=label,
        split=split,
        video_id=video_id,
    )
    rows = _rows_from_extract_result(
        item,
        {
            "features": features,
            "frames": frames,
            "feature_mode": spec.feature_mode,
            "target_fps": spec.target_fps,
            "extract_profile": f"live_record_{profile}",
            "smart_mode": "live",
        },
        schema=spec.name,
    )
    backup_dir = Path(backup_root) / f"bisindo_cli_live_record_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    copied: set[Path] = set()
    _backup_existing(parquet_path, backup_dir, copied)
    df_new = pd.DataFrame(rows)
    if parquet_path.exists():
        df_old = pd.read_parquet(parquet_path)
        if replace_existing and "video_id" in df_old.columns:
            mask = df_old["video_id"].astype(str).eq(str(video_id))
            if "label" in df_old.columns:
                mask = mask & df_old["label"].astype(str).map(clean_label).eq(label)
            df_old = df_old.loc[~mask].copy()
        df_out = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_out = df_new
    df_out.to_parquet(parquet_path, index=False)
    print(f"[APPEND parquet +{len(rows)} rows] {parquet_path}", flush=True)
    result.imported = 1
    result.rows = len(rows)
    result.backup_dir = backup_dir if copied else None
    return result


def _save_full_live_archive(
    *,
    label: str,
    split: str,
    video_id: str,
    schema_sequences: dict[str, np.ndarray],
    frames_by_schema: dict[str, list[dict]],
    full_root: str | Path,
    profile: str,
    camera_index: int,
) -> tuple[Path, Path, Path]:
    """Save all captured schema arrays in one full-feature archive folder."""

    root = Path(full_root)
    sample_dir = root / split / label
    sample_dir.mkdir(parents=True, exist_ok=True)
    npz_path = sample_dir / f"{video_id}.npz"
    meta_path = sample_dir / f"{video_id}_meta.json"
    manifest_path = root / "manifest.parquet"

    ordered = {name: np.asarray(arr, dtype=np.float32) for name, arr in schema_sequences.items()}
    payload = {name: arr for name, arr in ordered.items()}
    payload["schemas"] = np.asarray(list(ordered.keys()))
    payload["feature_dims"] = np.asarray([arr.shape[1] for arr in ordered.values()], dtype=np.int32)
    payload["frame_counts"] = np.asarray([arr.shape[0] for arr in ordered.values()], dtype=np.int32)
    payload["label"] = np.asarray(label)
    payload["split"] = np.asarray(split)
    payload["video_id"] = np.asarray(video_id)
    payload["profile"] = np.asarray(profile)
    payload["camera_index"] = np.asarray(int(camera_index), dtype=np.int32)
    np.savez_compressed(npz_path, **payload)

    metadata = {
        "video_saved": False,
        "label": label,
        "split": split,
        "video_id": video_id,
        "profile": profile,
        "camera_index": int(camera_index),
        "schemas": {
            name: (
                {
                    "feature_dim": int(arr.shape[1]),
                    "frames": int(arr.shape[0]),
                    "feature_schema": fs.get_schema(name).feature_schema,
                    "feature_mode": fs.get_schema(name).feature_mode,
                    "columns": fs.feature_column_names(name),
                }
                if name in fs.SCHEMAS
                else {
                    "feature_dim": int(arr.shape[1]),
                    "frames": int(arr.shape[0]),
                    "feature_schema": "mediapipe_holistic_full_raw",
                    "feature_mode": "full_raw_right_left_pose_face_shoulders",
                    "columns": fs.full_raw_feature_column_names(),
                }
            )
            for name, arr in ordered.items()
        },
        "frames_by_schema": frames_by_schema,
    }
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    row = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "video_saved": False,
        "label": label,
        "split": split,
        "video_id": video_id,
        "profile": profile,
        "camera_index": int(camera_index),
        "schemas": ",".join(ordered.keys()),
        "feature_dims": ",".join(str(arr.shape[1]) for arr in ordered.values()),
        "frame_counts": ",".join(str(arr.shape[0]) for arr in ordered.values()),
        "npz_path": str(npz_path),
        "meta_path": str(meta_path),
    }
    df_new = pd.DataFrame([row])
    if manifest_path.exists():
        df_old = pd.read_parquet(manifest_path)
        df_out = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_out = df_new
    df_out.to_parquet(manifest_path, index=False)
    return npz_path, meta_path, manifest_path



def _preview_resize(frame: np.ndarray, preview_width: int = 640) -> np.ndarray:
    import cv2

    preview_width = int(preview_width or 0)
    if preview_width <= 0 or frame.shape[1] <= preview_width:
        return frame
    scale = preview_width / float(frame.shape[1])
    return cv2.resize(frame, (preview_width, max(1, int(round(frame.shape[0] * scale)))), interpolation=cv2.INTER_AREA)


def _draw_record_overlay(frame: np.ndarray, lines: Sequence[str]) -> np.ndarray:
    import cv2

    vis = frame.copy()
    y = 24
    for text in lines:
        cv2.putText(vis, str(text), (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(vis, str(text), (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        y += 22
    return vis


def _read_latest_sized_frame(cap, profile_cfg: dict) -> np.ndarray | None:
    import cv2
    from smart_extract.live_bisindo_mp_real_shoulder_v6 import center_crop

    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    width = int(profile_cfg["width"])
    height = int(profile_cfg["height"])
    if frame.shape[1] != width or frame.shape[0] != height:
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    return center_crop(frame, 1.0)


def _capture_raw_live_frames(
    *,
    duration: float,
    camera_index: int,
    profile: str,
    target_fps: float,
    window: bool,
    pre_delay: float,
    preview_width: int = 640,
) -> tuple[list[np.ndarray], list[float], dict[str, float | int | str]]:
    """Capture BGR frames at exactly target_fps into RAM; do not save video."""

    import cv2
    import live_gru_fast
    from smart_extract.live_bisindo_mp_real_shoulder_v6 import LatestFrameCamera

    profile_name = live_gru_fast.normalize_profile(profile)
    profile_cfg = live_gru_fast.LIVE_PROFILES[profile_name]
    sample_fps = float(target_fps)
    sample_interval = 1.0 / max(sample_fps, 1e-6)
    target_frames = max(1, int(round(max(0.05, float(duration)) * sample_fps)))

    cap = None
    frames: list[np.ndarray] = []
    times: list[float] = []
    preview_i = 0
    preview_t0 = time.perf_counter()
    preview_fps = 0.0
    latest: np.ndarray | None = None

    try:
        cap = LatestFrameCamera(
            src=int(camera_index),
            width=int(profile_cfg["width"]),
            height=int(profile_cfg["height"]),
            fps=int(profile_cfg.get("camera_fps", 30)),
            use_gstreamer=bool(int(profile_cfg.get("use_gstreamer", 0))),
            fourcc="MJPG",
        ).start()

        if pre_delay > 0:
            ready_deadline = time.perf_counter() + float(pre_delay)
            while time.perf_counter() < ready_deadline:
                frame = _read_latest_sized_frame(cap, profile_cfg)
                if frame is not None:
                    latest = frame
                    if window:
                        remain = max(0.0, ready_deadline - time.perf_counter())
                        vis = _draw_record_overlay(
                            _preview_resize(frame, preview_width),
                            [
                                f"READY {remain:.1f}s | target save {sample_fps:.1f} FPS",
                                "video_saved=no | feature-only dataset recorder",
                            ],
                        )
                        cv2.imshow("BISINDO live dataset recorder", vis)
                if window and (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                    raise KeyboardInterrupt
                time.sleep(0.005)

        capture_start = time.perf_counter()
        for sample_idx in range(target_frames):
            target_time = capture_start + sample_idx * sample_interval
            while True:
                now = time.perf_counter()
                frame = _read_latest_sized_frame(cap, profile_cfg)
                if frame is not None:
                    latest = frame
                    preview_i += 1
                    elapsed_preview = now - preview_t0
                    if elapsed_preview >= 0.50:
                        inst = preview_i / max(elapsed_preview, 1e-6)
                        preview_fps = inst if preview_fps <= 0 else 0.75 * preview_fps + 0.25 * inst
                        preview_i = 0
                        preview_t0 = now
                    if window:
                        vis = _draw_record_overlay(
                            _preview_resize(frame, preview_width),
                            [
                                f"CAPTURE {sample_idx}/{target_frames} | preview {preview_fps:.1f} FPS | saved target {sample_fps:.1f} FPS",
                                f"profile={profile_name} cam={camera_index} | video_saved=no",
                            ],
                        )
                        cv2.imshow("BISINDO live dataset recorder", vis)

                if window and (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                    raise KeyboardInterrupt
                if now >= target_time:
                    break
                time.sleep(min(0.006, max(0.001, target_time - now)))

            if latest is None:
                # Give the camera a little more time instead of silently losing the target slot.
                wait_until = time.perf_counter() + 0.25
                while latest is None and time.perf_counter() < wait_until:
                    latest = _read_latest_sized_frame(cap, profile_cfg)
                    time.sleep(0.005)
            if latest is None:
                raise RuntimeError("Kamera belum mengirim frame.")
            frames.append(latest.copy())
            times.append(sample_idx / sample_fps)
            print(
                f"[CAPTURE] frame={sample_idx + 1}/{target_frames} target_fps={sample_fps:.1f} preview_fps={preview_fps:.1f}",
                flush=True,
            )

        capture_elapsed = time.perf_counter() - capture_start
        metadata = {
            "capture_mode": "memory_frames_then_feature_extract",
            "profile": profile_name,
            "target_fps": sample_fps,
            "target_frames": target_frames,
            "captured_frames": len(frames),
            "capture_elapsed_sec": float(capture_elapsed),
            "effective_capture_fps": float(len(frames) / max(capture_elapsed, 1e-6)),
        }
        return frames, times, metadata
    finally:
        if cap is not None:
            cap.release()


def _extract_live_features_from_frames(
    *,
    raw_frames: Sequence[np.ndarray],
    frame_times: Sequence[float],
    schemas: Sequence[str],
    profile: str,
    target_fps: float,
    window: bool,
    only_visible: bool,
    preview_width: int = 640,
) -> tuple[dict[str, np.ndarray], dict[str, list[dict]]]:
    """Convert already-captured RAM frames into all requested feature schemas."""

    import cv2
    import live_gru_fast
    import holistic_features
    from smart_extract.live_bisindo_mp_real_shoulder_v6 import (
        UltraMediaPipeExtractor,
        draw_shoulders,
        draw_simple_hand,
    )

    schema_names = tuple(fs.normalize_schema_name(name) for name in schemas)
    profile_name = live_gru_fast.normalize_profile(profile)
    profile_cfg = live_gru_fast.LIVE_PROFILES[profile_name]

    smart_extractor = None
    holistic_extractor = None
    sequences: dict[str, list[np.ndarray]] = {name: [] for name in schema_names}
    frames: dict[str, list[dict]] = {name: [] for name in schema_names}
    raw_full_sequence: list[np.ndarray] = []
    holistic_names = [name for name in schema_names if fs.get_schema(name).extractor == "holistic"]
    extract_t0 = time.perf_counter()

    try:
        if "smart180" in schema_names:
            smart_extractor = UltraMediaPipeExtractor(
                feature_mode=sc.FEATURE_MODE,
                shoulder_backend=str(profile_cfg["shoulder_backend"]),
                proc_width=int(profile_cfg["proc_width"]),
                pose_proc_width=int(profile_cfg["pose_proc_width"]),
                pose_every=int(profile_cfg["pose_every"]),
                hand_every=int(profile_cfg["hand_every"]),
                hand_model_complexity=0,
                pose_model_complexity=0,
                det_conf=float(profile_cfg["det_conf"]),
                track_conf=float(profile_cfg["track_conf"]),
                smooth_alpha=float(profile_cfg["smooth_alpha"]),
                hold_frames=int(profile_cfg["hold_frames"]),
                mirror_input=False,
                mirror_handedness=True,
                shoulder_smooth_alpha=float(profile_cfg["shoulder_smooth_alpha"]),
            )
        if holistic_names:
            holistic_extractor = holistic_features.MultiSchemaHolisticExtractor(
                holistic_names,
                proc_width=int(profile_cfg["proc_width"]),
                det_conf=float(profile_cfg["det_conf"]),
                track_conf=float(profile_cfg["track_conf"]),
                model_complexity=0,
                smooth_landmarks=True,
                refine_face_landmarks=False,
            )

        for frame_idx, work in enumerate(raw_frames):
            frame_t0 = time.perf_counter()
            results_by_schema = {}
            if smart_extractor is not None:
                results_by_schema["smart180"] = smart_extractor.process(work)
            if holistic_extractor is not None:
                results_by_schema.update(holistic_extractor.process(work))
                for _res in results_by_schema.values():
                    raw_full = getattr(_res, "raw_full", None)
                    if raw_full and raw_full.get("vector") is not None:
                        raw_full_sequence.append(np.asarray(raw_full["vector"], dtype=np.float32))
                        break

            preview_res = results_by_schema.get("smart180") or next(iter(results_by_schema.values()), None)
            time_sec = float(frame_times[frame_idx] if frame_idx < len(frame_times) else frame_idx / max(float(target_fps), 1e-6))
            for name in schema_names:
                res = results_by_schema.get(name)
                if res is None:
                    continue
                visible = bool(res.present[0] >= 0.5 or res.present[1] >= 0.5)
                if only_visible and not visible:
                    continue
                sequences[name].append(np.asarray(res.vector, dtype=np.float32))
                meta = _live_frame_meta(res, len(sequences[name]) - 1, time_sec, profile_name)
                meta["target_source_frame"] = int(frame_idx)
                meta["chosen_source_frame"] = int(frame_idx)
                meta["source_sample_index"] = int(frame_idx)
                meta["saved_target_fps"] = float(target_fps)
                meta["enhance_mode"] = "live_memory"
                frames[name].append(meta)

            extract_elapsed = time.perf_counter() - extract_t0
            extract_fps = (frame_idx + 1) / max(extract_elapsed, 1e-6)
            frame_ms = (time.perf_counter() - frame_t0) * 1000.0
            print(
                f"[EXTRACT] frame={frame_idx + 1}/{len(raw_frames)} extract_fps={extract_fps:.1f} frame_ms={frame_ms:.1f} saved_fps={float(target_fps):.1f}",
                flush=True,
            )
            if window:
                vis = _preview_resize(work, preview_width)
                if preview_res is not None:
                    draw_shoulders(vis, preview_res.shoulders)
                    draw_simple_hand(vis, preview_res.left, (0, 255, 0))
                    draw_simple_hand(vis, preview_res.right, (0, 180, 255))
                lines = [
                    f"PROCESS FEATURES {frame_idx + 1}/{len(raw_frames)} | extract {extract_fps:.1f} FPS",
                    f"stored target {float(target_fps):.1f} FPS | schemas={','.join(schema_names)} | no video",
                ]
                cv2.imshow("BISINDO live dataset recorder", _draw_record_overlay(vis, lines))
                if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                    raise KeyboardInterrupt
    finally:
        if smart_extractor is not None:
            smart_extractor.close()
        if holistic_extractor is not None:
            holistic_extractor.close()

    arrays: dict[str, np.ndarray] = {}
    for name in schema_names:
        if sequences[name]:
            arrays[name] = fs.ensure_feature_dim(np.stack(sequences[name]).astype(np.float32), name)
        else:
            arrays[name] = np.zeros((0, fs.get_schema(name).feature_dim), dtype=np.float32)
    if raw_full_sequence:
        arrays["full_raw"] = np.stack(raw_full_sequence).astype(np.float32)
    return arrays, frames


def _capture_live_feature_sample(
    *,
    schemas: Sequence[str],
    duration: float,
    camera_index: int,
    profile: str,
    target_fps: float | None = None,
    window: bool = False,
    pre_delay: float = 0.0,
    only_visible: bool = False,
    preview_width: int = 640,
) -> tuple[dict[str, np.ndarray], dict[str, list[dict]]]:
    """Capture one live sample directly as features, without VideoWriter.

    Important: recording is now two-phase by default:
    1. Capture RAM frames at exactly target_fps with a lightweight preview.
    2. Extract all requested schemas from those frames after capture.

    This keeps the live preview responsive and guarantees that the saved feature
    sequence has the expected 10-FPS frame count, even if feature extraction for
    schema=all is slower than real time on Jetson/CPU.
    """

    import cv2

    schema_names = tuple(fs.normalize_schema_name(name) for name in schemas)
    sample_fps = float(target_fps or max(fs.get_schema(name).target_fps for name in schema_names))

    raw_frames, frame_times, capture_meta = _capture_raw_live_frames(
        duration=float(duration),
        camera_index=int(camera_index),
        profile=str(profile),
        target_fps=sample_fps,
        window=bool(window),
        pre_delay=float(pre_delay),
        preview_width=int(preview_width),
    )
    print(
        f"[CAPTURE-DONE] frames={len(raw_frames)} target_fps={sample_fps:.1f} "
        f"effective_capture_fps={float(capture_meta.get('effective_capture_fps', 0.0)):.1f} video_saved=no",
        flush=True,
    )

    sequences, frames = _extract_live_features_from_frames(
        raw_frames=raw_frames,
        frame_times=frame_times,
        schemas=schema_names,
        profile=str(profile),
        target_fps=sample_fps,
        window=bool(window),
        only_visible=bool(only_visible),
        preview_width=int(preview_width),
    )
    if window:
        cv2.destroyWindow("BISINDO live dataset recorder")
    return sequences, frames

def cmd_record_live(args: argparse.Namespace) -> int:
    label = clean_label(args.label)
    if not label:
        raise ValueError("--label wajib untuk record-live")
    split = str(args.split or "train").lower()
    if split not in SPLITS:
        raise ValueError("split harus salah satu: train, val, test")

    schema_names = fs.expand_schema_names(args.schema)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    full_root = Path(args.full_root)
    exit_code = 0

    for idx in range(1, int(args.count) + 1):
        video_id = f"{split}_live_{label}_{stamp}_{idx:03d}"
        print(
            f"[LIVE-REC] {idx}/{args.count} label={label} split={split} id={video_id} "
            f"schema={','.join(schema_names)} video_saved=no"
        )
        try:
            sequences, frames = _capture_live_feature_sample(
                schemas=schema_names,
                duration=float(args.duration),
                camera_index=int(args.camera),
                profile=str(args.profile),
                target_fps=float(args.target_fps),
                window=bool(args.window),
                pre_delay=float(args.pre_delay),
                only_visible=bool(args.only_visible),
                preview_width=int(args.preview_width),
            )
        except KeyboardInterrupt:
            print("[LIVE-REC] dibatalkan user.")
            return 130

        train_sequences = {name: arr for name, arr in sequences.items() if name in fs.SCHEMAS}
        if any(arr.shape[0] < int(args.min_frames) for arr in train_sequences.values()):
            print(
                f"[WARN] sample terlalu pendek: "
                + ", ".join(f"{name}={arr.shape[0]}" for name, arr in train_sequences.items())
            )
            exit_code = 1
            if not args.keep_short:
                continue

        if not args.no_full_archive:
            npz_path, meta_path, manifest_path = _save_full_live_archive(
                label=label,
                split=split,
                video_id=video_id,
                schema_sequences=sequences,
                frames_by_schema=frames,
                full_root=full_root,
                profile=str(args.profile),
                camera_index=int(args.camera),
            )
            print(f"[FULL] {npz_path}")
            print(f"[FULL-META] {meta_path}")
            print(f"[FULL-MANIFEST] {manifest_path}")

        for schema_name, arr in sequences.items():
            if schema_name not in fs.SCHEMAS:
                continue
            if arr.shape[0] == 0:
                print(f"[SKIP] {schema_name}: 0 frame")
                exit_code = 1
                continue
            result = _append_live_schema_rows(
                label=label,
                split=split,
                video_id=video_id,
                schema=schema_name,
                features=arr,
                frames=frames[schema_name],
                dataset_dir=args.dataset_dir,
                backup_root=args.backup_root,
                camera_index=int(args.camera),
                profile=str(args.profile),
                overwrite_existing=bool(args.overwrite_existing),
            )
            print(
                f"record-live[{schema_name}]: imported={result.imported} skipped={result.skipped} "
                f"rows={result.rows} dim={arr.shape[1]} frames={arr.shape[0]}"
            )
            if result.backup_dir:
                print(f"backup[{schema_name}]: {result.backup_dir}")
    return exit_code


def _record_one_video(
    out_path: Path,
    camera_index: int,
    duration: float,
    width: int,
    height: int,
    fps: float,
    window: bool = False,
) -> None:
    import cv2

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(int(camera_index))
    if not cap.isOpened():
        raise RuntimeError(f"Kamera tidak bisa dibuka: {camera_index}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
    cap.set(cv2.CAP_PROP_FPS, float(fps))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, float(fps), (int(width), int(height)))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Tidak bisa menulis video: {out_path}")

    deadline = time.perf_counter() + max(0.1, float(duration))
    try:
        while time.perf_counter() < deadline:
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.005)
                continue
            if frame.shape[1] != int(width) or frame.shape[0] != int(height):
                frame = cv2.resize(frame, (int(width), int(height)))
            writer.write(frame)
            if window:
                cv2.imshow("BISINDO record", frame)
                if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                    break
    finally:
        writer.release()
        cap.release()
        if window:
            cv2.destroyAllWindows()


def cmd_record(args: argparse.Namespace) -> int:
    label = clean_label(args.label)
    if not label:
        raise ValueError("--label wajib untuk record")
    raw_root = Path(args.raw_root)
    created: list[ImportItem] = []
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for idx in range(1, int(args.count) + 1):
        video_id = f"{args.split}_{label}_{stamp}_{idx:03d}.mp4"
        out_path = raw_root / args.split / label / video_id
        print(f"[RECORD] {idx}/{args.count} -> {out_path}")
        _record_one_video(
            out_path,
            camera_index=args.camera,
            duration=args.duration,
            width=args.width,
            height=args.height,
            fps=args.fps,
            window=bool(args.window),
        )
        created.append(ImportItem(video_path=out_path, label=label, split=args.split, video_id=video_id))

    exit_code = 0
    for schema_name in fs.expand_schema_names(args.schema):
        result = append_import_items(
            created,
            dataset_dir=args.dataset_dir,
            backup_root=args.backup_root,
            schema=schema_name,
            save_gif=args.save_gif,
            quiet=args.quiet,
            overwrite_existing=bool(args.overwrite_existing),
        )
        print(
            f"record-import[{schema_name}]: imported={result.imported} skipped={result.skipped} "
            f"failed={result.failed} rows={result.rows}"
        )
        if result.failed:
            exit_code = 1
    return exit_code


def cmd_gui(args: argparse.Namespace) -> int:
    return subprocess.call([sys.executable, str(ROOT_DIR / "src" / "main_ui.py")])


def menu_loop() -> int:
    options = {
        "1": ["status"],
        "2": ["live"],
        "3": ["live", "--window"],
        "4": ["train", "--variant", "all"],
        "5": ["benchmark", "--variant", "all", "--device", "auto"],
        "6": ["gui"],
        "7": ["diagnose-live"],
        "8": ["record-live"],
        "9": ["gif", "vocab", "--schema", "all"],
    }
    while True:
        print()
        print("BISINDO CLI")
        print("1. Status")
        print("2. Live terminal")
        print("3. Live window")
        print("4. Train semua GRU")
        print("5. Benchmark GRU")
        print("6. GUI")
        print("7. Diagnose satu gesture live")
        print("8. Record dataset live feature-only (tanpa simpan video)")
        print("9. List vocab untuk GIF dataset")
        print("q. Keluar")
        choice = input("Pilih: ").strip().lower()
        if choice in {"q", "quit", "exit"}:
            return 0
        if choice == "8":
            label = input("Label vocab: ").strip()
            argv = ["record-live", "--label", label, "--window"] if label else None
        elif choice == "9":
            label = input("Generate GIF untuk vocab tertentu? kosong = list saja: ").strip()
            if label:
                schema = input("Schema (all/smart180/khukuh1629/adi1662) [all]: ").strip() or "all"
                split = input("Split (all/train/val/test) [all]: ").strip() or "all"
                argv = ["gif", "dataset", "--schema", schema, "--vocab", label, "--split", split]
            else:
                argv = options.get(choice)
        else:
            argv = options.get(choice)
        if argv is None:
            print("Pilihan tidak dikenal.")
            continue
        code = main(argv)
        if code:
            print(f"Command selesai dengan exit code {code}.")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="BISINDO terminal workflow")
    sub = parser.add_subparsers(dest="command")

    common_data = argparse.ArgumentParser(add_help=False)
    common_data.add_argument("--dataset-dir", default=str(DATASET_DIR))
    common_data.add_argument("--model-dir", default=str(MODEL_DIR))

    status = sub.add_parser("status", parents=[common_data], help="Cek dataset, model, runtime, dan GIF")
    status.add_argument("--schema", default="all", choices=SCHEMA_CHOICES)
    status.add_argument("--gif-dir", default=str(GIF_DIR))
    status.set_defaults(func=cmd_status)

    live = sub.add_parser("live", help="Live GRU dari terminal")
    live.add_argument("--schema", default=fs.DEFAULT_SCHEMA, choices=list(fs.SCHEMA_NAMES))
    live.add_argument("--variant", default="auto", help="Varian GRU, gru_*, atau auto")
    live.add_argument("--profile", "--mode", dest="profile", default=DEFAULT_LIVE_PROFILE, choices=LIVE_PROFILE_CHOICES)
    live.add_argument("--segment-mode", default="auto", choices=["auto", "rolling"])
    live.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    live.add_argument("--camera", type=int, default=0)
    live.add_argument("--threshold", type=float, default=0.65)
    live.add_argument(
        "--route",
        default="main",
        choices=["main", "chunk10", "threshold", "vote_all", "main_chunk10", "main_threshold", "boosted_stack"],
        help="Strategi routed inference multi-model",
    )
    live.add_argument("--stream-workers", type=int, default=1, help="Worker sampler kamera untuk pipeline live")
    live.add_argument("--mp-workers", type=int, default=1, help="Worker MediaPipe untuk pipeline live")
    live.add_argument("--inference-workers", type=int, default=1, help="Worker inference GRU untuk pipeline live")
    live.add_argument("--mp-method", default="holistic", choices=["holistic", "holistic_stabilized"], help="Metode ekstraksi MediaPipe (stabilized = anti-jitter/anti-missing)")
    live.add_argument("--window", action="store_true", help="Buka overlay kamera OpenCV")
    live.add_argument("--duration", type=float, default=0.0, help="Stop otomatis setelah N detik; 0 = jalan terus")
    live.add_argument("--print-interval", type=float, default=0.5)
    live.add_argument("--no-jit", action="store_true")
    live.add_argument("--specialist", default="all", help="Nama specialist aktif, atau all untuk semua specialist")
    live.add_argument("--no-specialist", action="store_true", help="Matikan auto-specialist routing")
    live.set_defaults(func=cmd_live)

    diagnose = sub.add_parser("diagnose-live", help="Capture satu gesture live, simpan NPZ/GIF, dan prediksi top-3")
    diagnose.add_argument("--schema", default=fs.DEFAULT_SCHEMA, choices=list(fs.SCHEMA_NAMES))
    diagnose.add_argument("--variant", default="auto", help="Varian GRU, gru_*, atau auto")
    diagnose.add_argument("--profile", "--mode", dest="profile", default=DEFAULT_LIVE_PROFILE, choices=LIVE_PROFILE_CHOICES)
    diagnose.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    diagnose.add_argument("--camera", type=int, default=0)
    diagnose.add_argument("--timeout", type=float, default=12.0)
    diagnose.add_argument("--out-dir", default=None)
    diagnose.add_argument("--window", action="store_true")
    diagnose.add_argument("--no-jit", action="store_true")
    diagnose.add_argument("--mp-method", default="holistic", choices=["holistic", "holistic_stabilized"], help="Metode ekstraksi MediaPipe (stabilized = anti-jitter/anti-missing)")
    diagnose.set_defaults(func=cmd_live_diagnose)

    train = sub.add_parser("train", parents=[common_data], help="Train GRU")
    train.add_argument(
        "--schema",
        action="append",
        default=None,
        metavar="SCHEMA",
        help="Schema target; bisa comma-list atau diulang. Default: smart180",
    )
    train.add_argument("--variant", default="all", help="Varian GRU, comma list, atau all")
    train.add_argument("--train-data", default="original", choices=["original", "with-augmentation", "with_augmentation", "both"])
    train.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    train.add_argument("--epochs", type=int, default=None)
    train.add_argument("--batch-size", type=int, default=None)
    train.add_argument("--lr", type=float, default=None)
    train.add_argument("--patience", type=int, default=None)
    train.add_argument("--limit-per-class", type=int, default=None)
    train.add_argument("--overwrite-existing", action="store_true", help="Backup lalu timpa checkpoint target yang sudah ada")
    train.add_argument("--backup-root", default=str(BACKUP_ROOT))
    train.set_defaults(func=cmd_train)

    train_suite = sub.add_parser("train-suite", parents=[common_data], help="Train main GRU + expert suites chunk10/threshold/boosted")
    train_suite.add_argument(
        "--schema",
        action="append",
        default=None,
        metavar="SCHEMA",
        help="Schema target; bisa comma-list atau diulang. Default: smart180",
    )
    train_suite.add_argument("--variant", default="all", help="Varian GRU, comma list, atau all")
    train_suite.add_argument("--train-data", default="original", choices=["original", "with-augmentation", "with_augmentation", "both"])
    train_suite.add_argument("--suite", default="main,chunk10,threshold,boosted", help="Comma list: main,chunk10,threshold,boosted atau all")
    train_suite.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    train_suite.add_argument("--epochs", type=int, default=None)
    train_suite.add_argument("--batch-size", type=int, default=None)
    train_suite.add_argument("--lr", type=float, default=None)
    train_suite.add_argument("--patience", type=int, default=None)
    train_suite.add_argument("--limit-per-class", type=int, default=None)
    train_suite.add_argument("--threshold-distance", type=float, default=None)
    train_suite.add_argument("--overwrite-existing", action="store_true", help="Backup lalu timpa checkpoint/suite target yang sudah ada")
    train_suite.add_argument("--backup-root", default=str(BACKUP_ROOT))
    train_suite.set_defaults(func=cmd_train_suite)

    eval_cmd = sub.add_parser("eval", parents=[common_data], help="Evaluasi checkpoint GRU")
    eval_cmd.add_argument("--schema", default=fs.DEFAULT_SCHEMA, choices=SCHEMA_CHOICES)
    eval_cmd.add_argument("--variant", default="all", help="Varian GRU, comma list, atau all")
    eval_cmd.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    eval_cmd.add_argument("--split", default="test", choices=["train", "val", "test"])
    eval_cmd.add_argument("--suite", default="main", help="main, route expert, comma list, atau all")
    eval_cmd.set_defaults(func=cmd_eval)

    bench = sub.add_parser("benchmark", parents=[common_data], help="Benchmark latency GRU")
    bench.add_argument("--schema", default=fs.DEFAULT_SCHEMA, choices=SCHEMA_CHOICES)
    bench.add_argument("--variant", default="all", help="Varian GRU, comma list, atau all")
    bench.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    bench.add_argument("--warmup", type=int, default=5)
    bench.add_argument("--runs", type=int, default=30)
    bench.add_argument("--threads", type=int, default=1)
    bench.add_argument("--no-jit", action="store_true")
    bench.set_defaults(func=cmd_benchmark)

    import_cmd = sub.add_parser("import", help="Append aman video ke dataset schema")
    import_cmd.add_argument("--schema", default=fs.DEFAULT_SCHEMA, choices=SCHEMA_CHOICES)
    import_cmd.add_argument("--path", action="append", required=True, help="Video/folder; bisa diulang")
    import_cmd.add_argument("--label", default=None, help="Wajib untuk single video")
    import_cmd.add_argument("--split", default="train", choices=["train", "val", "test"])
    import_cmd.add_argument("--dataset-dir", default=str(DATASET_DIR))
    import_cmd.add_argument("--backup-root", default=str(BACKUP_ROOT))
    import_cmd.add_argument("--save-gif", action="store_true")
    import_cmd.add_argument("--quiet", action="store_true")
    import_cmd.add_argument("--overwrite-existing", action="store_true", help="Backup lalu replace rows dengan video_id yang sama")
    import_cmd.set_defaults(func=cmd_import)

    extract_full = sub.add_parser("extract-full", parents=[common_data], help="Rebuild parquet dataset dari dataset_full_mediapipe landmark NPZ")
    extract_full.add_argument("--source", default=str(ROOT_DIR / "dataset_full_mediapipe"))
    extract_full.add_argument(
        "--schema",
        action="append",
        default=None,
        metavar="SCHEMA",
        help="Schema target; bisa comma-list atau diulang. Default: full",
    )
    extract_full.add_argument("--vocab", action="append", default=None, help="Opsional: vocab/label; bisa comma-list atau diulang")
    extract_full.add_argument("--clean", default="none", choices=["none", "backup"], help="backup = pindahkan dataset/model generated lama dulu")
    extract_full.add_argument("--backup-root", default=str(BACKUP_ROOT))
    extract_full.add_argument("--limit-per-class", type=int, default=None)
    extract_full.set_defaults(func=cmd_extract_full)


    record_live = sub.add_parser("record-live", help="Record dataset langsung dari kamera: simpan feature parquet + full archive, tanpa menyimpan video")
    record_live.add_argument("--schema", default="all", choices=SCHEMA_CHOICES)
    record_live.add_argument("--label", required=True)
    record_live.add_argument("--split", default="train", choices=["train", "val", "test"])
    record_live.add_argument("--count", type=int, default=1)
    record_live.add_argument("--duration", type=float, default=2.5)
    record_live.add_argument("--camera", type=int, default=0)
    record_live.add_argument("--profile", "--mode", dest="profile", default="fast10", choices=LIVE_PROFILE_CHOICES)
    record_live.add_argument("--target-fps", type=float, default=10.0)
    record_live.add_argument("--pre-delay", type=float, default=0.8, help="Delay sebelum mulai capture agar tangan siap")
    record_live.add_argument("--min-frames", type=int, default=8)
    record_live.add_argument("--keep-short", action="store_true", help="Tetap simpan walau frame kurang dari --min-frames")
    record_live.add_argument("--only-visible", action="store_true", help="Simpan hanya frame yang ada tangan terdeteksi")
    record_live.add_argument("--dataset-dir", default=str(DATASET_DIR))
    record_live.add_argument("--backup-root", default=str(BACKUP_ROOT))
    record_live.add_argument("--full-root", default=str(DATASET_DIR / "full_features"))
    record_live.add_argument("--no-full-archive", action="store_true", help="Matikan arsip full_features .npz/.json")
    record_live.add_argument("--window", action="store_true", help="Tampilkan preview OpenCV saat record")
    record_live.add_argument("--preview-width", type=int, default=640, help="Lebar window preview; kecilkan ke 480/360 kalau Jetson lag")
    record_live.add_argument("--overwrite-existing", action="store_true", help="Backup lalu replace rows dengan video_id yang sama")
    record_live.set_defaults(func=cmd_record_live)

    record = sub.add_parser("record", help="Rekam MP4 mentah lalu extract ke dataset schema")
    record.add_argument("--schema", default=fs.DEFAULT_SCHEMA, choices=SCHEMA_CHOICES)
    record.add_argument("--label", required=True)
    record.add_argument("--split", default="train", choices=["train", "val", "test"])
    record.add_argument("--count", type=int, default=1)
    record.add_argument("--duration", type=float, default=2.5)
    record.add_argument("--camera", type=int, default=0)
    record.add_argument("--width", type=int, default=640)
    record.add_argument("--height", type=int, default=480)
    record.add_argument("--fps", type=float, default=30.0)
    record.add_argument("--raw-root", default=str(ROOT_DIR / "record" / "raw"))
    record.add_argument("--dataset-dir", default=str(DATASET_DIR))
    record.add_argument("--backup-root", default=str(BACKUP_ROOT))
    record.add_argument("--save-gif", action="store_true")
    record.add_argument("--window", action="store_true")
    record.add_argument("--quiet", action="store_true")
    record.add_argument("--overwrite-existing", action="store_true", help="Backup lalu replace rows dengan video_id yang sama")
    record.set_defaults(func=cmd_record)

    photo = sub.add_parser("photo", help="Import foto dataset live-like lewat staging review")
    photo_sub = photo.add_subparsers(dest="photo_command", required=True)

    photo_scan = photo_sub.add_parser("scan", help="Scan layout folder foto tanpa ekstraksi")
    photo_scan.add_argument("--path", action="append", default=None, help="Image/folder foto; default record/photo")
    photo_scan.add_argument("--label", default=None, help="Label untuk single image/folder paksa")
    photo_scan.add_argument("--split", default="train", choices=["train", "val", "test"])
    photo_scan.add_argument("--plain", action="store_true")
    photo_scan.set_defaults(func=cmd_photo)

    photo_extract = photo_sub.add_parser("extract", help="Extract foto ke temp session untuk direview")
    photo_extract.add_argument("--path", action="append", default=None, help="Image/folder foto; default record/photo")
    photo_extract.add_argument("--schema", action="append", default=None, choices=SCHEMA_CHOICES, help="Schema target; bisa diulang")
    photo_extract.add_argument("--label", default=None, help="Label untuk single image/folder paksa")
    photo_extract.add_argument("--split", default="train", choices=["train", "val", "test"])
    photo_extract.add_argument("--profile", "--mode", dest="profile", default="fast10", choices=LIVE_PROFILE_CHOICES)
    photo_extract.add_argument("--workers", type=int, default=1)
    photo_extract.add_argument("--frames", type=int, default=10, help="Jumlah frame duplikat per foto")
    photo_extract.add_argument("--dataset-dir", default=str(DATASET_DIR))
    photo_extract.add_argument("--session-root", default=str(ROOT_DIR / "tmp" / "photo_extract"))
    photo_extract.add_argument("--overwrite-existing", action="store_true", help="Tetap extract staged entry walau video_id sudah ada")
    photo_extract.set_defaults(func=cmd_photo)

    photo_gif = photo_sub.add_parser("gif", help="Generate GIF temp untuk entry staged")
    photo_gif.add_argument("--session", required=True)
    photo_gif.add_argument("--entry-id", action="append", default=None)
    photo_gif.add_argument("--fps", type=float, default=10.0)
    photo_gif.add_argument("--width", type=int, default=420)
    photo_gif.add_argument("--height", type=int, default=420)
    photo_gif.add_argument("--draw-face", default="auto", choices=["auto", "off", "sparse", "mesh"])
    photo_gif.add_argument("--force", action="store_true")
    photo_gif.set_defaults(func=cmd_photo)

    photo_commit = photo_sub.add_parser("commit", help="Commit entry staged yang diterima ke parquet/assets")
    photo_commit.add_argument("--session", required=True)
    photo_commit.add_argument("--entry-id", action="append", default=None, help="Entry id yang diterima; bisa diulang")
    photo_commit.add_argument("--accept-file", default=None, help="JSON list entry_id dari GUI review")
    photo_commit.add_argument("--dataset-dir", default=str(DATASET_DIR))
    photo_commit.add_argument("--backup-root", default=str(BACKUP_ROOT))
    photo_commit.add_argument("--gif-dir", default=str(GIF_DIR))
    photo_commit.add_argument("--dry-run", action="store_true")
    photo_commit.add_argument("--overwrite-existing", action="store_true", help="Backup lalu replace rows dengan video_id yang sama")
    photo_commit.set_defaults(func=cmd_photo)

    gif = sub.add_parser("gif", help="Kelola GIF preview")
    gif_sub = gif.add_subparsers(dest="gif_command", required=True)
    gif_list = gif_sub.add_parser("list", help="List GIF")
    gif_list.add_argument("--schema", default="all", choices=SCHEMA_CHOICES)
    gif_list.add_argument("--vocab", default=None)
    gif_list.add_argument("--gif-dir", default=str(GIF_DIR))
    gif_list.set_defaults(func=cmd_gif)

    gif_vocab = gif_sub.add_parser("vocab", help="List vocab dataset yang bisa dibuat GIF, lengkap train/val/test")
    gif_vocab.add_argument("--schema", default="all", choices=SCHEMA_CHOICES)
    gif_vocab.add_argument("--dataset-dir", default=str(DATASET_DIR))
    gif_vocab.add_argument("--plain", action="store_true", help="Output TSV agar mudah dibaca GUI/script")
    gif_vocab.set_defaults(func=cmd_gif)

    gif_dataset = gif_sub.add_parser("dataset", help="Generate GIF ringan dari feature parquet dataset, tanpa video asli")
    gif_dataset.add_argument("--schema", default="all", choices=SCHEMA_CHOICES)
    gif_dataset.add_argument("--vocab", default=None, help="Vocab/label yang mau dibuat GIF")
    gif_dataset.add_argument("--all-vocab", action="store_true", help="Generate semua vocab yang ada di dataset")
    gif_dataset.add_argument("--split", default="all", choices=["all", "train", "val", "test"])
    gif_dataset.add_argument("--limit", type=int, default=5, help="Maks sample per split per vocab; 0 = semua")
    gif_dataset.add_argument("--fps", type=float, default=10.0)
    gif_dataset.add_argument("--width", type=int, default=420)
    gif_dataset.add_argument("--height", type=int, default=420)
    gif_dataset.add_argument("--max-frames", type=int, default=80, help="Batas frame per GIF agar ringan; 0 = semua")
    gif_dataset.add_argument("--draw-face", default="auto", choices=["auto", "off", "sparse", "mesh"], help="Gambar landmark wajah di GIF untuk schema yang punya wajah")
    gif_dataset.add_argument("--dataset-dir", default=str(DATASET_DIR))
    gif_dataset.add_argument("--gif-dir", default=str(GIF_DIR))
    gif_dataset.add_argument("--force", action="store_true", help="Timpa GIF yang sudah ada")
    gif_dataset.set_defaults(func=cmd_gif)

    gif_check = gif_sub.add_parser("check", help="Cek sample GIF vocab/video_id")
    gif_check.add_argument("--schema", default=fs.DEFAULT_SCHEMA, choices=list(fs.SCHEMA_NAMES))
    gif_check.add_argument("--vocab", required=True)
    gif_check.add_argument("--video-id", required=True)
    gif_check.set_defaults(func=cmd_gif)
    gif_make = gif_sub.add_parser("make", help="Generate GIF dari video")
    gif_make.add_argument("--schema", default=fs.DEFAULT_SCHEMA, choices=list(fs.SCHEMA_NAMES))
    gif_make.add_argument("--video", required=True)
    gif_make.add_argument("--label", required=True)
    gif_make.add_argument("--video-id", default=None)
    gif_make.add_argument("--out-dir", default=None)
    gif_make.set_defaults(func=cmd_gif)


    sample = sub.add_parser("sample", help="List/hapus sample dataset tertentu tanpa mengubah recorder/live inference")
    sample_sub = sample.add_subparsers(dest="sample_command", required=True)

    sample_list = sample_sub.add_parser("list", help="List sample dari vocab/split/schema")
    sample_list.add_argument("--schema", default="all", choices=SCHEMA_CHOICES)
    sample_list.add_argument("--vocab", default=None, help="Filter vocab/label")
    sample_list.add_argument("--split", default="all", choices=["all", "train", "val", "test"])
    sample_list.add_argument("--dataset-dir", default=str(DATASET_DIR))
    sample_list.add_argument("--plain", action="store_true", help="Output TSV untuk GUI/script")
    sample_list.set_defaults(func=cmd_sample)

    sample_delete = sample_sub.add_parser("delete", help="Hapus satu sample dari dataset semua schema + hapus GIF/archive terkait")
    sample_delete.add_argument("--schema", default="all", choices=SCHEMA_CHOICES)
    sample_delete.add_argument("--vocab", required=True, help="Vocab/label sample")
    sample_delete.add_argument("--split", default="all", choices=["all", "train", "val", "test"])
    sample_delete.add_argument("--video-id", required=True, help="video_id sample yang mau dihapus")
    sample_delete.add_argument("--dataset-dir", default=str(DATASET_DIR))
    sample_delete.add_argument("--backup-root", default=str(BACKUP_ROOT))
    sample_delete.add_argument("--gif-dir", default=str(GIF_DIR))
    sample_delete.add_argument("--full-root", default=str(DATASET_DIR / "full_features"))
    sample_delete.add_argument("--dry-run", action="store_true", help="Cek dulu tanpa menghapus")
    sample_delete.set_defaults(func=cmd_sample)

    augment = sub.add_parser("augment", help="Augmentasi dataset schema, feature-only, tidak menyentuh video")
    augment.add_argument(
        "--schema",
        action="append",
        default=None,
        metavar="SCHEMA",
        help="Schema target; bisa comma-list atau diulang. Default: smart180",
    )
    augment.add_argument("--dataset-dir", default=str(DATASET_DIR))
    augment.add_argument("--backup-root", default=str(BACKUP_ROOT))
    augment.add_argument("--split", default="train", help="train, val, test, all, atau comma list")
    augment.add_argument("--vocab", action="append", default=None, help="Opsional: vocab/label; bisa diulang")
    augment.add_argument("--all-vocab", action="store_true", help="Augment semua vocab yang ada di dataset")
    augment.add_argument("--target-per-class", type=int, default=0, help="Tambahkan sampai total sample per vocab/split mencapai target ini; 0 = nonaktif")
    augment.add_argument("--copies-per-sample", type=int, default=2, help="Jumlah augmentasi baru per sample sumber asli")
    augment.add_argument("--min-source-samples", type=int, default=5, help="Minimal sample asli per vocab/split sebelum boleh diaugmentasi")
    augment.add_argument("--intensity", type=float, default=1.0, help="Kekuatan jitter 0.0-2.0; default 1.0")
    augment.add_argument("--include-idle", action="store_true")
    augment.add_argument("--include-augmented-source", action="store_true", help="Izinkan hasil augmentasi lama dipakai lagi sebagai sumber")
    augment.add_argument("--seed", type=int, default=None)
    augment.add_argument("--overwrite-existing", action="store_true", help="Diterima untuk konsistensi; augment normal tetap append ID baru")
    augment.set_defaults(func=cmd_augment)

    augment_delete = sub.add_parser("augment-delete", help="Hapus hasil augmentasi tanpa menyentuh sample asli")
    augment_delete.add_argument(
        "--schema",
        action="append",
        default=None,
        metavar="SCHEMA",
        help="Schema target; bisa comma-list atau diulang. Default: all",
    )
    augment_delete.add_argument("--dataset-dir", default=str(DATASET_DIR))
    augment_delete.add_argument("--backup-root", default=str(BACKUP_ROOT))
    augment_delete.add_argument("--split", default="all", help="train, val, test, all, atau comma list")
    augment_delete.add_argument("--vocab", action="append", default=None, help="Opsional: vocab/label; bisa diulang")
    augment_delete.add_argument("--all-vocab", action="store_true", help="Hapus augmentasi untuk semua vocab")
    augment_delete.add_argument("--include-idle", action="store_true")
    augment_delete.add_argument("--dry-run", action="store_true", help="Cek jumlah yang akan dihapus tanpa menulis parquet")
    augment_delete.set_defaults(func=cmd_augment_delete)

    gui = sub.add_parser("gui", help="Buka Tkinter GUI")
    gui.set_defaults(func=cmd_gui)
    return parser


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if not argv:
        if sys.stdin.isatty():
            return menu_loop()
        parser = build_arg_parser()
        parser.print_help()
        return 0
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 1
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
