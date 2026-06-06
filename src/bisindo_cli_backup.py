"""Terminal-first BISINDO workflow CLI.

This module intentionally does not import Tkinter or the live camera stack at
module import time. Heavy pieces are loaded only by the commands that need them.
"""

from __future__ import annotations

import argparse
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
LIVE_PROFILE_CHOICES = ["accurate10", "fast10", "jetson10", "lite", "ultra", "fast", "quality"]
SCHEMA_CHOICES = [*fs.SCHEMA_NAMES, "all"]


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


def _backup_existing(parquet_path: Path, backup_dir: Path, copied: set[Path]) -> None:
    if not parquet_path.exists() or parquet_path in copied:
        return
    backup_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(parquet_path, backup_dir / parquet_path.name)
    copied.add(parquet_path)


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
) -> ImportResult:
    """Append schema rows safely, skipping duplicate video_id per vocab."""

    schema_spec = fs.get_schema(schema)
    dataset_root = fs.dataset_dir_for(schema_spec, dataset_dir)
    dataset_root.mkdir(parents=True, exist_ok=True)
    backup_dir = Path(backup_root) / f"bisindo_cli_import_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    copied: set[Path] = set()
    rows_by_label: dict[str, list[dict]] = {}
    gif_paths: list[Path] = []
    result = ImportResult(scanned=len(items), gif_paths=gif_paths)

    for idx, item in enumerate(items, 1):
        parquet_path = dataset_root / f"{item.label}.parquet"
        if item.video_id in _existing_video_ids(parquet_path):
            result.skipped += 1
            if not quiet:
                print(f"[SKIP] {idx}/{len(items)} {item.label}/{item.video_id}")
            continue
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
            df_out = pd.concat([df_old, df_new], ignore_index=True)
        else:
            df_out = df_new
        df_out.to_parquet(parquet_path, index=False)

    result.backup_dir = backup_dir if copied else None
    return result


def _augment_sequence(sequence: np.ndarray, rng: np.random.Generator, schema: str = fs.DEFAULT_SCHEMA) -> np.ndarray:
    schema_spec = fs.get_schema(schema)
    seq = fs.ensure_feature_dim(sequence, schema_spec).copy()
    if len(seq) >= 4 and rng.random() < 0.45:
        scale = float(rng.uniform(0.88, 1.15))
        new_len = max(3, int(round(len(seq) * scale)))
        old_x = np.linspace(0.0, 1.0, num=len(seq), dtype=np.float32)
        new_x = np.linspace(0.0, 1.0, num=new_len, dtype=np.float32)
        out = np.empty((new_len, schema_spec.feature_dim), dtype=np.float32)
        if schema_spec.name == "smart180":
            for col in range(sc.SLICE_META.start):
                out[:, col] = np.interp(new_x, old_x, seq[:, col]).astype(np.float32)
            nearest = np.clip(np.rint(new_x * (len(seq) - 1)).astype(int), 0, len(seq) - 1)
            out[:, sc.SLICE_META] = seq[nearest, sc.SLICE_META]
        else:
            for col in range(schema_spec.feature_dim):
                out[:, col] = np.interp(new_x, old_x, seq[:, col]).astype(np.float32)
        seq = out

    if schema_spec.name == "smart180" and rng.random() < 0.80:
        seq[:, 0 : sc.SLICE_META.start] += rng.normal(0.0, 0.006, size=(len(seq), sc.SLICE_META.start)).astype(np.float32)
    elif schema_spec.name != "smart180" and rng.random() < 0.80:
        seq += rng.normal(0.0, 0.003, size=seq.shape).astype(np.float32)
    if schema_spec.name == "smart180" and rng.random() < 0.55:
        seq[:, sc.SLICE_LEFT_ANGLES] += rng.normal(0.0, 0.02, size=(len(seq), sc.SLICE_LEFT_ANGLES.stop - sc.SLICE_LEFT_ANGLES.start)).astype(np.float32)
        seq[:, sc.SLICE_RIGHT_ANGLES] += rng.normal(0.0, 0.02, size=(len(seq), sc.SLICE_RIGHT_ANGLES.stop - sc.SLICE_RIGHT_ANGLES.start)).astype(np.float32)
    if len(seq) >= 6 and rng.random() < 0.25:
        keep = rng.random(len(seq)) > 0.06
        if keep.sum() >= 3:
            seq = seq[keep]
    return fs.ensure_feature_dim(seq, schema_spec)


def augment_dataset(
    dataset_dir: str | Path = DATASET_DIR,
    backup_root: str | Path = BACKUP_ROOT,
    schema: str = fs.DEFAULT_SCHEMA,
    split: str = "train",
    target_per_class: int = 200,
    include_idle: bool = False,
    seed: int | None = None,
) -> tuple[int, Path | None]:
    split = str(split or "train").lower()
    if split not in SPLITS:
        raise ValueError("split harus salah satu: train, val, test")
    schema_spec = fs.get_schema(schema)
    rng = np.random.default_rng(seed)
    backup_dir = Path(backup_root) / f"bisindo_cli_augment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    copied: set[Path] = set()
    total_generated = 0

    for parquet_path in fs.dataset_parquet_paths(schema_spec, dataset_dir):
        label = parquet_path.stem
        if not include_idle and label.lower() == "idle":
            continue
        df = pd.read_parquet(parquet_path)
        current = fs.filter_feature_rows(df, schema_spec)
        current = current[current["split"].astype(str).str.lower() == split]
        if current.empty:
            continue
        groups = [(str(video_id), group.sort_values("frame_num")) for video_id, group in current.groupby("video_id", sort=False)]
        existing = len(groups)
        needed = max(0, int(target_per_class) - existing)
        if needed <= 0:
            continue

        new_rows: list[dict] = []
        for aug_idx in range(needed):
            base_video_id, group = groups[int(rng.integers(0, len(groups)))]
            base_seq = fs.ensure_feature_dim([sc.parse_feature_value(value) for value in group["features"].tolist()], schema_spec)
            aug_seq = _augment_sequence(base_seq, rng, schema=schema_spec.name)
            new_video_id = f"{split}_generate_{label}_{datetime.now().strftime('%H%M%S')}_{aug_idx:04d}"
            for frame_num, vec in enumerate(aug_seq):
                meta = vec[sc.SLICE_META]
                presence = fs.presence_from_vector(schema_spec, vec)
                new_rows.append(
                    {
                        "video_id": new_video_id,
                        "label": label,
                        "frame_num": int(frame_num),
                        "split": split,
                        "schema": schema_spec.name,
                        "feature_version": schema_spec.feature_schema,
                        "feature_mode": schema_spec.feature_mode,
                        "feature_dim": schema_spec.feature_dim,
                        "target_fps": schema_spec.target_fps,
                        "extract_profile": "augment",
                        "source_media_path": "",
                        "source_frame_num": int(frame_num),
                        "chosen_source_frame": int(frame_num),
                        "time_sec": float(frame_num / schema_spec.target_fps),
                        "smart_mode": "augment",
                        "enhance_mode": "augment",
                        "left_present": float(presence["left_present"]),
                        "right_present": float(presence["right_present"]),
                        "left_detected": float(presence["left_present"]),
                        "right_detected": float(presence["right_present"]),
                        "left_held": float(meta[sc.IDX_META_LEFT_HELD]) if schema_spec.name == "smart180" else 0.0,
                        "right_held": float(meta[sc.IDX_META_RIGHT_HELD]) if schema_spec.name == "smart180" else 0.0,
                        "left_score": float(meta[sc.IDX_META_LEFT_SCORE]) if schema_spec.name == "smart180" else float(presence["left_present"]),
                        "right_score": float(meta[sc.IDX_META_RIGHT_SCORE]) if schema_spec.name == "smart180" else float(presence["right_present"]),
                        "augmented_from": base_video_id,
                        "features": sc.format_feature_value(vec),
                    }
                )
            total_generated += 1

        if new_rows:
            _backup_existing(parquet_path, backup_dir, copied)
            df_out = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
            df_out.to_parquet(parquet_path, index=False)
            print(f"[AUG] {label}: +{len({row['video_id'] for row in new_rows})} sequences")

    return total_generated, backup_dir if copied else None


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
    return gm.main(
        [
            "train",
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
            *([] if args.epochs is None else ["--epochs", str(args.epochs)]),
            *([] if args.batch_size is None else ["--batch-size", str(args.batch_size)]),
            *([] if args.lr is None else ["--lr", str(args.lr)]),
            *([] if args.patience is None else ["--patience", str(args.patience)]),
            *([] if args.limit_per_class is None else ["--limit-per-class", str(args.limit_per_class)]),
        ]
    )


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
    for schema_name in fs.expand_schema_names(args.schema):
        result = append_import_items(
            items,
            dataset_dir=args.dataset_dir,
            backup_root=args.backup_root,
            schema=schema_name,
            save_gif=args.save_gif,
            quiet=args.quiet,
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


def cmd_gif(args: argparse.Namespace) -> int:
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


def cmd_augment(args: argparse.Namespace) -> int:
    for schema_name in fs.expand_schema_names(args.schema):
        generated, backup_dir = augment_dataset(
            dataset_dir=args.dataset_dir,
            backup_root=args.backup_root,
            schema=schema_name,
            split=args.split,
            target_per_class=args.target_per_class,
            include_idle=args.include_idle,
            seed=args.seed,
        )
        print(f"augment[{schema_name}]: generated={generated}")
        if backup_dir:
            print(f"backup[{schema_name}]: {backup_dir}")
    return 0


def _format_live_status(item: dict) -> str:
    top = item.get("top") or []
    top_text = ""
    if top:
        top_text = " top=" + "/".join(f"{label}:{float(score):.2f}" for label, score in top[:3])
    return (
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
                print(
                    f"started: schema={item.get('schema', args.schema)} variant={item.get('variant')} profile={item.get('profile')} "
                    f"device={item.get('device')} reason={item.get('device_reason', '-')}"
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
                print(f"stopped: {item.get('message')}")
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
        "8": ["record"],
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
        print("8. Record dataset (butuh argumen --label kalau dipakai langsung)")
        print("q. Keluar")
        choice = input("Pilih: ").strip().lower()
        if choice in {"q", "quit", "exit"}:
            return 0
        if choice == "8":
            label = input("Label vocab: ").strip()
            argv = ["record", "--label", label] if label else None
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
    live.add_argument("--variant", default="auto", choices=["auto", *gm.VARIANT_NAMES, *(f"gru_{v}" for v in gm.VARIANT_NAMES)])
    live.add_argument("--profile", "--mode", dest="profile", default="accurate10", choices=LIVE_PROFILE_CHOICES)
    live.add_argument("--segment-mode", default="auto", choices=["auto", "rolling"])
    live.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    live.add_argument("--camera", type=int, default=0)
    live.add_argument("--threshold", type=float, default=0.65)
    live.add_argument("--window", action="store_true", help="Buka overlay kamera OpenCV")
    live.add_argument("--duration", type=float, default=0.0, help="Stop otomatis setelah N detik; 0 = jalan terus")
    live.add_argument("--print-interval", type=float, default=0.5)
    live.add_argument("--no-jit", action="store_true")
    live.set_defaults(func=cmd_live)

    diagnose = sub.add_parser("diagnose-live", help="Capture satu gesture live, simpan NPZ/GIF, dan prediksi top-3")
    diagnose.add_argument("--schema", default=fs.DEFAULT_SCHEMA, choices=list(fs.SCHEMA_NAMES))
    diagnose.add_argument("--variant", default="auto", choices=["auto", *gm.VARIANT_NAMES, *(f"gru_{v}" for v in gm.VARIANT_NAMES)])
    diagnose.add_argument("--profile", "--mode", dest="profile", default="accurate10", choices=LIVE_PROFILE_CHOICES)
    diagnose.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    diagnose.add_argument("--camera", type=int, default=0)
    diagnose.add_argument("--timeout", type=float, default=12.0)
    diagnose.add_argument("--out-dir", default=None)
    diagnose.add_argument("--window", action="store_true")
    diagnose.add_argument("--no-jit", action="store_true")
    diagnose.set_defaults(func=cmd_live_diagnose)

    train = sub.add_parser("train", parents=[common_data], help="Train GRU")
    train.add_argument("--schema", default=fs.DEFAULT_SCHEMA, choices=SCHEMA_CHOICES)
    train.add_argument("--variant", default="all", choices=[*gm.VARIANT_NAMES, "all"])
    train.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    train.add_argument("--epochs", type=int, default=None)
    train.add_argument("--batch-size", type=int, default=None)
    train.add_argument("--lr", type=float, default=None)
    train.add_argument("--patience", type=int, default=None)
    train.add_argument("--limit-per-class", type=int, default=None)
    train.set_defaults(func=cmd_train)

    eval_cmd = sub.add_parser("eval", parents=[common_data], help="Evaluasi checkpoint GRU")
    eval_cmd.add_argument("--schema", default=fs.DEFAULT_SCHEMA, choices=SCHEMA_CHOICES)
    eval_cmd.add_argument("--variant", default="all", choices=[*gm.VARIANT_NAMES, "all"])
    eval_cmd.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    eval_cmd.add_argument("--split", default="test", choices=["train", "val", "test"])
    eval_cmd.set_defaults(func=cmd_eval)

    bench = sub.add_parser("benchmark", parents=[common_data], help="Benchmark latency GRU")
    bench.add_argument("--schema", default=fs.DEFAULT_SCHEMA, choices=SCHEMA_CHOICES)
    bench.add_argument("--variant", default="all", choices=[*gm.VARIANT_NAMES, "all"])
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
    import_cmd.set_defaults(func=cmd_import)

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
    record.set_defaults(func=cmd_record)

    gif = sub.add_parser("gif", help="Kelola GIF preview")
    gif_sub = gif.add_subparsers(dest="gif_command", required=True)
    gif_list = gif_sub.add_parser("list", help="List GIF")
    gif_list.add_argument("--schema", default="all", choices=SCHEMA_CHOICES)
    gif_list.add_argument("--vocab", default=None)
    gif_list.add_argument("--gif-dir", default=str(GIF_DIR))
    gif_list.set_defaults(func=cmd_gif)
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

    augment = sub.add_parser("augment", help="Augmentasi dataset schema")
    augment.add_argument("--schema", default=fs.DEFAULT_SCHEMA, choices=SCHEMA_CHOICES)
    augment.add_argument("--dataset-dir", default=str(DATASET_DIR))
    augment.add_argument("--backup-root", default=str(BACKUP_ROOT))
    augment.add_argument("--split", default="train", choices=["train", "val", "test"])
    augment.add_argument("--target-per-class", type=int, default=200)
    augment.add_argument("--include-idle", action="store_true")
    augment.add_argument("--seed", type=int, default=None)
    augment.set_defaults(func=cmd_augment)

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
