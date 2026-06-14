#!/usr/bin/env python3
"""Rebuild dataset_parquets with Smart Extract V8 best-mode 180-D features."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from smart_extract import contract as sc  # noqa: E402
from smart_extract.extract_video_smart_v8 import extract_video_arrays, make_best_extract_args  # noqa: E402

VIDEO_EXTS = {".mkv", ".mp4", ".avi", ".mov", ".webm", ".m4v"}


def _scan_videos(record_video_dir: Path) -> list[tuple[Path, str, str]]:
    items: list[tuple[Path, str, str]] = []
    for split_dir in sorted(record_video_dir.iterdir()) if record_video_dir.exists() else []:
        if not split_dir.is_dir() or split_dir.name not in {"train", "val", "test"}:
            continue
        for vocab_dir in sorted(split_dir.iterdir()):
            if not vocab_dir.is_dir():
                continue
            for video_path in sorted(vocab_dir.iterdir()):
                if video_path.suffix.lower() in VIDEO_EXTS:
                    items.append((video_path, vocab_dir.name, split_dir.name))
    return items


def _rows_for_video(video_path: Path, vocab: str, split: str, quiet: bool) -> list[dict]:
    args = make_best_extract_args(quiet=quiet, save_gif=False, no_gif=True)
    result = extract_video_arrays(video_path, args=args, include_frames=False)
    if result is None:
        raise RuntimeError(f"extract failed: {video_path}")
    features = sc.ensure_feature_dim(result["features"])
    frames = list(result.get("frames", []))
    if len(features) == 0:
        raise RuntimeError(f"empty feature sequence: {video_path}")

    video_id = f"{split}_manual_{video_path.name}"
    rows = []
    for frame_num, vec in enumerate(features):
        meta = frames[frame_num] if frame_num < len(frames) else {}
        rows.append(
            {
                "video_id": video_id,
                "label": vocab,
                "frame_num": int(frame_num),
                "split": split,
                "feature_version": sc.FEATURE_SCHEMA,
                "feature_mode": sc.FEATURE_MODE,
                "feature_dim": sc.FEATURE_DIM,
                "target_fps": sc.TARGET_FPS,
                "extract_profile": sc.EXTRACT_PROFILE,
                "source_media_path": str(video_path),
                "source_frame_num": int(meta.get("target_source_frame", frame_num)),
                "chosen_source_frame": int(meta.get("chosen_source_frame", meta.get("target_source_frame", frame_num))),
                "time_sec": float(meta.get("time_sec", frame_num / sc.TARGET_FPS)),
                "smart_mode": "best",
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


def _validate_outputs(rows_by_vocab: dict[str, list[dict]]) -> None:
    if not rows_by_vocab:
        raise RuntimeError("no rows generated")
    empty = [vocab for vocab, rows in rows_by_vocab.items() if not rows]
    if empty:
        raise RuntimeError(f"empty vocab outputs: {empty}")
    for vocab, rows in rows_by_vocab.items():
        for row in rows:
            if row["feature_version"] != sc.FEATURE_SCHEMA:
                raise RuntimeError(f"{vocab}: wrong schema")
            vec = sc.parse_feature_value(row["features"])
            if vec.shape[0] != sc.FEATURE_DIM or not np.isfinite(vec).all():
                raise RuntimeError(f"{vocab}: invalid feature vector")


def migrate(record_video_dir: Path, parquet_dir: Path, backup_root: Path, quiet: bool = False) -> Path:
    videos = _scan_videos(record_video_dir)
    if not videos:
        raise RuntimeError(f"no videos found under {record_video_dir}")

    rows_by_vocab: dict[str, list[dict]] = defaultdict(list)
    failures = []
    for idx, (video_path, vocab, split) in enumerate(videos, 1):
        print(f"[{idx}/{len(videos)}] {split}/{vocab}/{video_path.name}")
        try:
            rows_by_vocab[vocab].extend(_rows_for_video(video_path, vocab, split, quiet=quiet))
        except Exception as exc:
            failures.append(f"{video_path}: {type(exc).__name__}: {exc}")

    if failures:
        raise RuntimeError("migration failed before replace:\n" + "\n".join(failures[:20]))

    _validate_outputs(rows_by_vocab)

    ts = time.strftime("%Y%m%d_%H%M%S")
    backup_dir = backup_root / f"dataset_parquets_backup_{ts}"
    backup_root.mkdir(parents=True, exist_ok=True)
    if parquet_dir.exists():
        shutil.copytree(parquet_dir, backup_dir)
    else:
        backup_dir.mkdir(parents=True, exist_ok=True)

    tmp_dir = parquet_dir.with_name(f"{parquet_dir.name}.smart_v8_tmp_{ts}")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)

    for vocab, rows in rows_by_vocab.items():
        df = pd.DataFrame(rows)
        df.to_parquet(tmp_dir / f"{vocab}.parquet", index=False)

    if parquet_dir.exists():
        shutil.rmtree(parquet_dir)
    tmp_dir.rename(parquet_dir)
    return backup_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate dataset parquet files to Smart Extract V8 best-mode 180-D")
    parser.add_argument("--record-video-dir", default=str(ROOT_DIR / "record" / "video"))
    parser.add_argument("--parquet-dir", default=str(ROOT_DIR / "dataset_parquets"))
    parser.add_argument("--backup-root", default=str(ROOT_DIR / "backups"))
    parser.add_argument("--quiet-extract", action="store_true")
    args = parser.parse_args()

    backup = migrate(
        record_video_dir=Path(args.record_video_dir),
        parquet_dir=Path(args.parquet_dir),
        backup_root=Path(args.backup_root),
        quiet=bool(args.quiet_extract),
    )
    print(f"[DONE] parquet migrated to {sc.FEATURE_SCHEMA}; backup: {backup}")


if __name__ == "__main__":
    main()
