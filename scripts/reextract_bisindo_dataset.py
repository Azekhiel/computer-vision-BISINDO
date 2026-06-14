#!/usr/bin/env python3
"""Re-extract BISINDO videos into the current feature schema.

This script preserves older schema rows and replaces only rows matching the
current feature schema for selected vocabularies.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import data_ingestion as di  # noqa: E402
import database_manager as dbm  # noqa: E402
import feature_engine as fe  # noqa: E402

MEDIA_EXTS = (".mkv", ".mp4", ".avi", ".mov", ".webm", ".jpg", ".jpeg", ".png", ".gif")


def _discover_vocab_dirs(dataset_root: Path, splits: list[str], limit_vocabs: set[str] | None) -> list[Path]:
    paths: list[Path] = []
    for split in splits:
        split_dir = dataset_root / split
        if not split_dir.exists():
            continue
        for vocab_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            if limit_vocabs and vocab_dir.name not in limit_vocabs:
                continue
            if any(p.is_file() and p.suffix.lower() in MEDIA_EXTS for p in vocab_dir.iterdir()):
                paths.append(vocab_dir)
    return paths


def _summarize(paths: list[Path]) -> dict:
    by_split: dict[str, dict[str, int]] = {}
    for path in paths:
        split = path.parent.name
        vocab = path.name
        media_count = sum(1 for p in path.iterdir() if p.is_file() and p.suffix.lower() in MEDIA_EXTS)
        by_split.setdefault(split, {})[vocab] = media_count
    return {
        "schema": fe.FEATURE_SCHEMA,
        "vocab_dirs": len(paths),
        "splits": by_split,
        "total_media": int(sum(sum(v.values()) for v in by_split.values())),
    }


def _clear_current_schema_rows(vocabs: list[str]) -> dict[str, int]:
    cleared: dict[str, int] = {}
    db_dir = Path(dbm.DATABASE_DIR)
    db_dir.mkdir(parents=True, exist_ok=True)
    for vocab in sorted(set(vocabs)):
        path = db_dir / f"{vocab}.parquet"
        if not path.exists():
            cleared[vocab] = 0
            continue
        df = pd.read_parquet(path)
        if "feature_version" not in df.columns:
            cleared[vocab] = 0
            continue
        keep = df[df["feature_version"] != fe.FEATURE_SCHEMA].copy()
        removed = int(len(df) - len(keep))
        if removed:
            keep.to_parquet(path, index=False)
        cleared[vocab] = removed
    return cleared


def main() -> int:
    parser = argparse.ArgumentParser(description="Re-extract BISINDO videos to the current Jetson-safe schema.")
    parser.add_argument("--dataset-root", default=str(ROOT_DIR / "record" / "video"))
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"], choices=["train", "val", "test"])
    parser.add_argument("--limit-vocabs", nargs="*", default=None, help="Only rebuild these vocab folder names.")
    parser.add_argument("--profile", default="offline_accuracy", choices=sorted(fe.EXTRACTION_PROFILES))
    parser.add_argument("--workers", type=int, default=1, help="Use 1 on Jetson when YOLO ROI is enabled.")
    parser.add_argument("--mp-device", choices=["CPU", "GPU"], default="GPU")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    limit_vocabs = {v.strip() for v in args.limit_vocabs if v.strip()} if args.limit_vocabs else None
    paths = _discover_vocab_dirs(dataset_root, args.splits, limit_vocabs)
    summary = _summarize(paths)
    summary["profile"] = args.profile
    summary["workers"] = int(args.workers)
    summary["mp_device"] = args.mp_device

    if args.dry_run:
        print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))
        return 0

    if not paths:
        print(json.dumps({**summary, "ok": False, "message": "No vocab media folders found."}, indent=2))
        return 1

    os.environ["BISINDO_EXTRACTION_PROFILE"] = args.profile
    di.MAX_WORKERS = max(1, int(args.workers))
    vocabs = sorted({p.name for p in paths})
    cleared = _clear_current_schema_rows(vocabs)
    ok, message = di.bulk_import([str(p) for p in paths], default_split="train", mp_device=args.mp_device)
    dbm.update_metadata("db_update")
    print(json.dumps({**summary, "cleared_current_rows": cleared, "ok": bool(ok), "message": message}, indent=2, ensure_ascii=False))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
