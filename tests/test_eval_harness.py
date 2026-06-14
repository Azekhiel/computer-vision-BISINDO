import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import eval_harness as eh
import feature_schemas as fs
import gru_manager as gm
from smart_extract import contract as sc


def _write_tiny_smart180(root: Path) -> None:
    schema_dir = root / "smart180"
    schema_dir.mkdir(parents=True)
    rows = []
    for label_idx, label in enumerate(["aku", "kamu"]):
        for split in ["train", "val", "test"]:
            for vid in range(2):
                for frame in range(5):
                    rows.append(
                        {
                            "video_id": f"{label}_{split}_{vid}",
                            "label": label,
                            "frame_num": frame,
                            "split": split,
                            "feature_version": sc.FEATURE_SCHEMA,
                            "feature_dim": sc.FEATURE_DIM,
                            "features": sc.format_feature_value(
                                np.full(sc.FEATURE_DIM, label_idx + frame / 10.0, dtype=np.float32)
                            ),
                        }
                    )
    pd.DataFrame(rows).to_parquet(schema_dir / "tiny.parquet", index=False)


def test_expand_schema_and_variant_requests():
    schemas = eh.expand_schema_request(["smart180", "faceref"])
    assert schemas[0] == "smart180"
    for name in fs.FACE_REF_SCHEMA_NAMES:
        assert name in schemas

    variants = eh.expand_variant_request(["adi", "base"])
    assert "adi" in variants
    for name in gm.BASE_VARIANT_NAMES:
        assert name in variants
    # de-duplicated
    assert len(variants) == len(set(variants))


def test_confusion_matrix_counts_and_labels():
    labels, matrix = eh.confusion_matrix(["a", "a", "b"], ["a", "b", "b"])
    assert labels == ["a", "b"]
    assert matrix.tolist() == [[1, 1], [0, 1]]


def test_run_matrix_trains_evaluates_and_writes_reports(tmp_path):
    data_dir = tmp_path / "data"
    model_dir = tmp_path / "models"
    out_root = tmp_path / "reports"
    _write_tiny_smart180(data_dir)

    summary = eh.run_matrix(
        schemas=["smart180"],
        variants=["adi", "biattn"],
        dataset_dir=data_dir,
        model_dir=model_dir,
        out_root=out_root,
        train_missing=True,
        epochs=1,
        device="cpu",
        benchmark=False,
    )

    out_dir = Path(summary["out_dir"])
    assert (out_dir / "summary.csv").exists()
    assert (out_dir / "summary.md").exists()
    assert (out_dir / "raw_results.json").exists()
    assert (out_dir / "confusion_smart180__adi.csv").exists()
    assert (out_dir / "confusion_smart180__biattn.csv").exists()

    results = summary["results"]
    assert len(results) == 2
    by_variant = {row["variant"]: row for row in results}
    assert set(by_variant) == {"adi", "biattn"}
    for row in results:
        assert row["status"] == "ok"
        assert 0.0 <= float(row["f1_macro"]) <= 1.0
        assert int(row["samples"]) > 0


def test_run_matrix_marks_missing_checkpoint_without_training(tmp_path):
    data_dir = tmp_path / "data"
    model_dir = tmp_path / "models"
    out_root = tmp_path / "reports"
    _write_tiny_smart180(data_dir)

    summary = eh.run_matrix(
        schemas=["smart180"],
        variants=["tcn"],
        dataset_dir=data_dir,
        model_dir=model_dir,
        out_root=out_root,
        train_missing=False,
        benchmark=False,
    )
    row = summary["results"][0]
    assert row["status"] == "missing_checkpoint"
