"""Training, loading, and evaluation utilities for GRU BISINDO models."""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import time
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import gru_adi
import gru_hybrid
import gru_khukuh
import gru_biattn
import gru_convfront
import tcn_sign
import transformer_sign
import feature_schemas as fs
import jetson_runtime as jr
from smart_extract import contract as sc


ROOT_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT_DIR / "dataset_parquets"
MODEL_DIR = ROOT_DIR / "models"
BACKUP_ROOT = ROOT_DIR / "backups"
GRU_PREFIX = "gru_"
EXCLUDED_LABELS = {"idle"}
EVAL_SUITE_NAMES = ("main", "chunk10", "threshold", "main_chunk10", "main_threshold", "vote_all", "boosted_stack")
SPECIALIST_DEFAULT_EPOCHS = 40
SPECIALIST_DEFAULT_BATCH_SIZE = 16
SPECIALIST_DEFAULT_PATIENCE = 8


@dataclass(frozen=True)
class VariantSpec:
    name: str
    display_name: str
    module: object
    target_frames: int
    default_lr: float
    default_batch_size: int
    default_epochs: int
    default_patience: int
    default_l1: float = 0.0
    default_l2: float = 1e-5


VARIANTS: dict[str, VariantSpec] = {
    "khukuh": VariantSpec(
        name="khukuh",
        display_name="GRU Khukuh",
        module=gru_khukuh,
        target_frames=gru_khukuh.TARGET_FRAMES,
        default_lr=gru_khukuh.DEFAULT_LR,
        default_batch_size=64,
        default_epochs=gru_khukuh.DEFAULT_EPOCHS,
        default_patience=25,
        default_l1=1e-6,
        default_l2=1e-5,
    ),
    "adi": VariantSpec(
        name="adi",
        display_name="GRU Adi",
        module=gru_adi,
        target_frames=gru_adi.TARGET_FRAMES,
        default_lr=gru_adi.DEFAULT_LR,
        default_batch_size=gru_adi.DEFAULT_BATCH_SIZE,
        default_epochs=gru_adi.DEFAULT_EPOCHS,
        default_patience=gru_adi.DEFAULT_PATIENCE,
        default_l1=0.0,
        default_l2=1e-5,
    ),
    "hybrid": VariantSpec(
        name="hybrid",
        display_name="GRU Hybrid",
        module=gru_hybrid,
        target_frames=gru_hybrid.TARGET_FRAMES,
        default_lr=gru_hybrid.DEFAULT_LR,
        default_batch_size=64,
        default_epochs=gru_hybrid.DEFAULT_EPOCHS,
        default_patience=gru_hybrid.DEFAULT_PATIENCE,
        default_l1=1e-6,
        default_l2=1e-5,
    ),
    "biattn": VariantSpec(
        name="biattn",
        display_name="BiGRU + Attention",
        module=gru_biattn,
        target_frames=gru_biattn.TARGET_FRAMES,
        default_lr=gru_biattn.DEFAULT_LR,
        default_batch_size=gru_biattn.DEFAULT_BATCH_SIZE,
        default_epochs=gru_biattn.DEFAULT_EPOCHS,
        default_patience=gru_biattn.DEFAULT_PATIENCE,
        default_l1=0.0,
        default_l2=1e-5,
    ),
    "convfront": VariantSpec(
        name="convfront",
        display_name="Conv1d Front + GRU",
        module=gru_convfront,
        target_frames=gru_convfront.TARGET_FRAMES,
        default_lr=gru_convfront.DEFAULT_LR,
        default_batch_size=gru_convfront.DEFAULT_BATCH_SIZE,
        default_epochs=gru_convfront.DEFAULT_EPOCHS,
        default_patience=gru_convfront.DEFAULT_PATIENCE,
        default_l1=0.0,
        default_l2=1e-5,
    ),
    "tcn": VariantSpec(
        name="tcn",
        display_name="Temporal ConvNet (TCN)",
        module=tcn_sign,
        target_frames=tcn_sign.TARGET_FRAMES,
        default_lr=tcn_sign.DEFAULT_LR,
        default_batch_size=tcn_sign.DEFAULT_BATCH_SIZE,
        default_epochs=tcn_sign.DEFAULT_EPOCHS,
        default_patience=tcn_sign.DEFAULT_PATIENCE,
        default_l1=0.0,
        default_l2=1e-5,
    ),
    "transformer": VariantSpec(
        name="transformer",
        display_name="Mini Transformer",
        module=transformer_sign,
        target_frames=transformer_sign.TARGET_FRAMES,
        default_lr=transformer_sign.DEFAULT_LR,
        default_batch_size=transformer_sign.DEFAULT_BATCH_SIZE,
        default_epochs=transformer_sign.DEFAULT_EPOCHS,
        default_patience=transformer_sign.DEFAULT_PATIENCE,
        default_l1=0.0,
        default_l2=1e-5,
    ),
}
BASE_VARIANT_NAMES = tuple(VARIANTS.keys())
AUGMENTED_SUFFIX = "_dengan_augmentasi"
AUGMENTED_VARIANT_NAMES = tuple(f"{variant}{AUGMENTED_SUFFIX}" for variant in BASE_VARIANT_NAMES)
VARIANT_NAMES = BASE_VARIANT_NAMES + AUGMENTED_VARIANT_NAMES
TRAIN_DATA_MODES = ("original", "with_augmentation", "both")
AUGMENTATION_FILTER_MODES = ("include", "exclude", "only")


@dataclass
class SequenceSample:
    label: str
    video_id: str
    split: str
    sequence: np.ndarray
    is_augmented: bool = False


def normalize_variant_name(name: str) -> str:
    value = str(name or "").strip().lower().replace("-", "_")
    if value.startswith(GRU_PREFIX):
        value = value[len(GRU_PREFIX) :]
    aliases = {
        "khukuh_augmented": f"khukuh{AUGMENTED_SUFFIX}",
        "adi_augmented": f"adi{AUGMENTED_SUFFIX}",
        "hybrid_augmented": f"hybrid{AUGMENTED_SUFFIX}",
        "khukuh_aug": f"khukuh{AUGMENTED_SUFFIX}",
        "adi_aug": f"adi{AUGMENTED_SUFFIX}",
        "hybrid_aug": f"hybrid{AUGMENTED_SUFFIX}",
        "biattn_augmented": f"biattn{AUGMENTED_SUFFIX}",
        "biattn_aug": f"biattn{AUGMENTED_SUFFIX}",
        "bigru_attn": "biattn",
        "convfront_augmented": f"convfront{AUGMENTED_SUFFIX}",
        "convfront_aug": f"convfront{AUGMENTED_SUFFIX}",
        "conv": "convfront",
        "tcn_augmented": f"tcn{AUGMENTED_SUFFIX}",
        "tcn_aug": f"tcn{AUGMENTED_SUFFIX}",
        "transformer_augmented": f"transformer{AUGMENTED_SUFFIX}",
        "transformer_aug": f"transformer{AUGMENTED_SUFFIX}",
        "xformer": "transformer",
    }
    value = aliases.get(value, value)
    if value not in VARIANT_NAMES:
        raise ValueError(f"Unknown GRU variant '{name}'. Pilih: {', '.join(VARIANT_NAMES)}")
    return value


def base_variant_name(variant: str) -> str:
    value = normalize_variant_name(variant)
    if value.endswith(AUGMENTED_SUFFIX):
        value = value[: -len(AUGMENTED_SUFFIX)]
    if value not in VARIANTS:
        raise ValueError(f"Unknown base GRU variant '{variant}'. Pilih: {', '.join(BASE_VARIANT_NAMES)}")
    return value


def augmented_variant_name(variant: str) -> str:
    return f"{base_variant_name(variant)}{AUGMENTED_SUFFIX}"


def is_augmented_variant(variant: str) -> bool:
    return normalize_variant_name(variant).endswith(AUGMENTED_SUFFIX)


def variant_spec(variant: str) -> VariantSpec:
    return VARIANTS[base_variant_name(variant)]


def normalize_train_data_mode(value: str | None = None) -> str:
    raw = str(value or "original").strip().lower().replace("-", "_")
    aliases = {
        "ori": "original",
        "asli": "original",
        "base": "original",
        "without_augmentation": "original",
        "no_augmentation": "original",
        "with_augmentation": "with_augmentation",
        "with_aug": "with_augmentation",
        "aug": "with_augmentation",
        "augmented": "with_augmentation",
        "augmentation": "with_augmentation",
        "augmentasi": "with_augmentation",
        "dengan_augmentasi": "with_augmentation",
        "plus_augmentasi": "with_augmentation",
        "both": "both",
        "all": "both",
        "semua": "both",
    }
    mode = aliases.get(raw, raw)
    if mode not in TRAIN_DATA_MODES:
        raise ValueError(f"Unknown train data mode '{value}'. Pilih: {', '.join(TRAIN_DATA_MODES)}")
    return mode


def variant_train_data_mode(variant: str) -> str:
    return "with_augmentation" if is_augmented_variant(variant) else "original"


def expand_variant_request(variant: str | None, train_data: str | None = None) -> tuple[str, ...]:
    raw = str(variant or "all").strip().lower().replace("-", "_")
    mode = normalize_train_data_mode(train_data)
    if "," in raw:
        variants: list[str] = []
        for part in raw.split(","):
            for item in expand_variant_request(part.strip(), mode):
                if item not in variants:
                    variants.append(item)
        return tuple(variants)
    if raw == "all":
        if mode == "original":
            return BASE_VARIANT_NAMES
        if mode == "with_augmentation":
            return AUGMENTED_VARIANT_NAMES
        return VARIANT_NAMES
    normalized = normalize_variant_name(raw)
    if mode == "both":
        return (base_variant_name(normalized), augmented_variant_name(normalized))
    if mode == "with_augmentation" and not is_augmented_variant(normalized):
        return (augmented_variant_name(normalized),)
    return (normalized,)


def normalize_augmentation_filter_mode(value: str | None = None) -> str:
    raw = str(value or "include").strip().lower().replace("-", "_")
    aliases = {
        "all": "include",
        "with": "include",
        "with_augmentation": "include",
        "original": "exclude",
        "ori": "exclude",
        "asli": "exclude",
        "no_aug": "exclude",
        "no_augmentation": "exclude",
        "only_aug": "only",
        "augmented": "only",
        "augmentation": "only",
    }
    mode = aliases.get(raw, raw)
    if mode not in AUGMENTATION_FILTER_MODES:
        raise ValueError(f"Unknown augmentation filter '{value}'. Pilih: {', '.join(AUGMENTATION_FILTER_MODES)}")
    return mode


def _truthy_series(series: pd.Series) -> bool:
    if series.empty:
        return False
    text = series.fillna("").astype(str).str.strip().str.lower()
    truthy = {"1", "true", "yes", "y", "iya", "ya"}
    if text.isin(truthy).any():
        return True
    try:
        return bool(series.fillna(False).astype(bool).any())
    except Exception:
        return False


def sample_is_augmented(group: pd.DataFrame | None = None, video_id: str | None = None) -> bool:
    vid = str(video_id or "").lower()
    if "_augmentation" in vid or "_aug_" in vid or "_augmented" in vid:
        return True
    if group is None or group.empty:
        return False
    if "is_augmented" in group.columns and _truthy_series(group["is_augmented"]):
        return True
    if "augmented_from" in group.columns:
        try:
            if group["augmented_from"].fillna("").astype(str).str.strip().ne("").any():
                return True
        except Exception:
            pass
    if "extract_profile" in group.columns:
        try:
            if group["extract_profile"].astype(str).str.lower().eq("augment").any():
                return True
        except Exception:
            pass
    return False


def normalize_eval_suite_name(name: str) -> str:
    value = str(name or "main").strip().lower()
    aliases = {
        "main_gru": "main",
        "main-gru": "main",
        "utama": "main",
        "boosted": "boosted_stack",
    }
    value = aliases.get(value, value)
    if value not in EVAL_SUITE_NAMES and value != "all":
        raise ValueError(f"Unknown eval suite '{name}'. Pilih: all, {', '.join(EVAL_SUITE_NAMES)}")
    return value


def expand_eval_suite_names(value: str | Iterable[str] | None) -> tuple[str, ...]:
    if value is None:
        return ("main",)
    if isinstance(value, str):
        raw_values = [item.strip() for item in value.split(",") if item.strip()]
    else:
        raw_values = [str(item).strip() for item in value if str(item).strip()]
    if not raw_values:
        return ("main",)
    suites: list[str] = []
    for raw in raw_values:
        suite = normalize_eval_suite_name(raw)
        if suite == "all":
            for item in EVAL_SUITE_NAMES:
                if item not in suites:
                    suites.append(item)
        elif suite not in suites:
            suites.append(suite)
    return tuple(suites)


def classification_metrics(y_true: Iterable[str], y_pred: Iterable[str]) -> dict[str, float]:
    true = [str(value) for value in y_true]
    pred = [str(value) for value in y_pred]
    if len(true) != len(pred):
        raise ValueError("y_true dan y_pred harus sama panjang.")
    total = len(true)
    if total == 0:
        return {
            "accuracy": 0.0,
            "precision_macro": 0.0,
            "recall_macro": 0.0,
            "f1_macro": 0.0,
            "precision_micro": 0.0,
            "recall_micro": 0.0,
            "f1_micro": 0.0,
        }

    labels = sorted(set(true) | set(pred))
    per_label = []
    micro_tp = micro_fp = micro_fn = 0
    for label in labels:
        tp = sum(1 for a, b in zip(true, pred) if a == label and b == label)
        fp = sum(1 for a, b in zip(true, pred) if a != label and b == label)
        fn = sum(1 for a, b in zip(true, pred) if a == label and b != label)
        micro_tp += tp
        micro_fp += fp
        micro_fn += fn
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_label.append((precision, recall, f1))

    precision_micro = micro_tp / (micro_tp + micro_fp) if (micro_tp + micro_fp) else 0.0
    recall_micro = micro_tp / (micro_tp + micro_fn) if (micro_tp + micro_fn) else 0.0
    f1_micro = 2.0 * precision_micro * recall_micro / (precision_micro + recall_micro) if (precision_micro + recall_micro) else 0.0
    return {
        "accuracy": sum(1 for a, b in zip(true, pred) if a == b) / total,
        "precision_macro": float(np.mean([item[0] for item in per_label])) if per_label else 0.0,
        "recall_macro": float(np.mean([item[1] for item in per_label])) if per_label else 0.0,
        "f1_macro": float(np.mean([item[2] for item in per_label])) if per_label else 0.0,
        "precision_micro": precision_micro,
        "recall_micro": recall_micro,
        "f1_micro": f1_micro,
    }


def artifact_paths(variant: str, model_dir: str | Path = MODEL_DIR, schema: str = fs.DEFAULT_SCHEMA) -> dict[str, Path]:
    variant = normalize_variant_name(variant)
    root = fs.model_dir_for(schema, model_dir)
    stem = f"{GRU_PREFIX}{variant}"
    return {
        "weights": root / f"{stem}.pth",
        "labels": root / f"{stem}_labels.json",
        "metadata": root / f"{stem}_metadata.json",
    }


def legacy_artifact_paths(variant: str, model_dir: str | Path = MODEL_DIR) -> dict[str, Path]:
    variant = normalize_variant_name(variant)
    root = Path(model_dir)
    stem = f"{GRU_PREFIX}{variant}"
    return {
        "weights": root / f"{stem}.pth",
        "labels": root / f"{stem}_labels.json",
        "metadata": root / f"{stem}_metadata.json",
    }


def _existing_artifact_paths(variant: str, model_dir: str | Path = MODEL_DIR, schema: str = fs.DEFAULT_SCHEMA) -> dict[str, Path]:
    paths = artifact_paths(variant, model_dir, schema=schema)
    if paths["weights"].exists() or fs.normalize_schema_name(schema) != fs.DEFAULT_SCHEMA:
        return paths
    legacy = legacy_artifact_paths(variant, model_dir)
    if legacy["weights"].exists():
        return legacy
    return paths


def checkpoint_exists(variant: str, model_dir: str | Path = MODEL_DIR, schema: str = fs.DEFAULT_SCHEMA) -> bool:
    paths = _existing_artifact_paths(variant, model_dir, schema=schema)
    return paths["weights"].exists() and paths["labels"].exists()


def _clean_specialist_token(value: str) -> str:
    text = str(value or "").strip().lower().replace(" ", "_").replace("-", "_")
    cleaned = "".join(char if (char.isalnum() or char == "_") else "_" for char in text)
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_")


def normalize_specialist_labels(labels: Iterable[str] | str) -> tuple[str, ...]:
    if isinstance(labels, str):
        raw_values = labels.replace(";", ",").split(",")
    else:
        raw_values = list(labels)
    out: list[str] = []
    for raw in raw_values:
        label = _clean_specialist_token(str(raw))
        if not label or label in EXCLUDED_LABELS or label in out:
            continue
        out.append(label)
    if len(out) < 2:
        raise ValueError("Spesialis butuh minimal 2 vocab non-idle.")
    return tuple(sorted(out))


def specialist_name_from_labels(labels: Iterable[str] | str) -> str:
    return "_".join(normalize_specialist_labels(labels))


def normalize_specialist_name(name: str | None = None, labels: Iterable[str] | str | None = None) -> str:
    cleaned = _clean_specialist_token(name or "")
    if not cleaned and labels is not None:
        cleaned = specialist_name_from_labels(labels)
    if not cleaned:
        raise ValueError("Nama spesialis wajib diisi atau turunkan dari label.")
    return cleaned


def specialist_artifact_paths(
    variant: str,
    specialist_name: str,
    model_dir: str | Path = MODEL_DIR,
    schema: str = fs.DEFAULT_SCHEMA,
) -> dict[str, Path]:
    variant = normalize_variant_name(variant)
    schema_spec = fs.get_schema(schema)
    name = normalize_specialist_name(specialist_name)
    root = fs.model_dir_for(schema_spec.name, model_dir) / "specialists" / name
    stem = f"{GRU_PREFIX}{variant}"
    return {
        "weights": root / f"{stem}.pth",
        "labels": root / f"{stem}_labels.json",
        "metadata": root / f"{stem}_metadata.json",
    }


def specialist_checkpoint_exists(
    variant: str,
    specialist_name: str,
    model_dir: str | Path = MODEL_DIR,
    schema: str = fs.DEFAULT_SCHEMA,
) -> bool:
    paths = specialist_artifact_paths(variant, specialist_name, model_dir=model_dir, schema=schema)
    return paths["weights"].exists() and paths["labels"].exists()


def _backup_relative_path(path: Path) -> Path:
    try:
        return path.resolve().relative_to(ROOT_DIR.resolve())
    except Exception:
        return Path(path.parent.name) / path.name


def backup_existing_files(
    paths: Iterable[str | Path],
    *,
    backup_root: str | Path = BACKUP_ROOT,
    prefix: str = "gru_artifacts",
    copied: set[Path] | None = None,
) -> Path | None:
    existing: list[Path] = []
    for raw_path in paths:
        path = Path(raw_path)
        if path.exists() and path.is_file():
            existing.append(path)
    if not existing:
        return None

    backup_dir = Path(backup_root) / f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    for path in existing:
        resolved = path.resolve()
        if copied is not None and resolved in copied:
            continue
        dest = backup_dir / _backup_relative_path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
        if copied is not None:
            copied.add(resolved)
        print(f"[BACKUP] {path} -> {dest}", flush=True)
    return backup_dir


def get_device(device: str = "auto") -> torch.device:
    requested = str(device or "auto").lower()
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        print("CUDA tidak tersedia, fallback ke CPU.")
        requested = "cpu"
    return torch.device(requested)


def configure_torch_runtime(num_threads: int | None = None) -> None:
    if num_threads is None:
        return
    try:
        torch.set_num_threads(max(1, int(num_threads)))
    except Exception:
        pass
    try:
        torch.set_num_interop_threads(1)
    except Exception:
        pass


def build_model(variant: str, input_dim: int = sc.FEATURE_DIM, num_classes: int = 1) -> nn.Module:
    spec = variant_spec(variant)
    return spec.module.build_model(input_dim=int(input_dim), num_classes=int(num_classes))


def trace_for_inference(
    model: nn.Module,
    target_frames: int,
    device: torch.device,
    enabled: bool = True,
    feature_dim: int = sc.FEATURE_DIM,
) -> nn.Module:
    if not enabled or device.type != "cpu":
        return model
    example = torch.zeros(1, int(target_frames), int(feature_dim), device=device)
    try:
        with torch.inference_mode():
            traced = torch.jit.trace(model, example, check_trace=False)
            traced.eval()
            return torch.jit.optimize_for_inference(traced)
    except Exception:
        return model


def resample_sequence(sequence: np.ndarray, target_frames: int, feature_dim: int = sc.FEATURE_DIM) -> np.ndarray:
    seq = sc.ensure_feature_dim(sequence, int(feature_dim))
    target_frames = int(target_frames)
    if target_frames <= 0:
        raise ValueError("target_frames harus > 0")
    if len(seq) == target_frames:
        return seq.astype(np.float32, copy=False)
    if len(seq) == 1:
        return np.repeat(seq, target_frames, axis=0).astype(np.float32, copy=False)

    old_x = np.linspace(0.0, 1.0, num=len(seq), dtype=np.float32)
    new_x = np.linspace(0.0, 1.0, num=target_frames, dtype=np.float32)
    out = np.empty((target_frames, seq.shape[1]), dtype=np.float32)
    for col in range(seq.shape[1]):
        out[:, col] = np.interp(new_x, old_x, seq[:, col]).astype(np.float32)
    return out


def _sequence_from_group(group: pd.DataFrame, feature_dim: int = sc.FEATURE_DIM) -> np.ndarray:
    group = group.sort_values("frame_num") if "frame_num" in group.columns else group
    features = [sc.parse_feature_value(value) for value in group["features"].tolist()]
    return sc.ensure_feature_dim(features, int(feature_dim))


def load_sequences(
    dataset_dir: str | Path = DATASET_DIR,
    split: str | None = None,
    include_idle: bool = False,
    limit_per_class: int | None = None,
    schema: str = fs.DEFAULT_SCHEMA,
    augmentation_filter: str = "include",
) -> list[SequenceSample]:
    """Load parquet rows for one feature schema, grouped as video sequences."""

    schema_spec = fs.get_schema(schema)
    augmentation_mode = normalize_augmentation_filter_mode(augmentation_filter)
    dataset_root = Path(dataset_dir)
    if not dataset_root.exists():
        raise FileNotFoundError(f"Folder dataset tidak ditemukan: {dataset_root}")

    samples: list[SequenceSample] = []
    per_class_counter: dict[str, int] = {}
    seen_samples: set[tuple[str, str]] = set()
    for parquet_path in fs.dataset_parquet_paths(schema_spec, dataset_root):
        try:
            df = pd.read_parquet(parquet_path)
        except Exception as exc:
            print(f"Skip {parquet_path.name}: gagal dibaca ({exc})")
            continue

        df = fs.filter_feature_rows(df, schema_spec)
        if df.empty or "features" not in df.columns:
            continue
        if split is not None and "split" in df.columns:
            df = df[df["split"].astype(str).str.lower() == str(split).lower()]
        if df.empty:
            continue

        if "label" not in df.columns:
            df = df.copy()
            df["label"] = parquet_path.stem
        if not include_idle:
            df = df[~df["label"].astype(str).str.lower().isin(EXCLUDED_LABELS)]
        if df.empty:
            continue

        group_cols = ["label"]
        if "video_id" in df.columns:
            group_cols.append("video_id")
        else:
            df = df.copy()
            df["video_id"] = parquet_path.stem
            group_cols.append("video_id")

        for (label, video_id), group in df.groupby(group_cols, sort=False):
            label = str(label)
            augmented = sample_is_augmented(group, str(video_id))
            if augmentation_mode == "exclude" and augmented:
                continue
            if augmentation_mode == "only" and not augmented:
                continue
            sample_key = (label, str(video_id))
            if sample_key in seen_samples:
                continue
            if limit_per_class is not None and per_class_counter.get(label, 0) >= int(limit_per_class):
                continue
            seq = _sequence_from_group(group, schema_spec.feature_dim)
            if len(seq) < 1:
                continue
            sample_split = str(group["split"].iloc[0]) if "split" in group.columns else "train"
            samples.append(SequenceSample(label=label, video_id=str(video_id), split=sample_split, sequence=seq, is_augmented=augmented))
            seen_samples.add(sample_key)
            per_class_counter[label] = per_class_counter.get(label, 0) + 1

    return samples


def dataset_summary(dataset_dir: str | Path = DATASET_DIR, schema: str = fs.DEFAULT_SCHEMA) -> dict[str, object]:
    samples = load_sequences(dataset_dir=dataset_dir, include_idle=True, schema=schema)
    labels = sorted({sample.label for sample in samples})
    counts: dict[str, dict[str, int]] = {}
    for sample in samples:
        counts.setdefault(sample.label, {})
        split = sample.split.lower()
        counts[sample.label][split] = counts[sample.label].get(split, 0) + 1
    return {
        "total_samples": len(samples),
        "num_classes_with_idle": len(labels),
        "num_classifier_classes": len([label for label in labels if label.lower() not in EXCLUDED_LABELS]),
        "labels": labels,
        "counts": counts,
    }


def make_label_maps(samples: Iterable[SequenceSample]) -> tuple[dict[str, int], dict[int, str]]:
    labels = sorted({sample.label for sample in samples if sample.label.lower() not in EXCLUDED_LABELS})
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}
    return label_to_idx, idx_to_label


class GRUSequenceDataset(Dataset):
    def __init__(
        self,
        samples: list[SequenceSample],
        label_to_idx: dict[str, int],
        target_frames: int,
        feature_dim: int = sc.FEATURE_DIM,
    ) -> None:
        self.items = [sample for sample in samples if sample.label in label_to_idx]
        self.label_to_idx = label_to_idx
        self.target_frames = int(target_frames)
        self.feature_dim = int(feature_dim)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.items[index]
        seq = resample_sequence(sample.sequence, self.target_frames, self.feature_dim)
        x = torch.from_numpy(seq)
        y = torch.tensor(self.label_to_idx[sample.label], dtype=torch.long)
        return x, y


def _regularization_loss(model: nn.Module, l1: float, l2: float) -> torch.Tensor:
    params = [param for param in model.parameters() if param.requires_grad and param.ndim > 1]
    if not params or (l1 <= 0.0 and l2 <= 0.0):
        return next(model.parameters()).new_tensor(0.0)
    loss = params[0].new_tensor(0.0)
    if l1 > 0.0:
        loss = loss + float(l1) * sum(param.abs().sum() for param in params)
    if l2 > 0.0:
        loss = loss + float(l2) * sum(param.pow(2).sum() for param in params)
    return loss


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    l1: float = 0.0,
    l2: float = 0.0,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            logits = model(x)
            loss = criterion(logits, y)
            if training:
                loss = loss + _regularization_loss(model, l1=l1, l2=l2)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

        total_loss += float(loss.detach().cpu()) * int(y.numel())
        correct += int((logits.argmax(dim=1) == y).sum().detach().cpu())
        total += int(y.numel())

    if total == 0:
        return 0.0, 0.0
    return total_loss / total, correct / total


def train_variant(
    variant: str,
    dataset_dir: str | Path = DATASET_DIR,
    model_dir: str | Path = MODEL_DIR,
    schema: str = fs.DEFAULT_SCHEMA,
    epochs: int | None = None,
    batch_size: int | None = None,
    lr: float | None = None,
    patience: int | None = None,
    device: str = "auto",
    limit_per_class: int | None = None,
    l1: float | None = None,
    l2: float | None = None,
    overwrite_existing: bool = False,
    backup_root: str | Path = BACKUP_ROOT,
    train_data: str | None = None,
) -> tuple[bool, str]:
    variant = normalize_variant_name(variant)
    requested_mode = normalize_train_data_mode(train_data or variant_train_data_mode(variant))
    if requested_mode == "both":
        raise ValueError("train_variant hanya menerima satu mode data; pakai expand_variant_request untuk both.")
    if requested_mode == "with_augmentation" and not is_augmented_variant(variant):
        variant = augmented_variant_name(variant)
    train_data_mode = "with_augmentation" if is_augmented_variant(variant) else "original"
    base_variant = base_variant_name(variant)
    spec = variant_spec(variant)
    schema_spec = fs.get_schema(schema)
    paths = artifact_paths(variant, model_dir, schema=schema_spec.name)
    existing_targets = [path for path in paths.values() if path.exists()]
    if existing_targets and not overwrite_existing:
        msg = (
            f"[SKIP checkpoint exists] {schema_spec.name}/gru_{variant}: "
            + ", ".join(str(path) for path in existing_targets)
        )
        print(msg, flush=True)
        return True, msg

    epochs = int(epochs or spec.default_epochs)
    batch_size = int(batch_size or spec.default_batch_size)
    lr = float(lr or spec.default_lr)
    patience = int(patience if patience is not None else spec.default_patience)
    l1 = float(spec.default_l1 if l1 is None else l1)
    l2 = float(spec.default_l2 if l2 is None else l2)

    samples = load_sequences(
        dataset_dir=dataset_dir,
        include_idle=False,
        limit_per_class=limit_per_class,
        schema=schema_spec.name,
        augmentation_filter="include" if train_data_mode == "with_augmentation" else "exclude",
    )
    train_samples = [sample for sample in samples if sample.split.lower() == "train"]
    val_samples = [sample for sample in samples if sample.split.lower() == "val" and not sample.is_augmented]

    if not train_samples:
        return False, f"Tidak ada data train {schema_spec.display_name} di {dataset_dir}."

    label_to_idx, idx_to_label = make_label_maps(train_samples)
    val_samples = [sample for sample in val_samples if sample.label in label_to_idx]
    if len(label_to_idx) < 2:
        return False, "Butuh minimal 2 kelas non-idle untuk training GRU."

    train_dataset = GRUSequenceDataset(train_samples, label_to_idx, spec.target_frames, schema_spec.feature_dim)
    val_dataset = GRUSequenceDataset(val_samples, label_to_idx, spec.target_frames, schema_spec.feature_dim)
    effective_batch = max(1, min(batch_size, len(train_dataset)))
    if len(train_dataset) >= 2:
        effective_batch = max(2, effective_batch)
    drop_last = len(train_dataset) > effective_batch and len(train_dataset) % effective_batch == 1
    train_loader = DataLoader(
        train_dataset,
        batch_size=effective_batch,
        shuffle=True,
        num_workers=0,
        drop_last=drop_last,
    )
    val_loader = DataLoader(val_dataset, batch_size=max(1, min(effective_batch, max(1, len(val_dataset)))), shuffle=False, num_workers=0)

    selected_device = get_device(device)
    model = build_model(variant, input_dim=schema_spec.feature_dim, num_classes=len(label_to_idx)).to(selected_device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_score = -1.0
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    stale_epochs = 0
    history: list[dict[str, float]] = []

    print(
        f"Training {spec.display_name}: {len(train_dataset)} train, {len(val_dataset)} val, "
        f"{len(label_to_idx)} kelas, schema={schema_spec.name}:{schema_spec.feature_dim}, "
        f"target={spec.target_frames}, data={train_data_mode}, device={selected_device}"
    )
    for epoch in tqdm(range(1, epochs + 1), desc=f"train-{variant}", unit="epoch"):
        train_loss, train_acc = _run_epoch(
            model,
            train_loader,
            selected_device,
            criterion,
            optimizer=optimizer,
            l1=l1,
            l2=l2,
        )
        if len(val_dataset) > 0:
            with torch.inference_mode():
                val_loss, val_acc = _run_epoch(model, val_loader, selected_device, criterion)
        else:
            val_loss, val_acc = train_loss, train_acc

        history.append(
            {
                "epoch": float(epoch),
                "train_loss": float(train_loss),
                "train_acc": float(train_acc),
                "val_loss": float(val_loss),
                "val_acc": float(val_acc),
            }
        )

        score = val_acc
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1

        if patience > 0 and stale_epochs >= patience:
            print(f"Early stopping epoch {epoch}; best epoch {best_epoch} val_acc={best_score:.4f}")
            break

    model.load_state_dict(best_state)
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    labels_json = {str(idx): label for idx, label in idx_to_label.items()}
    metadata = {
        "variant": variant,
        "base_variant": base_variant,
        "display_name": spec.display_name,
        "training_data_mode": train_data_mode,
        "uses_augmented_data": train_data_mode == "with_augmentation",
        "schema": schema_spec.name,
        "schema_display_name": schema_spec.display_name,
        "feature_schema": schema_spec.feature_schema,
        "feature_mode": schema_spec.feature_mode,
        "feature_dim": schema_spec.feature_dim,
        "target_fps": schema_spec.target_fps,
        "target_frames": spec.target_frames,
        "num_classes": len(label_to_idx),
        "labels": labels_json,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "train_augmented_samples": sum(1 for sample in train_samples if sample.is_augmented),
        "val_augmented_samples": sum(1 for sample in val_samples if sample.is_augmented),
        "epochs_requested": epochs,
        "epochs_run": len(history),
        "best_epoch": best_epoch,
        "best_val_acc": float(best_score),
        "batch_size": effective_batch,
        "lr": lr,
        "l1": l1,
        "l2": l2,
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    checkpoint = {
        "model_state": model.state_dict(),
        "metadata": metadata,
        "labels": labels_json,
    }
    backup_existing_files(paths.values(), backup_root=backup_root, prefix="gru_checkpoint")
    torch.save(checkpoint, paths["weights"])
    paths["labels"].write_text(json.dumps(labels_json, indent=2), encoding="utf-8")
    paths["metadata"].write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return True, f"{spec.display_name} {schema_spec.name} tersimpan: {paths['weights']} (best val acc {best_score:.3f})"


def _specialist_sample_copy(sample: SequenceSample) -> SequenceSample:
    return SequenceSample(
        label=_clean_specialist_token(sample.label),
        video_id=sample.video_id,
        split=sample.split,
        sequence=sample.sequence,
        is_augmented=sample.is_augmented,
    )


def train_specialist_variant(
    variant: str,
    specialist_labels: Iterable[str] | str,
    specialist_name: str | None = None,
    dataset_dir: str | Path = DATASET_DIR,
    model_dir: str | Path = MODEL_DIR,
    schema: str = fs.DEFAULT_SCHEMA,
    epochs: int | None = None,
    batch_size: int | None = None,
    lr: float | None = None,
    patience: int | None = None,
    device: str = "auto",
    limit_per_class: int | None = None,
    l1: float | None = None,
    l2: float | None = None,
    overwrite_existing: bool = False,
    backup_root: str | Path = BACKUP_ROOT,
    train_data: str | None = None,
) -> tuple[bool, str]:
    labels = normalize_specialist_labels(specialist_labels)
    label_set = set(labels)
    name = normalize_specialist_name(specialist_name, labels)
    variant = normalize_variant_name(variant)
    requested_mode = normalize_train_data_mode(train_data or variant_train_data_mode(variant))
    if requested_mode == "both":
        raise ValueError("train_specialist_variant hanya menerima satu mode data.")
    if requested_mode == "with_augmentation" and not is_augmented_variant(variant):
        variant = augmented_variant_name(variant)
    train_data_mode = "with_augmentation" if is_augmented_variant(variant) else "original"
    base_variant = base_variant_name(variant)
    spec = variant_spec(variant)
    schema_spec = fs.get_schema(schema)
    paths = specialist_artifact_paths(variant, name, model_dir=model_dir, schema=schema_spec.name)
    existing_targets = [path for path in paths.values() if path.exists()]
    if existing_targets and not overwrite_existing:
        msg = (
            f"[SKIP specialist exists] {schema_spec.name}/specialists/{name}/gru_{variant}: "
            + ", ".join(str(path) for path in existing_targets)
        )
        print(msg, flush=True)
        return True, msg

    epochs = int(epochs or SPECIALIST_DEFAULT_EPOCHS)
    batch_size = int(batch_size or SPECIALIST_DEFAULT_BATCH_SIZE)
    lr = float(lr or spec.default_lr)
    patience = int(patience if patience is not None else SPECIALIST_DEFAULT_PATIENCE)
    l1 = float(spec.default_l1 if l1 is None else l1)
    l2 = float(spec.default_l2 if l2 is None else l2)

    samples = load_sequences(
        dataset_dir=dataset_dir,
        include_idle=False,
        limit_per_class=limit_per_class,
        schema=schema_spec.name,
        augmentation_filter="include" if train_data_mode == "with_augmentation" else "exclude",
    )
    specialist_samples = [
        _specialist_sample_copy(sample)
        for sample in samples
        if _clean_specialist_token(sample.label) in label_set
    ]
    train_samples = [sample for sample in specialist_samples if sample.split.lower() == "train"]
    val_samples = [sample for sample in specialist_samples if sample.split.lower() == "val" and not sample.is_augmented]

    if not train_samples:
        return False, f"Tidak ada data train untuk spesialis {name} ({', '.join(labels)}) di {dataset_dir}."
    present = {sample.label for sample in train_samples}
    missing = [label for label in labels if label not in present]
    if missing:
        return False, f"Data train spesialis {name} belum lengkap. Missing: {', '.join(missing)}."

    label_to_idx, idx_to_label = make_label_maps(train_samples)
    val_samples = [sample for sample in val_samples if sample.label in label_to_idx]
    if len(label_to_idx) < 2:
        return False, "Butuh minimal 2 kelas non-idle untuk training spesialis."

    train_dataset = GRUSequenceDataset(train_samples, label_to_idx, spec.target_frames, schema_spec.feature_dim)
    val_dataset = GRUSequenceDataset(val_samples, label_to_idx, spec.target_frames, schema_spec.feature_dim)
    effective_batch = max(1, min(batch_size, len(train_dataset)))
    if len(train_dataset) >= 2:
        effective_batch = max(2, effective_batch)
    drop_last = len(train_dataset) > effective_batch and len(train_dataset) % effective_batch == 1
    train_loader = DataLoader(
        train_dataset,
        batch_size=effective_batch,
        shuffle=True,
        num_workers=0,
        drop_last=drop_last,
    )
    val_loader = DataLoader(val_dataset, batch_size=max(1, min(effective_batch, max(1, len(val_dataset)))), shuffle=False, num_workers=0)

    selected_device = get_device(device)
    model = build_model(variant, input_dim=schema_spec.feature_dim, num_classes=len(label_to_idx)).to(selected_device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_score = -1.0
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    stale_epochs = 0
    history: list[dict[str, float]] = []

    print(
        f"Training specialist {name} {spec.display_name}: {len(train_dataset)} train, {len(val_dataset)} val, "
        f"{len(label_to_idx)} kelas ({', '.join(labels)}), schema={schema_spec.name}:{schema_spec.feature_dim}, "
        f"target={spec.target_frames}, data={train_data_mode}, device={selected_device}"
    )
    for epoch in tqdm(range(1, epochs + 1), desc=f"specialist-{name}-{variant}", unit="epoch"):
        train_loss, train_acc = _run_epoch(
            model,
            train_loader,
            selected_device,
            criterion,
            optimizer=optimizer,
            l1=l1,
            l2=l2,
        )
        if len(val_dataset) > 0:
            with torch.inference_mode():
                val_loss, val_acc = _run_epoch(model, val_loader, selected_device, criterion)
        else:
            val_loss, val_acc = train_loss, train_acc

        history.append(
            {
                "epoch": float(epoch),
                "train_loss": float(train_loss),
                "train_acc": float(train_acc),
                "val_loss": float(val_loss),
                "val_acc": float(val_acc),
            }
        )

        if val_acc > best_score:
            best_score = val_acc
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1

        if patience > 0 and stale_epochs >= patience:
            print(f"Early stopping specialist {name} epoch {epoch}; best epoch {best_epoch} val_acc={best_score:.4f}")
            break

    model.load_state_dict(best_state)
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    labels_json = {str(idx): label for idx, label in idx_to_label.items()}
    metadata = {
        "classifier_scope": "specialist",
        "specialist_name": name,
        "specialist_labels": list(labels),
        "variant": variant,
        "base_variant": base_variant,
        "display_name": spec.display_name,
        "training_data_mode": train_data_mode,
        "uses_augmented_data": train_data_mode == "with_augmentation",
        "schema": schema_spec.name,
        "schema_display_name": schema_spec.display_name,
        "feature_schema": schema_spec.feature_schema,
        "feature_mode": schema_spec.feature_mode,
        "feature_dim": schema_spec.feature_dim,
        "target_fps": schema_spec.target_fps,
        "target_frames": spec.target_frames,
        "num_classes": len(label_to_idx),
        "labels": labels_json,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "train_augmented_samples": sum(1 for sample in train_samples if sample.is_augmented),
        "val_augmented_samples": sum(1 for sample in val_samples if sample.is_augmented),
        "epochs_requested": epochs,
        "epochs_run": len(history),
        "best_epoch": best_epoch,
        "best_val_acc": float(best_score),
        "batch_size": effective_batch,
        "lr": lr,
        "l1": l1,
        "l2": l2,
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    checkpoint = {
        "model_state": model.state_dict(),
        "metadata": metadata,
        "labels": labels_json,
    }
    backup_existing_files(paths.values(), backup_root=backup_root, prefix="gru_specialist")
    torch.save(checkpoint, paths["weights"])
    paths["labels"].write_text(json.dumps(labels_json, indent=2), encoding="utf-8")
    paths["metadata"].write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return True, f"Specialist {name} {spec.display_name} {schema_spec.name} tersimpan: {paths['weights']} (best val acc {best_score:.3f})"


def train_all(train_data: str = "original", **kwargs) -> dict[str, tuple[bool, str]]:
    results = {}
    for variant in expand_variant_request("all", train_data):
        results[variant] = train_variant(variant, train_data=variant_train_data_mode(variant), **kwargs)
    return results


def load_labels(variant: str, model_dir: str | Path = MODEL_DIR, schema: str = fs.DEFAULT_SCHEMA) -> dict[int, str]:
    paths = _existing_artifact_paths(variant, model_dir, schema=schema)
    data = json.loads(paths["labels"].read_text(encoding="utf-8"))
    return {int(idx): str(label) for idx, label in data.items()}


def load_metadata(variant: str, model_dir: str | Path = MODEL_DIR, schema: str = fs.DEFAULT_SCHEMA) -> dict[str, object]:
    paths = _existing_artifact_paths(variant, model_dir, schema=schema)
    return json.loads(paths["metadata"].read_text(encoding="utf-8"))


def available_variants(
    model_dir: str | Path = MODEL_DIR,
    schema: str = fs.DEFAULT_SCHEMA,
    variants: Iterable[str] | None = None,
) -> list[str]:
    variant_pool = tuple(normalize_variant_name(variant) for variant in (variants or VARIANT_NAMES))
    return [variant for variant in variant_pool if checkpoint_exists(variant, model_dir, schema=schema)]


def select_best_available_variant(
    model_dir: str | Path = MODEL_DIR,
    schema: str = fs.DEFAULT_SCHEMA,
    variants: Iterable[str] | None = None,
) -> str:
    schema_spec = fs.get_schema(schema)
    variant_pool = tuple(variants) if variants is not None else BASE_VARIANT_NAMES
    candidates = available_variants(model_dir, schema=schema_spec.name, variants=variant_pool)
    if not candidates:
        raise FileNotFoundError(f"Belum ada checkpoint GRU {schema_spec.name} yang bisa dipakai live.")

    def score(variant: str) -> float:
        try:
            metadata = load_metadata(variant, model_dir, schema=schema_spec.name)
            if metadata.get("feature_schema") != schema_spec.feature_schema:
                return -1.0
            return float(metadata.get("best_val_acc", -1.0))
        except Exception:
            return -1.0

    return max(candidates, key=score)


def load_checkpoint(
    variant: str,
    model_dir: str | Path = MODEL_DIR,
    device: str | torch.device = "auto",
    schema: str = fs.DEFAULT_SCHEMA,
) -> tuple[nn.Module, dict[int, str], dict[str, object], torch.device]:
    variant = normalize_variant_name(variant)
    schema_spec = fs.get_schema(schema)
    selected_device = device if isinstance(device, torch.device) else get_device(str(device))
    paths = _existing_artifact_paths(variant, model_dir, schema=schema_spec.name)
    if not paths["weights"].exists():
        raise FileNotFoundError(f"Checkpoint belum ada: {paths['weights']}")

    labels = load_labels(variant, model_dir, schema=schema_spec.name)
    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8")) if paths["metadata"].exists() else {}
    if metadata.get("feature_schema") not in (None, schema_spec.feature_schema):
        raise RuntimeError(f"Checkpoint stale: {metadata.get('feature_schema')} != {schema_spec.feature_schema}")

    model = build_model(variant, input_dim=schema_spec.feature_dim, num_classes=len(labels))
    checkpoint = torch.load(paths["weights"], map_location=selected_device)
    state = checkpoint.get("model_state", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state)
    model.to(selected_device)
    model.eval()
    return model, labels, metadata, selected_device


def load_specialist_labels(
    variant: str,
    specialist_name: str,
    model_dir: str | Path = MODEL_DIR,
    schema: str = fs.DEFAULT_SCHEMA,
) -> dict[int, str]:
    paths = specialist_artifact_paths(variant, specialist_name, model_dir=model_dir, schema=schema)
    data = json.loads(paths["labels"].read_text(encoding="utf-8"))
    return {int(idx): str(label) for idx, label in data.items()}


def load_specialist_metadata(
    variant: str,
    specialist_name: str,
    model_dir: str | Path = MODEL_DIR,
    schema: str = fs.DEFAULT_SCHEMA,
) -> dict[str, object]:
    paths = specialist_artifact_paths(variant, specialist_name, model_dir=model_dir, schema=schema)
    return json.loads(paths["metadata"].read_text(encoding="utf-8"))


def load_specialist_checkpoint(
    variant: str,
    specialist_name: str,
    model_dir: str | Path = MODEL_DIR,
    device: str | torch.device = "auto",
    schema: str = fs.DEFAULT_SCHEMA,
) -> tuple[nn.Module, dict[int, str], dict[str, object], torch.device]:
    variant = normalize_variant_name(variant)
    schema_spec = fs.get_schema(schema)
    name = normalize_specialist_name(specialist_name)
    selected_device = device if isinstance(device, torch.device) else get_device(str(device))
    paths = specialist_artifact_paths(variant, name, model_dir=model_dir, schema=schema_spec.name)
    if not paths["weights"].exists():
        raise FileNotFoundError(f"Checkpoint spesialis belum ada: {paths['weights']}")

    labels = load_specialist_labels(variant, name, model_dir=model_dir, schema=schema_spec.name)
    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8")) if paths["metadata"].exists() else {}
    if metadata.get("classifier_scope") not in (None, "specialist"):
        raise RuntimeError(f"Checkpoint bukan spesialis: {metadata.get('classifier_scope')}")
    if metadata.get("feature_schema") not in (None, schema_spec.feature_schema):
        raise RuntimeError(f"Checkpoint spesialis stale: {metadata.get('feature_schema')} != {schema_spec.feature_schema}")

    model = build_model(variant, input_dim=schema_spec.feature_dim, num_classes=len(labels))
    checkpoint = torch.load(paths["weights"], map_location=selected_device)
    state = checkpoint.get("model_state", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state)
    model.to(selected_device)
    model.eval()
    return model, labels, metadata, selected_device


def predict_sequence(
    model: nn.Module,
    sequence: np.ndarray,
    labels: dict[int, str],
    target_frames: int,
    device: torch.device,
    feature_dim: int = sc.FEATURE_DIM,
) -> tuple[str, float, list[tuple[str, float]]]:
    seq = resample_sequence(sequence, target_frames, feature_dim)
    x = torch.from_numpy(seq).unsqueeze(0).to(device)
    with torch.inference_mode():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
    order = np.argsort(-probs)
    top = [(labels[int(idx)], float(probs[int(idx)])) for idx in order[: min(3, len(order))]]
    best_idx = int(order[0])
    return labels[best_idx], float(probs[best_idx]), top


def evaluate_variant(
    variant: str,
    dataset_dir: str | Path = DATASET_DIR,
    model_dir: str | Path = MODEL_DIR,
    schema: str = fs.DEFAULT_SCHEMA,
    split: str = "test",
    device: str = "auto",
    suite: str = "main",
) -> dict[str, object]:
    variant = normalize_variant_name(variant)
    suite = normalize_eval_suite_name(suite)
    if suite == "all":
        raise ValueError("evaluate_variant hanya menerima satu suite. Pakai expand_eval_suite_names untuk all.")
    spec = variant_spec(variant)
    schema_spec = fs.get_schema(schema)
    samples = load_sequences(dataset_dir=dataset_dir, split=split, include_idle=False, schema=schema_spec.name, augmentation_filter="exclude")
    if suite == "main":
        model, labels, metadata, selected_device = load_checkpoint(
            variant,
            model_dir=model_dir,
            device=device,
            schema=schema_spec.name,
        )

        def predict_fn(sequence: np.ndarray) -> tuple[str, float, list[tuple[str, float]]]:
            return predict_sequence(model, sequence, labels, spec.target_frames, selected_device, feature_dim=schema_spec.feature_dim)

        allowed_labels = set(labels.values())
    else:
        import gru_experts as ge

        ge.require_route_available(variant, schema_spec.name, model_dir, suite)
        predictor = ge.RoutedGRUPredictor(
            variant,
            schema_spec.name,
            model_dir,
            device=device,
            route=suite,
        )
        metadata = predictor.main_metadata

        def predict_fn(sequence: np.ndarray) -> tuple[str, float, list[tuple[str, float]]]:
            return predictor.predict(sequence)

        allowed_labels = set(predictor.main_labels.values())

    y_true: list[str] = []
    y_pred: list[str] = []
    confidences: list[float] = []
    for sample in samples:
        if sample.label not in allowed_labels:
            continue
        pred, conf, _ = predict_fn(sample.sequence)
        y_true.append(sample.label)
        y_pred.append(pred)
        confidences.append(conf)

    metrics = classification_metrics(y_true, y_pred)
    return {
        "variant": variant,
        "schema": schema_spec.name,
        "suite": suite,
        "split": split,
        "metadata": metadata,
        "y_true": y_true,
        "y_pred": y_pred,
        "confidence": confidences,
        "samples": len(y_true),
        "metrics": metrics,
        **metrics,
    }


def benchmark_variant(
    variant: str,
    model_dir: str | Path = MODEL_DIR,
    schema: str = fs.DEFAULT_SCHEMA,
    device: str = "cpu",
    warmup: int = 5,
    runs: int = 30,
    threads: int | None = 1,
    use_jit: bool = True,
) -> dict[str, object]:
    variant = normalize_variant_name(variant)
    spec = variant_spec(variant)
    schema_spec = fs.get_schema(schema)
    requested_device = str(device or "auto").lower()
    if requested_device == "auto":
        selected_device_name, device_reason = jr.select_live_device(variant, requested="auto", torch_module=torch)
    elif requested_device == "cuda":
        selected_device_name, device_reason = jr.select_live_device(variant, requested="cuda", torch_module=torch)
    elif requested_device == "cpu":
        selected_device_name, device_reason = jr.select_live_device(variant, requested="cpu", torch_module=torch)
    else:
        raise ValueError("device harus salah satu: auto, cpu, cuda")
    configure_torch_runtime(threads)
    model, labels, metadata, selected_device = load_checkpoint(
        variant,
        model_dir=model_dir,
        device=selected_device_name,
        schema=schema_spec.name,
    )
    model = trace_for_inference(
        model,
        spec.target_frames,
        selected_device,
        enabled=use_jit,
        feature_dim=schema_spec.feature_dim,
    )
    sequence = np.random.default_rng(42).normal(size=(spec.target_frames, schema_spec.feature_dim)).astype(np.float32)
    for _ in range(max(0, int(warmup))):
        predict_sequence(model, sequence, labels, spec.target_frames, selected_device, feature_dim=schema_spec.feature_dim)
    timings = []
    for _ in range(max(1, int(runs))):
        t0 = time.perf_counter()
        predict_sequence(model, sequence, labels, spec.target_frames, selected_device, feature_dim=schema_spec.feature_dim)
        timings.append((time.perf_counter() - t0) * 1000.0)
    arr = np.asarray(timings, dtype=np.float32)
    return {
        "variant": variant,
        "schema": schema_spec.name,
        "requested_device": requested_device,
        "device": str(selected_device),
        "device_reason": device_reason,
        "target_frames": spec.target_frames,
        "num_classes": len(labels),
        "runs": int(runs),
        "mean_ms": float(arr.mean()),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "metadata": metadata,
        "threads": threads,
        "jit": bool(use_jit and selected_device.type == "cpu"),
    }


def list_model_status(model_dir: str | Path = MODEL_DIR, schema: str = fs.DEFAULT_SCHEMA) -> dict[str, str]:
    schema_spec = fs.get_schema(schema)
    status: dict[str, str] = {}
    for variant in VARIANT_NAMES:
        paths = _existing_artifact_paths(variant, model_dir, schema=schema_spec.name)
        if not checkpoint_exists(variant, model_dir, schema=schema_spec.name):
            status[variant] = "belum trained"
            continue
        try:
            metadata = json.loads(paths["metadata"].read_text(encoding="utf-8")) if paths["metadata"].exists() else {}
            schema_ok = metadata.get("feature_schema") == schema_spec.feature_schema
            trained_at = metadata.get("trained_at", "-")
            val_acc = metadata.get("best_val_acc", None)
            val_text = f", val={float(val_acc):.3f}" if isinstance(val_acc, (float, int)) else ""
            warning = ""
            if isinstance(val_acc, (float, int)) and float(val_acc) < 0.70:
                warning = " [LOW]"
            status[variant] = f"OK {trained_at}{val_text}{warning}" if schema_ok else "stale schema"
        except Exception as exc:
            status[variant] = f"metadata error: {exc}"
    return status


def _cmd_train(args: argparse.Namespace) -> int:
    variants = expand_variant_request(args.variant, getattr(args, "train_data", "original"))
    schemas = fs.expand_schema_names(args.schema)
    exit_code = 0
    for schema_name in schemas:
        for variant in variants:
            ok, msg = train_variant(
                variant,
                dataset_dir=args.dataset_dir,
                model_dir=args.model_dir,
                schema=schema_name,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                patience=args.patience,
                device=args.device,
                limit_per_class=args.limit_per_class,
                l1=args.l1,
                l2=args.l2,
                overwrite_existing=bool(args.overwrite_existing),
                backup_root=args.backup_root,
                train_data=variant_train_data_mode(variant),
            )
            print(msg)
            if not ok:
                exit_code = 1
    return exit_code


def _cmd_status(args: argparse.Namespace) -> int:
    schemas = fs.expand_schema_names(args.schema)
    for schema_name in schemas:
        spec = fs.get_schema(schema_name)
        try:
            summary = dataset_summary(args.dataset_dir, schema=schema_name)
            print(f"Dataset {schema_name} ({spec.feature_dim}D): {summary['total_samples']} sampel, {summary['num_classifier_classes']} kelas classifier")
        except Exception as exc:
            print(f"Dataset {schema_name} ({spec.feature_dim}D): unavailable ({exc})")
    diag = jr.diagnostics(torch_module=torch)
    if diag.get("warning"):
        print(f"Runtime warning: {diag['warning']}")
    for schema_name in schemas:
        print(f"Checkpoint {schema_name}:")
        for variant, value in list_model_status(args.model_dir, schema=schema_name).items():
            print(f"  {variant}: {value}")
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    variants = expand_variant_request(args.variant, getattr(args, "train_data", "both"))
    suites = expand_eval_suite_names(getattr(args, "suite", "main"))
    print(
        "schema | variant | suite | split | samples | accuracy | precision_macro | recall_macro | f1_macro | "
        "precision_micro | recall_micro | f1_micro"
    )
    for schema_name in fs.expand_schema_names(args.schema):
        for variant in variants:
            for suite in suites:
                try:
                    result = evaluate_variant(
                        variant,
                        dataset_dir=args.dataset_dir,
                        model_dir=args.model_dir,
                        schema=schema_name,
                        split=args.split,
                        device=args.device,
                        suite=suite,
                    )
                except FileNotFoundError as exc:
                    print(f"{schema_name}/{variant}/{suite}: skip ({exc})")
                    continue
                if not result["y_true"]:
                    print(f"{schema_name}/{variant}/{suite}: tidak ada sampel evaluasi.")
                    continue
                print(
                    f"{schema_name} | {variant} | {suite} | {args.split} | {int(result['samples'])} | "
                    f"{float(result['accuracy']):.4f} | {float(result['precision_macro']):.4f} | "
                    f"{float(result['recall_macro']):.4f} | {float(result['f1_macro']):.4f} | "
                    f"{float(result['precision_micro']):.4f} | {float(result['recall_micro']):.4f} | "
                    f"{float(result['f1_micro']):.4f}"
                )
    return 0


def _cmd_benchmark(args: argparse.Namespace) -> int:
    variants = expand_variant_request(args.variant, getattr(args, "train_data", "both"))
    exit_code = 0
    for schema_name in fs.expand_schema_names(args.schema):
        for variant in variants:
            try:
                result = benchmark_variant(
                    variant,
                    model_dir=args.model_dir,
                    schema=schema_name,
                    device=args.device,
                    warmup=args.warmup,
                    runs=args.runs,
                    threads=args.threads,
                    use_jit=not args.no_jit,
                )
            except FileNotFoundError as exc:
                print(f"{schema_name}/{variant}: skip ({exc})")
                continue
            except (RuntimeError, ValueError) as exc:
                print(f"{schema_name}/{variant}: error ({exc})")
                exit_code = 1
                continue
            print(
                f"{schema_name}/{variant}: mean={result['mean_ms']:.2f} ms "
                f"p50={result['p50_ms']:.2f} ms p95={result['p95_ms']:.2f} ms "
                f"frames={result['target_frames']} classes={result['num_classes']} "
                f"requested={result['requested_device']} device={result['device']}"
                f" threads={result['threads']} jit={result['jit']}"
                f" reason={result['device_reason']}"
            )
    return exit_code


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GRU BISINDO trainer/evaluator")
    sub = parser.add_subparsers(dest="command", required=True)

    common_train_eval = argparse.ArgumentParser(add_help=False)
    common_train_eval.add_argument("--dataset-dir", default=str(DATASET_DIR))
    common_train_eval.add_argument("--model-dir", default=str(MODEL_DIR))
    common_train_eval.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    common_train_eval.add_argument("--schema", default=fs.DEFAULT_SCHEMA, choices=[*fs.SCHEMA_NAMES, "all", "base", "original", "face", "full", "extra"])

    train = sub.add_parser("train", parents=[common_train_eval], help="Train satu/semua model GRU")
    train.add_argument("--variant", default="all", help="Varian GRU, comma list, atau all")
    train.add_argument("--train-data", default="original", choices=["original", "with-augmentation", "with_augmentation", "both"])
    train.add_argument("--epochs", type=int, default=None)
    train.add_argument("--batch-size", type=int, default=None)
    train.add_argument("--lr", type=float, default=None)
    train.add_argument("--patience", type=int, default=None)
    train.add_argument("--limit-per-class", type=int, default=None)
    train.add_argument("--l1", type=float, default=None)
    train.add_argument("--l2", type=float, default=None)
    train.add_argument("--overwrite-existing", action="store_true", help="Backup lalu timpa checkpoint target yang sudah ada")
    train.add_argument("--backup-root", default=str(BACKUP_ROOT))
    train.set_defaults(func=_cmd_train)

    status = sub.add_parser("status", help="Tampilkan status dataset/model")
    status.add_argument("--dataset-dir", default=str(DATASET_DIR))
    status.add_argument("--model-dir", default=str(MODEL_DIR))
    status.add_argument("--schema", default=fs.DEFAULT_SCHEMA, choices=[*fs.SCHEMA_NAMES, "all", "base", "original", "face", "full", "extra"])
    status.set_defaults(func=_cmd_status)

    eval_cmd = sub.add_parser("eval", parents=[common_train_eval], help="Evaluasi checkpoint GRU")
    eval_cmd.add_argument("--variant", default="all", help="Varian GRU, comma list, atau all")
    eval_cmd.add_argument("--train-data", default="both", choices=["original", "with-augmentation", "with_augmentation", "both"])
    eval_cmd.add_argument("--suite", default="main", help="main, route expert, comma list, atau all")
    eval_cmd.add_argument("--split", default="test", choices=["train", "val", "test"])
    eval_cmd.set_defaults(func=_cmd_eval)

    bench = sub.add_parser("benchmark", parents=[common_train_eval], help="Benchmark latency inference model-only")
    bench.add_argument("--variant", default="all", help="Varian GRU, comma list, atau all")
    bench.add_argument("--train-data", default="both", choices=["original", "with-augmentation", "with_augmentation", "both"])
    bench.add_argument("--warmup", type=int, default=5)
    bench.add_argument("--runs", type=int, default=30)
    bench.add_argument("--threads", type=int, default=1)
    bench.add_argument("--no-jit", action="store_true")
    bench.set_defaults(func=_cmd_benchmark)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
