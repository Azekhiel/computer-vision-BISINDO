"""Expert GRU suites and routed prediction for BISINDO models."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from datetime import datetime
import json
import math
from pathlib import Path
import pickle
from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import feature_schemas as fs
import gru_manager as gm


SUITE_CHOICES = ("main", "chunk10", "threshold", "boosted")
ROUTE_CHOICES = ("main", "chunk10", "threshold", "vote_all", "main_chunk10", "main_threshold", "boosted_stack")
EXPERT_PREFIX = "gru_expert"


@dataclass(frozen=True)
class PrototypeBundle:
    labels: list[str]
    mean: np.ndarray
    std: np.ndarray
    prototypes: dict[str, np.ndarray]


def parse_suite_names(value: str | Iterable[str] | None) -> tuple[str, ...]:
    if value is None:
        return ("main",)
    if isinstance(value, str):
        parts = [part.strip().lower() for part in value.split(",")]
    else:
        parts = [str(part).strip().lower() for part in value]
    if "all" in parts:
        parts = list(SUITE_CHOICES)
    out: list[str] = []
    for part in parts:
        if not part:
            continue
        if part not in SUITE_CHOICES:
            raise ValueError(f"Unknown suite '{part}'. Pilih: {', '.join(SUITE_CHOICES)}")
        if part not in out:
            out.append(part)
    return tuple(out or ("main",))


def normalize_route_name(value: str | None) -> str:
    route = str(value or "main").strip().lower()
    aliases = {
        "direct": "main",
        "utama": "main",
        "per10": "chunk10",
        "chunk": "chunk10",
        "threshold_all": "threshold",
        "vote": "vote_all",
        "all_vote": "vote_all",
        "main_per10": "main_chunk10",
        "utama_per10": "main_chunk10",
        "main_thresh": "main_threshold",
        "stack": "boosted_stack",
        "boosted": "boosted_stack",
    }
    route = aliases.get(route, route)
    if route not in ROUTE_CHOICES:
        raise ValueError(f"Unknown route '{value}'. Pilih: {', '.join(ROUTE_CHOICES)}")
    return route


def suite_root(model_dir: str | Path, schema: str, suite: str, variant: str) -> Path:
    schema_root = fs.model_dir_for(schema, model_dir)
    return schema_root / "experts" / suite / f"gru_{gm.normalize_variant_name(variant)}"


def _expert_paths(root: Path, index: int) -> dict[str, Path]:
    item_root = root / f"chunk_{index:03d}"
    return {
        "root": item_root,
        "weights": item_root / f"{EXPERT_PREFIX}.pth",
        "labels": item_root / f"{EXPERT_PREFIX}_labels.json",
        "metadata": item_root / f"{EXPERT_PREFIX}_metadata.json",
    }


def _suite_metadata_path(root: Path) -> Path:
    return root / "suite_metadata.json"


def _prototype_npz_path(root: Path) -> Path:
    return root / "prototypes.npz"


def _boosted_path(root: Path) -> Path:
    return root / "boosted_stack.pkl"


def route_required_suites(route: str) -> tuple[str, ...]:
    route = normalize_route_name(route)
    if route in {"chunk10", "main_chunk10"}:
        return ("chunk10",)
    if route in {"threshold", "main_threshold"}:
        return ("threshold",)
    if route == "vote_all":
        return ("chunk10", "threshold")
    if route == "boosted_stack":
        return ("boosted",)
    return ()


def _expert_suite_ready(root: Path) -> bool:
    meta_path = _suite_metadata_path(root)
    if not meta_path.exists():
        return False
    try:
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    groups = metadata.get("groups") or []
    if not groups:
        return False
    for idx, _group in enumerate(groups):
        paths = _expert_paths(root, idx)
        if not paths["weights"].exists() or not paths["labels"].exists() or not paths["metadata"].exists():
            return False
    return _prototype_npz_path(root).exists()


def _boosted_suite_ready(root: Path) -> bool:
    return _suite_metadata_path(root).exists() and _prototype_npz_path(root).exists() and _boosted_path(root).exists()


def _suite_existing_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    if root.is_file():
        return [root]
    return sorted(path for path in root.rglob("*") if path.is_file())


def _skip_existing_suite_result(root: Path, *, schema: str, variant: str, suite: str) -> dict[str, object]:
    files = _suite_existing_files(root)
    message = f"[SKIP checkpoint exists] {schema}/gru_{variant}/{suite}: {root} ({len(files)} files)"
    print(message, flush=True)
    return {
        "ok": True,
        "skipped": True,
        "message": message,
        "suite": suite,
        "variant": variant,
        "schema": schema,
        "root": str(root),
        "existing_files": len(files),
    }


def _backup_existing_suite(root: Path, *, suite: str, backup_root: str | Path) -> Path | None:
    return gm.backup_existing_files(
        _suite_existing_files(root),
        backup_root=backup_root,
        prefix=f"gru_{suite}_suite",
    )


def route_available(variant: str, schema: str, model_dir: str | Path, route: str) -> bool:
    route = normalize_route_name(route)
    if route == "main":
        return gm.checkpoint_exists(variant, model_dir=model_dir, schema=schema)
    required = route_required_suites(route)
    if route == "vote_all":
        return any(route_available(variant, schema, model_dir, suite) for suite in required)
    for suite in required:
        root = suite_root(model_dir, schema, suite, variant)
        if suite == "boosted":
            if not _boosted_suite_ready(root):
                return False
        elif not _expert_suite_ready(root):
            return False
    return True


def require_route_available(variant: str, schema: str, model_dir: str | Path, route: str) -> None:
    route = normalize_route_name(route)
    if not route_available(variant, schema, model_dir, route):
        raise FileNotFoundError(f"Suite/route {schema}/gru_{gm.normalize_variant_name(variant)}/{route} belum ada.")


def _samples_for_training(dataset_dir: str | Path, schema: str) -> list[gm.SequenceSample]:
    return gm.load_sequences(dataset_dir=dataset_dir, include_idle=False, schema=schema)


def build_prototypes(
    samples: list[gm.SequenceSample],
    variant: str,
    schema: str,
) -> PrototypeBundle:
    variant = gm.normalize_variant_name(variant)
    spec = gm.VARIANTS[variant]
    schema_spec = fs.get_schema(schema)
    labels = sorted({sample.label for sample in samples if sample.label.lower() not in gm.EXCLUDED_LABELS})
    if len(labels) < 2:
        raise ValueError("Butuh minimal 2 kelas untuk prototype grouping.")

    rows: list[np.ndarray] = []
    row_labels: list[str] = []
    for sample in samples:
        if sample.label not in labels or sample.split.lower() != "train":
            continue
        seq = gm.resample_sequence(sample.sequence, spec.target_frames, schema_spec.feature_dim).reshape(-1)
        rows.append(seq.astype(np.float32))
        row_labels.append(sample.label)
    if not rows:
        raise ValueError("Tidak ada sample train untuk prototype grouping.")

    x = np.stack(rows).astype(np.float32)
    mean = x.mean(axis=0).astype(np.float32)
    std = x.std(axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0
    z = (x - mean) / std
    prototypes: dict[str, np.ndarray] = {}
    for label in labels:
        idx = [i for i, row_label in enumerate(row_labels) if row_label == label]
        prototypes[label] = z[idx].mean(axis=0).astype(np.float32)
    return PrototypeBundle(labels=labels, mean=mean, std=std, prototypes=prototypes)


def _distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / math.sqrt(max(1, a.size)))


def make_chunk10_groups(bundle: PrototypeBundle, max_group_size: int = 10) -> list[list[str]]:
    remaining = list(bundle.labels)
    groups: list[list[str]] = []
    while remaining:
        seed = remaining.pop(0)
        group = [seed]
        while remaining and len(group) < int(max_group_size):
            centroid = np.mean([bundle.prototypes[label] for label in group], axis=0)
            nearest = min(remaining, key=lambda label: _distance(bundle.prototypes[label], centroid))
            remaining.remove(nearest)
            group.append(nearest)
        groups.append(group)
    return groups


def make_threshold_groups(
    bundle: PrototypeBundle,
    threshold: float | None = None,
    min_group_size: int = 2,
    max_group_size: int = 12,
) -> tuple[list[list[str]], float]:
    labels = list(bundle.labels)
    if threshold is None:
        nearest = []
        for label in labels:
            others = [other for other in labels if other != label]
            if others:
                nearest.append(min(_distance(bundle.prototypes[label], bundle.prototypes[other]) for other in others))
        threshold = float(np.median(nearest) * 2.25) if nearest else 0.0

    remaining = set(labels)
    groups: list[list[str]] = []
    while remaining:
        seed = sorted(remaining)[0]
        remaining.remove(seed)
        candidates = sorted(
            list(remaining),
            key=lambda label: _distance(bundle.prototypes[seed], bundle.prototypes[label]),
        )
        group = [seed]
        for label in candidates:
            if len(group) >= int(max_group_size):
                break
            if _distance(bundle.prototypes[seed], bundle.prototypes[label]) <= float(threshold):
                group.append(label)
        if len(group) < int(min_group_size) and remaining:
            for label in candidates:
                if label in remaining:
                    group.append(label)
                    break
        for label in group[1:]:
            remaining.discard(label)
        groups.append(group)
    return groups, float(threshold)


def _make_label_maps_for_labels(labels: Iterable[str]) -> tuple[dict[str, int], dict[int, str]]:
    ordered = sorted(str(label) for label in labels)
    label_to_idx = {label: idx for idx, label in enumerate(ordered)}
    return label_to_idx, {idx: label for label, idx in label_to_idx.items()}


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
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
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
        total_loss += float(loss.detach().cpu()) * int(y.numel())
        correct += int((logits.argmax(dim=1) == y).sum().detach().cpu())
        total += int(y.numel())
    if total == 0:
        return 0.0, 0.0
    return total_loss / total, correct / total


def train_expert_group(
    *,
    variant: str,
    schema: str,
    labels: list[str],
    samples: list[gm.SequenceSample],
    out_root: Path,
    epochs: int | None,
    batch_size: int | None,
    lr: float | None,
    patience: int | None,
    device: str,
    group_index: int,
    suite: str,
) -> dict[str, object]:
    variant = gm.normalize_variant_name(variant)
    spec = gm.VARIANTS[variant]
    schema_spec = fs.get_schema(schema)
    labels = sorted(labels)
    if len(labels) < 2:
        return {"ok": False, "message": "skip group dengan <2 label", "labels": labels}

    epochs = int(epochs or spec.default_epochs)
    batch_size = int(batch_size or spec.default_batch_size)
    lr = float(lr or spec.default_lr)
    patience = int(patience if patience is not None else spec.default_patience)
    label_to_idx, idx_to_label = _make_label_maps_for_labels(labels)

    group_samples = [sample for sample in samples if sample.label in label_to_idx]
    train_samples = [sample for sample in group_samples if sample.split.lower() == "train"]
    val_samples = [sample for sample in group_samples if sample.split.lower() == "val"]
    if not train_samples:
        return {"ok": False, "message": "tidak ada train sample", "labels": labels}

    train_dataset = gm.GRUSequenceDataset(train_samples, label_to_idx, spec.target_frames, schema_spec.feature_dim)
    val_dataset = gm.GRUSequenceDataset(val_samples, label_to_idx, spec.target_frames, schema_spec.feature_dim)
    effective_batch = max(1, min(batch_size, len(train_dataset)))
    train_loader = DataLoader(train_dataset, batch_size=effective_batch, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=max(1, min(effective_batch, max(1, len(val_dataset)))), shuffle=False, num_workers=0)
    selected_device = gm.get_device(device)
    model = gm.build_model(variant, input_dim=schema_spec.feature_dim, num_classes=len(label_to_idx)).to(selected_device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_score = -1.0
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    stale = 0
    history: list[dict[str, float]] = []
    for epoch in tqdm(range(1, epochs + 1), desc=f"{suite}-{variant}-{group_index:03d}", unit="epoch"):
        train_loss, train_acc = _run_epoch(model, train_loader, selected_device, criterion, optimizer)
        if len(val_dataset) > 0:
            with torch.inference_mode():
                val_loss, val_acc = _run_epoch(model, val_loader, selected_device, criterion)
        else:
            val_loss, val_acc = train_loss, train_acc
        history.append({"epoch": float(epoch), "train_loss": train_loss, "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_acc})
        if val_acc > best_score:
            best_score = val_acc
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if patience > 0 and stale >= patience:
            break

    model.load_state_dict(best_state)
    paths = _expert_paths(out_root, group_index)
    paths["root"].mkdir(parents=True, exist_ok=True)
    labels_json = {str(idx): label for idx, label in idx_to_label.items()}
    metadata = {
        "suite": suite,
        "group_index": int(group_index),
        "variant": variant,
        "schema": schema_spec.name,
        "feature_schema": schema_spec.feature_schema,
        "feature_dim": schema_spec.feature_dim,
        "target_frames": spec.target_frames,
        "labels": labels_json,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "epochs_run": len(history),
        "best_epoch": best_epoch,
        "best_val_acc": float(best_score),
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    torch.save({"model_state": model.state_dict(), "metadata": metadata, "labels": labels_json}, paths["weights"])
    paths["labels"].write_text(json.dumps(labels_json, indent=2), encoding="utf-8")
    paths["metadata"].write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {"ok": True, "message": str(paths["weights"]), "metadata": metadata, "labels": labels}


def _save_prototypes(root: Path, bundle: PrototypeBundle) -> None:
    root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        _prototype_npz_path(root),
        labels=np.asarray(bundle.labels),
        mean=bundle.mean.astype(np.float32),
        std=bundle.std.astype(np.float32),
        prototypes=np.stack([bundle.prototypes[label] for label in bundle.labels]).astype(np.float32),
    )


def _load_prototypes(root: Path) -> PrototypeBundle:
    data = np.load(_prototype_npz_path(root), allow_pickle=True)
    labels = [str(label) for label in data["labels"].tolist()]
    proto_arr = data["prototypes"].astype(np.float32)
    return PrototypeBundle(
        labels=labels,
        mean=data["mean"].astype(np.float32),
        std=data["std"].astype(np.float32),
        prototypes={label: proto_arr[idx] for idx, label in enumerate(labels)},
    )


def train_group_suite(
    *,
    suite: str,
    variant: str,
    schema: str,
    dataset_dir: str | Path,
    model_dir: str | Path,
    epochs: int | None,
    batch_size: int | None,
    lr: float | None,
    patience: int | None,
    device: str,
    threshold: float | None = None,
    overwrite_existing: bool = False,
    backup_root: str | Path = gm.BACKUP_ROOT,
) -> dict[str, object]:
    suite = str(suite).lower()
    if suite not in {"chunk10", "threshold"}:
        raise ValueError("train_group_suite hanya untuk chunk10/threshold")
    schema_spec = fs.get_schema(schema)
    variant = gm.normalize_variant_name(variant)
    root = suite_root(model_dir, schema_spec.name, suite, variant)
    if _suite_existing_files(root) and not overwrite_existing:
        return _skip_existing_suite_result(root, schema=schema_spec.name, variant=variant, suite=suite)
    _backup_existing_suite(root, suite=suite, backup_root=backup_root)

    samples = _samples_for_training(dataset_dir, schema_spec.name)
    bundle = build_prototypes(samples, variant, schema_spec.name)
    _save_prototypes(root, bundle)
    if suite == "chunk10":
        groups = make_chunk10_groups(bundle)
        used_threshold = None
    else:
        groups, used_threshold = make_threshold_groups(bundle, threshold=threshold)

    results = []
    for idx, labels in enumerate(groups):
        result = train_expert_group(
            variant=variant,
            schema=schema_spec.name,
            labels=labels,
            samples=samples,
            out_root=root,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            patience=patience,
            device=device,
            group_index=idx,
            suite=suite,
        )
        results.append(result)
        print(f"{suite}[{schema_spec.name}/{variant}/{idx:03d}]: {result.get('message')}", flush=True)

    metadata = {
        "suite": suite,
        "variant": gm.normalize_variant_name(variant),
        "schema": schema_spec.name,
        "feature_schema": schema_spec.feature_schema,
        "target": "labels",
        "groups": groups,
        "threshold": used_threshold,
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "results": results,
    }
    _suite_metadata_path(root).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def _main_probabilities(
    model: nn.Module,
    sequence: np.ndarray,
    target_frames: int,
    device: torch.device,
    feature_dim: int,
) -> np.ndarray:
    seq = gm.resample_sequence(sequence, target_frames, feature_dim)
    x = torch.from_numpy(seq).unsqueeze(0).to(device)
    with torch.inference_mode():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
    return probs.astype(np.float32)


def _prototype_distance_features(sequence: np.ndarray, bundle: PrototypeBundle, variant: str, schema: str) -> np.ndarray:
    spec = gm.VARIANTS[gm.normalize_variant_name(variant)]
    schema_spec = fs.get_schema(schema)
    flat = gm.resample_sequence(sequence, spec.target_frames, schema_spec.feature_dim).reshape(-1).astype(np.float32)
    z = (flat - bundle.mean) / bundle.std
    dists = np.asarray([_distance(z, bundle.prototypes[label]) for label in bundle.labels], dtype=np.float32)
    if dists.size == 0:
        return np.zeros(3, dtype=np.float32)
    return np.asarray([float(dists.min()), float(np.median(dists)), float(dists.max())], dtype=np.float32)


def train_boosted_stack(
    *,
    variant: str,
    schema: str,
    dataset_dir: str | Path,
    model_dir: str | Path,
    device: str,
    overwrite_existing: bool = False,
    backup_root: str | Path = gm.BACKUP_ROOT,
) -> dict[str, object]:
    from sklearn.ensemble import GradientBoostingClassifier

    variant = gm.normalize_variant_name(variant)
    schema_spec = fs.get_schema(schema)
    root = suite_root(model_dir, schema_spec.name, "boosted", variant)
    if _suite_existing_files(root) and not overwrite_existing:
        return _skip_existing_suite_result(root, schema=schema_spec.name, variant=variant, suite="boosted")
    _backup_existing_suite(root, suite="boosted", backup_root=backup_root)

    model, labels, metadata, selected_device = gm.load_checkpoint(variant, model_dir=model_dir, device=device, schema=schema_spec.name)
    target_frames = int(metadata.get("target_frames", gm.VARIANTS[variant].target_frames))
    samples = _samples_for_training(dataset_dir, schema_spec.name)
    fit_samples = [sample for sample in samples if sample.split.lower() == "val" and sample.label in set(labels.values())]
    if not fit_samples:
        fit_samples = [sample for sample in samples if sample.split.lower() == "train" and sample.label in set(labels.values())]
    if len({sample.label for sample in fit_samples}) < 2:
        raise ValueError("Butuh minimal 2 label untuk boosted stacker.")

    bundle = build_prototypes(samples, variant, schema_spec.name)
    _save_prototypes(root, bundle)
    inv_labels = {label: idx for idx, label in labels.items()}

    x_rows = []
    y_rows = []
    for sample in fit_samples:
        probs = _main_probabilities(model, sample.sequence, target_frames, selected_device, schema_spec.feature_dim)
        ordered = np.sort(probs)[::-1]
        margin = float(ordered[0] - ordered[1]) if len(ordered) > 1 else float(ordered[0])
        entropy = float(-(probs * np.log(np.clip(probs, 1e-8, 1.0))).sum())
        proto = _prototype_distance_features(sample.sequence, bundle, variant, schema_spec.name)
        x_rows.append(np.concatenate((probs, np.asarray([margin, entropy], dtype=np.float32), proto)).astype(np.float32))
        y_rows.append(inv_labels[sample.label])

    clf = GradientBoostingClassifier(random_state=42)
    clf.fit(np.stack(x_rows), np.asarray(y_rows, dtype=np.int64))
    root.mkdir(parents=True, exist_ok=True)
    with _boosted_path(root).open("wb") as f:
        pickle.dump(clf, f)
    boosted_metadata = {
        "suite": "boosted",
        "variant": variant,
        "schema": schema_spec.name,
        "feature_schema": schema_spec.feature_schema,
        "base_labels": {str(idx): label for idx, label in labels.items()},
        "samples": len(fit_samples),
        "feature_dim": int(np.stack(x_rows).shape[1]),
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    _suite_metadata_path(root).write_text(json.dumps(boosted_metadata, indent=2), encoding="utf-8")
    return boosted_metadata


def train_suite(
    *,
    variant: str,
    schema: str,
    suites: str | Iterable[str],
    dataset_dir: str | Path = gm.DATASET_DIR,
    model_dir: str | Path = gm.MODEL_DIR,
    epochs: int | None = None,
    batch_size: int | None = None,
    lr: float | None = None,
    patience: int | None = None,
    device: str = "auto",
    threshold: float | None = None,
    limit_per_class: int | None = None,
    overwrite_existing: bool = False,
    backup_root: str | Path = gm.BACKUP_ROOT,
) -> dict[str, object]:
    variant = gm.normalize_variant_name(variant)
    schema_spec = fs.get_schema(schema)
    selected_suites = parse_suite_names(suites)
    results: dict[str, object] = {}
    if "main" in selected_suites:
        ok, msg = gm.train_variant(
            variant,
            dataset_dir=dataset_dir,
            model_dir=model_dir,
            schema=schema_spec.name,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            patience=patience,
            device=device,
            limit_per_class=limit_per_class,
            overwrite_existing=overwrite_existing,
            backup_root=backup_root,
        )
        print(msg, flush=True)
        results["main"] = {"ok": ok, "message": msg}
    if any(suite in selected_suites for suite in ("boosted", "chunk10", "threshold")) and not gm.checkpoint_exists(variant, model_dir, schema=schema_spec.name):
        ok, msg = gm.train_variant(
            variant,
            dataset_dir=dataset_dir,
            model_dir=model_dir,
            schema=schema_spec.name,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            patience=patience,
            device=device,
            limit_per_class=limit_per_class,
            overwrite_existing=overwrite_existing,
            backup_root=backup_root,
        )
        print(msg, flush=True)
        results.setdefault("main", {"ok": ok, "message": msg})
    if "chunk10" in selected_suites:
        results["chunk10"] = train_group_suite(
            suite="chunk10",
            variant=variant,
            schema=schema_spec.name,
            dataset_dir=dataset_dir,
            model_dir=model_dir,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            patience=patience,
            device=device,
            overwrite_existing=overwrite_existing,
            backup_root=backup_root,
        )
    if "threshold" in selected_suites:
        results["threshold"] = train_group_suite(
            suite="threshold",
            variant=variant,
            schema=schema_spec.name,
            dataset_dir=dataset_dir,
            model_dir=model_dir,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            patience=patience,
            device=device,
            threshold=threshold,
            overwrite_existing=overwrite_existing,
            backup_root=backup_root,
        )
    if "boosted" in selected_suites:
        results["boosted"] = train_boosted_stack(
            variant=variant,
            schema=schema_spec.name,
            dataset_dir=dataset_dir,
            model_dir=model_dir,
            device=device,
            overwrite_existing=overwrite_existing,
            backup_root=backup_root,
        )
    return results


def _load_expert(root: Path, index: int, variant: str, schema: str, device: torch.device):
    paths = _expert_paths(root, index)
    labels = json.loads(paths["labels"].read_text(encoding="utf-8"))
    labels_map = {int(idx): str(label) for idx, label in labels.items()}
    schema_spec = fs.get_schema(schema)
    model = gm.build_model(variant, input_dim=schema_spec.feature_dim, num_classes=len(labels_map))
    checkpoint = torch.load(paths["weights"], map_location=device)
    state = checkpoint.get("model_state", checkpoint)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    return model, labels_map, metadata


class RoutedGRUPredictor:
    def __init__(self, variant: str, schema: str, model_dir: str | Path, device: str | torch.device = "auto", route: str = "main") -> None:
        self.variant = gm.normalize_variant_name(variant)
        self.schema = fs.normalize_schema_name(schema)
        self.schema_spec = fs.get_schema(self.schema)
        self.route = normalize_route_name(route)
        self.main_model, self.main_labels, self.main_metadata, self.device = gm.load_checkpoint(
            self.variant,
            model_dir=model_dir,
            device=device,
            schema=self.schema,
        )
        self.target_frames = int(self.main_metadata.get("target_frames", gm.VARIANTS[self.variant].target_frames))
        self.model_dir = Path(model_dir)

    def _predict_main(self, sequence: np.ndarray) -> tuple[str, float, list[tuple[str, float]], np.ndarray]:
        probs = _main_probabilities(self.main_model, sequence, self.target_frames, self.device, self.schema_spec.feature_dim)
        order = np.argsort(-probs)
        top = [(self.main_labels[int(idx)], float(probs[int(idx)])) for idx in order[: min(3, len(order))]]
        best = int(order[0])
        return self.main_labels[best], float(probs[best]), top, probs

    def _nearest_group_index(self, suite: str, sequence: np.ndarray, allowed_labels: set[str] | None = None) -> int | None:
        root = suite_root(self.model_dir, self.schema, suite, self.variant)
        metadata = json.loads(_suite_metadata_path(root).read_text(encoding="utf-8"))
        bundle = _load_prototypes(root)
        spec = gm.VARIANTS[self.variant]
        flat = gm.resample_sequence(sequence, spec.target_frames, self.schema_spec.feature_dim).reshape(-1).astype(np.float32)
        z = (flat - bundle.mean) / bundle.std
        group_scores = []
        for idx, group in enumerate(metadata.get("groups", [])):
            if allowed_labels and not set(group).intersection(allowed_labels):
                continue
            centroid = np.mean([bundle.prototypes[label] for label in group if label in bundle.prototypes], axis=0)
            group_scores.append((idx, _distance(z, centroid)))
        if not group_scores:
            return None
        return min(group_scores, key=lambda item: item[1])[0]

    def _predict_expert(self, suite: str, sequence: np.ndarray, group_index: int) -> tuple[str, float, list[tuple[str, float]]]:
        root = suite_root(self.model_dir, self.schema, suite, self.variant)
        model, labels, metadata = _load_expert(root, group_index, self.variant, self.schema, self.device)
        target_frames = int(metadata.get("target_frames", self.target_frames))
        return gm.predict_sequence(model, sequence, labels, target_frames, self.device, feature_dim=self.schema_spec.feature_dim)

    def _vote_all(self, sequence: np.ndarray) -> tuple[str, float, list[tuple[str, float]]]:
        votes: dict[str, float] = {}
        label, conf, top, _probs = self._predict_main(sequence)
        votes[label] = votes.get(label, 0.0) + conf
        for suite in ("chunk10", "threshold"):
            root = suite_root(self.model_dir, self.schema, suite, self.variant)
            meta_path = _suite_metadata_path(root)
            if not meta_path.exists():
                continue
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            for idx, _group in enumerate(metadata.get("groups", [])):
                try:
                    ex_label, ex_conf, _ = self._predict_expert(suite, sequence, idx)
                except Exception:
                    continue
                votes[ex_label] = votes.get(ex_label, 0.0) + ex_conf
        if not votes:
            return label, conf, top
        ranked = sorted(votes.items(), key=lambda item: item[1], reverse=True)
        total = max(1e-6, sum(votes.values()))
        top_vote = [(label, float(score / total)) for label, score in ranked[:3]]
        return top_vote[0][0], top_vote[0][1], top_vote

    def _boosted(self, sequence: np.ndarray) -> tuple[str, float, list[tuple[str, float]]]:
        root = suite_root(self.model_dir, self.schema, "boosted", self.variant)
        if not _boosted_path(root).exists():
            label, conf, top, _ = self._predict_main(sequence)
            return label, conf, top
        with _boosted_path(root).open("rb") as f:
            clf = pickle.load(f)
        base_label, base_conf, base_top, probs = self._predict_main(sequence)
        bundle = _load_prototypes(root)
        proto = _prototype_distance_features(sequence, bundle, self.variant, self.schema)
        ordered = np.sort(probs)[::-1]
        margin = float(ordered[0] - ordered[1]) if len(ordered) > 1 else float(ordered[0])
        entropy = float(-(probs * np.log(np.clip(probs, 1e-8, 1.0))).sum())
        feats = np.concatenate((probs, np.asarray([margin, entropy], dtype=np.float32), proto)).reshape(1, -1)
        if hasattr(clf, "predict_proba"):
            boosted_probs = clf.predict_proba(feats)[0]
            class_order = [int(value) for value in clf.classes_]
            order = np.argsort(-boosted_probs)
            top = [(self.main_labels[class_order[int(idx)]], float(boosted_probs[int(idx)])) for idx in order[: min(3, len(order))]]
            return top[0][0], top[0][1], top
        pred = int(clf.predict(feats)[0])
        return self.main_labels.get(pred, base_label), base_conf, base_top

    def predict(self, sequence: np.ndarray) -> tuple[str, float, list[tuple[str, float]]]:
        if self.route == "main":
            label, conf, top, _ = self._predict_main(sequence)
            return label, conf, top
        if self.route == "chunk10":
            idx = self._nearest_group_index("chunk10", sequence)
            if idx is not None:
                return self._predict_expert("chunk10", sequence, idx)
        if self.route == "threshold":
            idx = self._nearest_group_index("threshold", sequence)
            if idx is not None:
                return self._predict_expert("threshold", sequence, idx)
        if self.route == "main_chunk10":
            label, _conf, _top, _ = self._predict_main(sequence)
            idx = self._nearest_group_index("chunk10", sequence, allowed_labels={label})
            if idx is not None:
                return self._predict_expert("chunk10", sequence, idx)
        if self.route == "main_threshold":
            label, _conf, _top, _ = self._predict_main(sequence)
            idx = self._nearest_group_index("threshold", sequence, allowed_labels={label})
            if idx is not None:
                return self._predict_expert("threshold", sequence, idx)
        if self.route == "vote_all":
            return self._vote_all(sequence)
        if self.route == "boosted_stack":
            return self._boosted(sequence)
        label, conf, top, _ = self._predict_main(sequence)
        return label, conf, top
