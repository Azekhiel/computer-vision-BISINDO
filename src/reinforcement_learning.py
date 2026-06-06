"""Isolated reinforcement fine-tuning for BISINDO GRU checkpoints."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np
import torch
from torch import nn

import feature_schemas as fs
import gru_experts as ge
import gru_manager as gm


MODEL_REINFORCEMENT_DIR = gm.ROOT_DIR / "model_reinforcement"


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _checkpoint_paths(variant: str, model_dir: str | Path, schema: str) -> dict[str, Path]:
    if hasattr(gm, "_existing_artifact_paths"):
        return gm._existing_artifact_paths(variant, model_dir, schema=schema)  # type: ignore[attr-defined]
    return gm.artifact_paths(variant, model_dir=model_dir, schema=schema)


def _copy_missing_file(src: Path, dest: Path) -> bool:
    if dest.exists():
        return False
    if not src.exists():
        raise FileNotFoundError(f"Artefak base tidak ditemukan: {src}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    return True


def _copy_missing_tree(src: Path, dest: Path) -> int:
    if not src.exists():
        raise FileNotFoundError(f"Suite base tidak ditemukan: {src}")
    copied = 0
    for path in sorted(src.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(src)
        copied += int(_copy_missing_file(path, dest / rel))
    return copied


def resolve_variant(schema: str, variant: str, base_model_dir: str | Path = gm.MODEL_DIR) -> str:
    schema_name = fs.normalize_schema_name(schema)
    value = str(variant or "auto").strip().lower()
    if value in {"auto", "best"}:
        return gm.select_best_available_variant(model_dir=base_model_dir, schema=schema_name)
    return gm.normalize_variant_name(value)


def clone_base_artifacts(
    *,
    schema: str,
    variant: str,
    route: str,
    base_model_dir: str | Path = gm.MODEL_DIR,
    model_root: str | Path = MODEL_REINFORCEMENT_DIR,
) -> dict[str, Any]:
    schema_name = fs.normalize_schema_name(schema)
    variant_name = resolve_variant(schema_name, variant, base_model_dir=base_model_dir)
    route_name = ge.normalize_route_name(route)

    if not gm.checkpoint_exists(variant_name, model_dir=base_model_dir, schema=schema_name):
        raise FileNotFoundError(f"Checkpoint base belum ada: {schema_name}/gru_{variant_name}")
    ge.require_route_available(variant_name, schema_name, base_model_dir, route_name)

    base_paths = _checkpoint_paths(variant_name, base_model_dir, schema_name)
    dest_paths = gm.artifact_paths(variant_name, model_dir=model_root, schema=schema_name)
    copied_files = []
    for key in ("weights", "labels", "metadata"):
        if _copy_missing_file(base_paths[key], dest_paths[key]):
            copied_files.append(str(dest_paths[key]))

    metadata = {}
    if dest_paths["metadata"].exists():
        try:
            metadata = json.loads(dest_paths["metadata"].read_text(encoding="utf-8"))
        except Exception:
            metadata = {}
    metadata.setdefault("variant", variant_name)
    metadata.setdefault("schema", schema_name)
    metadata["reinforcement"] = {
        "enabled": True,
        "base_model_dir": str(base_model_dir),
        "model_root": str(model_root),
        "route": route_name,
        "cloned_or_checked_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    dest_paths["metadata"].parent.mkdir(parents=True, exist_ok=True)
    dest_paths["metadata"].write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    copied_suites: dict[str, int] = {}
    for suite in ge.route_required_suites(route_name):
        base_root = ge.suite_root(base_model_dir, schema_name, suite, variant_name)
        dest_root = ge.suite_root(model_root, schema_name, suite, variant_name)
        copied_suites[suite] = _copy_missing_tree(base_root, dest_root)

    return {
        "schema": schema_name,
        "variant": variant_name,
        "route": route_name,
        "model_root": str(model_root),
        "checkpoint": str(dest_paths["weights"]),
        "copied_files": copied_files,
        "copied_suites": copied_suites,
    }


@dataclass
class Correction:
    sequence: np.ndarray
    label: str
    predicted_label: str = "-"
    confidence: float = 0.0
    prediction_id: int = 0


@dataclass
class ReinforcementSession:
    schema: str
    variant: str = "auto"
    route: str = "main"
    device: str = "auto"
    base_model_dir: str | Path = gm.MODEL_DIR
    model_root: str | Path = MODEL_REINFORCEMENT_DIR
    backup_root: str | Path = gm.BACKUP_ROOT
    lr: float = 1e-4
    steps_per_correction: int = 2
    replay_limit: int = 32
    session_id: str = field(default_factory=lambda: f"rl_{_timestamp()}")

    def __post_init__(self) -> None:
        self.schema = fs.normalize_schema_name(self.schema)
        self.route = ge.normalize_route_name(self.route)
        self.model_root = Path(self.model_root)
        self.base_model_dir = Path(self.base_model_dir)
        self.backup_root = Path(self.backup_root)
        self.variant = str(self.variant or "auto")
        self.lr = float(self.lr)
        self.steps_per_correction = max(1, int(self.steps_per_correction))
        self.replay_limit = max(1, int(self.replay_limit))
        self.started = False
        self.resolved_variant = ""
        self.labels: dict[int, str] = {}
        self.metadata: dict[str, Any] = {}
        self.correction_count = 0
        self.replay: list[Correction] = []
        self.session_dir = self.model_root / "sessions" / self.session_id
        self.corrections_dir = self.session_dir / "corrections"
        self.corrections_jsonl = self.session_dir / "corrections.jsonl"
        self.last_backup_dir: Path | None = None

    def start(self) -> dict[str, Any]:
        info = clone_base_artifacts(
            schema=self.schema,
            variant=self.variant,
            route=self.route,
            base_model_dir=self.base_model_dir,
            model_root=self.model_root,
        )
        self.resolved_variant = str(info["variant"])
        _model, labels, metadata, _device = gm.load_checkpoint(
            self.resolved_variant,
            model_dir=self.model_root,
            device=self.device,
            schema=self.schema,
        )
        self.labels = labels
        self.metadata = metadata
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.corrections_dir.mkdir(parents=True, exist_ok=True)
        session_meta = {
            "session_id": self.session_id,
            "schema": self.schema,
            "variant": self.resolved_variant,
            "route": self.route,
            "device": self.device,
            "model_root": str(self.model_root),
            "base_model_dir": str(self.base_model_dir),
            "started_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        (self.session_dir / "session_metadata.json").write_text(json.dumps(session_meta, indent=2), encoding="utf-8")
        self.started = True
        info["labels"] = self.available_labels()
        info["session_dir"] = str(self.session_dir)
        return info

    def available_labels(self) -> list[str]:
        return [self.labels[idx] for idx in sorted(self.labels)]

    def predict(self, sequence: np.ndarray) -> tuple[str, float, list[tuple[str, float]]]:
        self._require_started()
        if self.route == "main":
            model, labels, metadata, device = gm.load_checkpoint(
                self.resolved_variant,
                model_dir=self.model_root,
                device=self.device,
                schema=self.schema,
            )
            target_frames = int(metadata.get("target_frames", gm.variant_spec(self.resolved_variant).target_frames))
            return gm.predict_sequence(model, sequence, labels, target_frames, device, feature_dim=fs.get_schema(self.schema).feature_dim)
        predictor = ge.RoutedGRUPredictor(
            self.resolved_variant,
            self.schema,
            self.model_root,
            device=self.device,
            route=self.route,
        )
        return predictor.predict(sequence)

    def apply_correction(
        self,
        *,
        sequence: np.ndarray,
        corrected_label: str,
        predicted_label: str = "-",
        confidence: float = 0.0,
        prediction_id: int = 0,
    ) -> dict[str, Any]:
        self._require_started()
        label = str(corrected_label or "").strip()
        if label not in set(self.labels.values()):
            raise ValueError(f"Label koreksi '{corrected_label}' tidak ada di checkpoint ini.")
        arr = fs.ensure_feature_dim(sequence, self.schema).astype(np.float32, copy=True)
        correction = Correction(
            sequence=arr,
            label=label,
            predicted_label=str(predicted_label or "-"),
            confidence=float(confidence or 0.0),
            prediction_id=int(prediction_id or 0),
        )
        self.replay.append(correction)
        self.replay = self.replay[-self.replay_limit :]
        self.correction_count += 1
        self._write_correction_record(correction)
        train_info = self._fine_tune()
        return {
            "ok": True,
            "correction_count": self.correction_count,
            "label": label,
            **train_info,
        }

    def finish(self) -> Path:
        if not self.model_root.exists():
            raise FileNotFoundError(f"Folder reinforcement belum ada: {self.model_root}")
        backup_dir = self.backup_root / f"cleanup_{_timestamp()}" / "model_reinforcement"
        suffix = 1
        while backup_dir.exists():
            backup_dir = self.backup_root / f"cleanup_{_timestamp()}_{suffix:02d}" / "model_reinforcement"
            suffix += 1
        self.last_backup_dir = backup_dir
        if self.started:
            self._write_session_finished()
        backup_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(self.model_root, backup_dir)
        return backup_dir

    def _require_started(self) -> None:
        if not self.started or not self.resolved_variant:
            raise RuntimeError("Sesi reinforcement belum start.")

    def _write_correction_record(self, correction: Correction) -> None:
        self.corrections_dir.mkdir(parents=True, exist_ok=True)
        npz_path = self.corrections_dir / f"correction_{self.correction_count:06d}.npz"
        np.savez_compressed(npz_path, sequence=correction.sequence, label=correction.label)
        record = {
            "index": self.correction_count,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "schema": self.schema,
            "variant": self.resolved_variant,
            "route": self.route,
            "predicted_label": correction.predicted_label,
            "corrected_label": correction.label,
            "confidence": correction.confidence,
            "prediction_id": correction.prediction_id,
            "sequence_npz": str(npz_path),
            "frames": int(correction.sequence.shape[0]),
            "feature_dim": int(correction.sequence.shape[1]),
        }
        self.corrections_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with self.corrections_jsonl.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def _fine_tune(self) -> dict[str, Any]:
        model, labels, metadata, device = gm.load_checkpoint(
            self.resolved_variant,
            model_dir=self.model_root,
            device=self.device,
            schema=self.schema,
        )
        label_to_idx = {label: idx for idx, label in labels.items()}
        feature_dim = fs.get_schema(self.schema).feature_dim
        target_frames = int(metadata.get("target_frames", gm.variant_spec(self.resolved_variant).target_frames))
        items = self.replay[-self.replay_limit :]
        x_np = np.stack([gm.resample_sequence(item.sequence, target_frames, feature_dim) for item in items]).astype(np.float32)
        y_np = np.asarray([label_to_idx[item.label] for item in items], dtype=np.int64)
        if x_np.shape[0] == 1:
            x_np = np.concatenate([x_np, x_np.copy()], axis=0)
            y_np = np.concatenate([y_np, y_np.copy()], axis=0)
        x = torch.from_numpy(x_np).to(device)
        y = torch.from_numpy(y_np).to(device)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()
        last_loss = 0.0
        for _step in range(self.steps_per_correction):
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            last_loss = float(loss.detach().cpu())
        model.eval()
        metadata = dict(metadata)
        metadata["reinforcement"] = {
            **dict(metadata.get("reinforcement") or {}),
            "enabled": True,
            "session_id": self.session_id,
            "route": self.route,
            "correction_count": self.correction_count,
            "lr": self.lr,
            "steps_per_correction": self.steps_per_correction,
            "last_corrected_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        metadata["correction_count"] = self.correction_count
        paths = gm.artifact_paths(self.resolved_variant, model_dir=self.model_root, schema=self.schema)
        paths["weights"].parent.mkdir(parents=True, exist_ok=True)
        labels_json = {str(idx): label for idx, label in labels.items()}
        torch.save({"model_state": model.state_dict(), "metadata": metadata, "labels": labels_json}, paths["weights"])
        paths["metadata"].write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        paths["labels"].write_text(json.dumps(labels_json, indent=2), encoding="utf-8")
        self.metadata = metadata
        self.labels = labels
        return {
            "loss": last_loss,
            "checkpoint": str(paths["weights"]),
        }

    def _write_session_finished(self) -> None:
        meta_path = self.session_dir / "session_metadata.json"
        try:
            data = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        except Exception:
            data = {}
        data.update(
            {
                "finished_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "correction_count": self.correction_count,
                "last_backup_dir": str(self.last_backup_dir) if self.last_backup_dir else "",
            }
        )
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
