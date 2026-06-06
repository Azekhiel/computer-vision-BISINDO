import json
import os
import sys
from pathlib import Path

import numpy as np
import torch


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import feature_schemas as fs
import gru_manager as gm
import reinforcement_learning as rl


def _write_base_checkpoint(model_dir: Path, schema: str = "smart180", variant: str = "adi") -> dict[str, Path]:
    schema_spec = fs.get_schema(schema)
    paths = gm.artifact_paths(variant, model_dir=model_dir, schema=schema)
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    labels = {"0": "aku", "1": "kamu"}
    model = gm.build_model(variant, input_dim=schema_spec.feature_dim, num_classes=len(labels))
    metadata = {
        "variant": variant,
        "schema": schema,
        "feature_schema": schema_spec.feature_schema,
        "feature_dim": schema_spec.feature_dim,
        "target_frames": gm.variant_spec(variant).target_frames,
        "num_classes": len(labels),
        "labels": labels,
        "best_val_acc": 0.9,
    }
    torch.save({"model_state": model.state_dict(), "metadata": metadata, "labels": labels}, paths["weights"])
    paths["labels"].write_text(json.dumps(labels, indent=2), encoding="utf-8")
    paths["metadata"].write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return paths


def test_reinforcement_session_clones_updates_copy_and_backs_up_without_touching_base(tmp_path):
    base_model_dir = tmp_path / "models"
    model_root = tmp_path / "model_reinforcement"
    backup_root = tmp_path / "backups"
    base_paths = _write_base_checkpoint(base_model_dir)
    original_base_weights = base_paths["weights"].read_bytes()

    session = rl.ReinforcementSession(
        schema="smart180",
        variant="adi",
        route="main",
        device="cpu",
        base_model_dir=base_model_dir,
        model_root=model_root,
        backup_root=backup_root,
        lr=1e-4,
        steps_per_correction=1,
    )
    info = session.start()

    assert info["variant"] == "adi"
    rl_paths = gm.artifact_paths("adi", model_dir=model_root, schema="smart180")
    assert rl_paths["weights"].exists()
    assert base_paths["weights"].read_bytes() == original_base_weights

    sequence = np.zeros((4, fs.get_schema("smart180").feature_dim), dtype=np.float32)
    result = session.apply_correction(
        sequence=sequence,
        corrected_label="kamu",
        predicted_label="aku",
        confidence=0.42,
        prediction_id=7,
    )

    assert result["correction_count"] == 1
    assert session.corrections_jsonl.exists()
    record = json.loads(session.corrections_jsonl.read_text(encoding="utf-8").strip())
    assert record["predicted_label"] == "aku"
    assert record["corrected_label"] == "kamu"
    assert Path(record["sequence_npz"]).exists()
    assert base_paths["weights"].read_bytes() == original_base_weights
    assert json.loads(rl_paths["metadata"].read_text(encoding="utf-8"))["correction_count"] == 1

    backup_dir = session.finish()

    assert backup_dir.parent.name.startswith("cleanup_")
    assert (backup_dir / "gru" / "smart180" / "gru_adi.pth").exists()
    assert (backup_dir / "sessions" / session.session_id / "corrections.jsonl").exists()
    assert base_paths["weights"].read_bytes() == original_base_weights
