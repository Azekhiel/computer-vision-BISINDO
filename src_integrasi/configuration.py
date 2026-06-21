"""Loader configuration.json untuk integrasi.

Membaca src_integrasi/configuration.json dan menyusun AssistantConfig dari knob:
schema, model, suite, augmentasi (bool), specialist, threshold, dan llm. Augmentasi
dipisah dari model: kalau true, varian otomatis jadi "<model>_dengan_augmentasi".

Tidak menyentuh logika kamera/live sama sekali — hanya menentukan schema/varian/route yang
dipakai. Kalau file tidak ada / rusak / field kosong, fallback ke default AssistantConfig
(tidak pernah crash). Nilai tidak valid memunculkan pesan error yang jelas.

Lihat src_integrasi/isi_konfiguration.md untuk daftar lengkap nilai valid.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
for _base in (ROOT_DIR, ROOT_DIR / "src", ROOT_DIR / "LLM"):
    _path = str(_base)
    if _path not in sys.path:
        sys.path.insert(0, _path)

import feature_schemas as fs  # noqa: E402
import gru_manager as gm  # noqa: E402
import bisindo_live_assistant as live_assistant  # noqa: E402

CONFIG_PATH = Path(__file__).resolve().parent / "configuration.json"


def load_config_dict(path: Path | None = None) -> dict[str, Any]:
    """Baca configuration.json mentah. Return {} kalau tidak ada / tidak bisa di-parse."""
    target = Path(path) if path is not None else CONFIG_PATH
    if not target.exists():
        return {}
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def resolve_overrides(raw: dict[str, Any]) -> dict[str, Any]:
    """Validasi knob dan ubah jadi override AssistantConfig (live_schema/live_variant/live_route).

    Field yang tidak diisi (None / kosong) dilewati supaya default AssistantConfig dipakai.
    Field tidak valid -> ValueError dengan daftar pilihan yang benar.
    """
    overrides: dict[str, Any] = {}

    schema = raw.get("schema")
    if schema not in (None, ""):
        resolved = fs.normalize_schema_name(schema)
        if resolved not in fs.SCHEMA_NAMES:
            raise ValueError(
                f"Schema '{schema}' tidak dikenal. Pilih: {', '.join(fs.SCHEMA_NAMES)}"
            )
        overrides["live_schema"] = resolved

    model = raw.get("model")
    augmentasi = raw.get("augmentasi", None)
    if model not in (None, ""):
        base = gm.base_variant_name(model)  # raises ValueError kalau bukan base variant valid
        use_aug = bool(augmentasi) if augmentasi is not None else False
        overrides["live_variant"] = gm.augmented_variant_name(base) if use_aug else base

    suite = raw.get("suite")
    if suite not in (None, ""):
        overrides["live_route"] = gm.normalize_eval_suite_name(suite)

    specialist = raw.get("specialist")
    if specialist is not None:
        text = str(specialist).strip().lower()
        if specialist is False or text in {"off", "none", "false", "no", "0"}:
            overrides["specialist_enabled"] = False
        elif text in {"", "all", "*", "semua"}:
            if text != "":  # kosong -> skip (pakai default), "all" -> eksplisit
                overrides["specialist_enabled"] = True
                overrides["specialist_name"] = "all"
        else:
            # Satu atau beberapa nama specialist (dipisah koma) -> normalisasi + dedup.
            names: list[str] = []
            for token in str(specialist).split(","):
                token = token.strip()
                if not token:
                    continue
                normalized = gm.normalize_specialist_name(token)
                if normalized not in names:
                    names.append(normalized)
            if not names:
                raise ValueError("Field 'specialist' tidak menghasilkan nama yang valid.")
            overrides["specialist_enabled"] = True
            overrides["specialist_name"] = ",".join(names)

    threshold = raw.get("threshold")
    if threshold is not None and str(threshold).strip().lower() not in {"", "default"}:
        try:
            value = float(threshold)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Field 'threshold' harus angka 0-1 atau 'default', dapat: {threshold!r}") from exc
        if not (0.0 < value <= 1.0):
            raise ValueError(f"Field 'threshold' harus dalam rentang (0, 1], dapat: {value}")
        overrides["confidence_threshold"] = value

    llm = raw.get("llm")
    if llm is not None and str(llm).strip().lower() not in {"", "default"}:
        overrides["llm_model"] = str(llm).strip()

    return overrides


def load_configuration(path: Path | None = None, **extra: Any) -> live_assistant.AssistantConfig:
    """Bangun AssistantConfig dari configuration.json.

    `extra` (mis. dari CLI args) selalu menang atas isi file, supaya argumen runtime tetap override.
    """
    overrides = resolve_overrides(load_config_dict(path))
    overrides.update(extra)
    return live_assistant.AssistantConfig(**overrides)
