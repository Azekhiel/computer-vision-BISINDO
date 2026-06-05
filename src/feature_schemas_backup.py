"""Feature schema registry for BISINDO datasets and GRU checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from smart_extract import contract as sc


ROOT_DIR = Path(__file__).resolve().parents[1]
DATASET_ROOT = ROOT_DIR / "dataset_parquets"
MODEL_ROOT = ROOT_DIR / "models"
DEFAULT_SCHEMA = "smart180"


@dataclass(frozen=True)
class FeatureSchema:
    name: str
    display_name: str
    feature_schema: str
    feature_mode: str
    feature_dim: int
    dataset_subdir: str
    model_subdir: str
    extractor: str
    target_fps: float = 10.0


SCHEMAS: dict[str, FeatureSchema] = {
    "smart180": FeatureSchema(
        name="smart180",
        display_name="Smart V8 180-D",
        feature_schema=sc.FEATURE_SCHEMA,
        feature_mode=sc.FEATURE_MODE,
        feature_dim=sc.FEATURE_DIM,
        dataset_subdir="smart180",
        model_subdir="smart180",
        extractor="smart_v8",
        target_fps=sc.TARGET_FPS,
    ),
    "khukuh1629": FeatureSchema(
        name="khukuh1629",
        display_name="Khukuh Holistic 1629-D",
        feature_schema="bisindo_khukuh_holistic_1629_10fps",
        feature_mode="khukuh_holistic_1629",
        feature_dim=1629,
        dataset_subdir="khukuh1629",
        model_subdir="khukuh1629",
        extractor="holistic",
        target_fps=10.0,
    ),
    "adi1662": FeatureSchema(
        name="adi1662",
        display_name="Adi Holistic 1662-D",
        feature_schema="bisindo_adi_holistic_1662_10fps",
        feature_mode="adi_holistic_1662",
        feature_dim=1662,
        dataset_subdir="adi1662",
        model_subdir="adi1662",
        extractor="holistic",
        target_fps=10.0,
    ),
}
SCHEMA_NAMES = tuple(SCHEMAS.keys())


def normalize_schema_name(schema: str | None = None) -> str:
    value = str(schema or DEFAULT_SCHEMA).strip().lower().replace("-", "_")
    aliases = {
        "default": DEFAULT_SCHEMA,
        "smart": "smart180",
        "v8": "smart180",
        "180": "smart180",
        "khukuh": "khukuh1629",
        "1629": "khukuh1629",
        "adi": "adi1662",
        "1662": "adi1662",
    }
    value = aliases.get(value, value)
    if value not in SCHEMAS:
        raise ValueError(f"Unknown feature schema '{schema}'. Pilih: {', '.join(SCHEMA_NAMES)}")
    return value


def expand_schema_names(schema: str | None = None) -> tuple[str, ...]:
    value = str(schema or DEFAULT_SCHEMA).strip().lower()
    if value == "all":
        return SCHEMA_NAMES
    return (normalize_schema_name(value),)


def get_schema(schema: str | FeatureSchema | None = None) -> FeatureSchema:
    if isinstance(schema, FeatureSchema):
        return schema
    return SCHEMAS[normalize_schema_name(schema)]


def dataset_dir_for(schema: str | FeatureSchema | None = None, dataset_root: str | Path = DATASET_ROOT) -> Path:
    spec = get_schema(schema)
    root = Path(dataset_root)
    if root.name == spec.dataset_subdir:
        return root
    return root / spec.dataset_subdir


def dataset_parquet_paths(
    schema: str | FeatureSchema | None = None,
    dataset_root: str | Path = DATASET_ROOT,
    include_legacy_smart180: bool = True,
) -> list[Path]:
    spec = get_schema(schema)
    root = Path(dataset_root)
    paths: list[Path] = []
    schema_dir = dataset_dir_for(spec, root)
    if schema_dir.exists():
        paths.extend(sorted(schema_dir.glob("*.parquet")))
    direct_paths = sorted(root.glob("*.parquet")) if root.exists() else []
    if root.name == spec.dataset_subdir:
        paths.extend(path for path in direct_paths if path not in paths)
    elif spec.name == "smart180" and include_legacy_smart180:
        paths.extend(path for path in direct_paths if path not in paths)
    elif not paths:
        paths.extend(direct_paths)
    return paths


def model_dir_for(schema: str | FeatureSchema | None = None, model_root: str | Path = MODEL_ROOT) -> Path:
    spec = get_schema(schema)
    root = Path(model_root)
    if root.name == spec.model_subdir:
        return root
    if root.name == "gru":
        return root / spec.model_subdir
    return root / "gru" / spec.model_subdir


def parse_feature_value(value) -> np.ndarray:
    return sc.parse_feature_value(value)


def format_feature_value(features: Iterable[float]) -> str:
    return sc.format_feature_value(features)


def ensure_feature_dim(sequence, schema: str | FeatureSchema | int | None = None) -> np.ndarray:
    expected_dim = int(schema if isinstance(schema, int) else get_schema(schema).feature_dim)
    return sc.ensure_feature_dim(sequence, expected_dim)


def filter_feature_rows(df: pd.DataFrame, schema: str | FeatureSchema | None = None) -> pd.DataFrame:
    spec = get_schema(schema)
    if "feature_version" not in df.columns:
        return df.iloc[0:0].copy()
    if "feature_dim" in df.columns:
        dims = df["feature_dim"]
        try:
            dims = dims.astype(int)
        except Exception:
            def _dim(value):
                try:
                    return int(float(value))
                except Exception:
                    return -1
            dims = dims.apply(_dim)
        return df[(df["feature_version"] == spec.feature_schema) & (dims == spec.feature_dim)].copy()
    return df[df["feature_version"] == spec.feature_schema].copy()


def sample_gif_paths(
    schema: str | FeatureSchema,
    vocab: str,
    video_id: str,
    root_dir: str | Path,
    modes: Iterable[str] = ("overlay", "skeleton"),
) -> dict[str, str]:
    spec = get_schema(schema)
    safe_video_id = "".join(c if c.isalnum() or c in "._-" else "_" for c in str(video_id))
    base = Path(root_dir) / "assets" / "gifs" / "samples" / spec.name / str(vocab)
    return {mode: str(base / f"{safe_video_id}_{mode}.gif") for mode in modes}


def presence_from_vector(schema: str | FeatureSchema, vector: np.ndarray) -> dict[str, float | bool]:
    spec = get_schema(schema)
    vec = np.asarray(vector, dtype=np.float32).reshape(-1)
    if vec.shape[0] < spec.feature_dim:
        return {"left_present": 0.0, "right_present": 0.0, "shoulder_ok": False, "visible": False}

    if spec.name == "smart180":
        meta = vec[sc.SLICE_META]
        left = float(meta[sc.IDX_META_LEFT_PRESENT])
        right = float(meta[sc.IDX_META_RIGHT_PRESENT])
        shoulder = bool(meta[sc.IDX_META_SHOULDER_OK] >= 0.5)
    elif spec.name == "khukuh1629":
        right_chunk = vec[0:63]
        left_chunk = vec[63:126]
        pose_chunk = vec[126:225]
        left = float(np.linalg.norm(left_chunk) > 1e-6)
        right = float(np.linalg.norm(right_chunk) > 1e-6)
        shoulder = bool(np.linalg.norm(pose_chunk) > 1e-6)
    else:
        pose_chunk = vec[0:132]
        left_chunk = vec[1536:1599]
        right_chunk = vec[1599:1662]
        left = float(np.linalg.norm(left_chunk) > 1e-6)
        right = float(np.linalg.norm(right_chunk) > 1e-6)
        shoulder = bool(np.linalg.norm(pose_chunk) > 1e-6)

    return {
        "left_present": left,
        "right_present": right,
        "shoulder_ok": shoulder,
        "visible": bool(left >= 0.5 or right >= 0.5),
    }


def motion_score(
    schema: str | FeatureSchema,
    prev: np.ndarray | None,
    curr: np.ndarray | None,
) -> tuple[float, bool]:
    spec = get_schema(schema)
    if curr is None:
        return 0.0, False
    if spec.name == "smart180":
        return sc.motion_score(prev, curr)

    curr_vec = ensure_feature_dim(curr, spec)[0]
    present = presence_from_vector(spec, curr_vec)
    visible = bool(present["visible"])
    if prev is None:
        return 0.0, visible
    prev_vec = ensure_feature_dim(prev, spec)[0]
    if spec.name == "khukuh1629":
        chunks = [(63, 126), (0, 63)]
    else:
        chunks = [(1536, 1599), (1599, 1662)]
    scores = []
    for start, stop in chunks:
        if np.linalg.norm(curr_vec[start:stop]) > 1e-6:
            scores.append(float(np.linalg.norm(curr_vec[start:stop] - prev_vec[start:stop]) / np.sqrt(stop - start)))
    return (float(max(scores)) if scores else 0.0), visible
