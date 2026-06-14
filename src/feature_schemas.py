"""Feature schema registry for BISINDO datasets and GRU checkpoints.

v10 policy:
- ``--schema all`` now means every maintained schema, including
  smart180_face1584. Use ``base`` / ``original`` for the original three schemas
  only: smart180, khukuh1629, adi1662.
- Khukuh 1629-D and Adi 1662-D already include MediaPipe face landmarks.
- Add only one extra face schema: smart180_face1584 = smart180 + face468 xyz.
  It is opt-in via ``--schema smart180_face1584`` / ``--schema face`` /
  ``--schema full`` so old recording commands do not suddenly get heavier.
"""

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
    uses_face: bool = False
    base_schema: str = ""
    notes: str = ""


BASE_SCHEMA_NAMES = ("smart180", "khukuh1629", "adi1662")
# Face-reference schemas: smart180 hands + minimal face landmarks used purely as a
# *position reference* for the hands (see src/face_reference.py).
FACE_REF_SCHEMA_NAMES = (
    "smart180_mouthdyn214",
    "smart180_mouthstat206",
    "smart180_handface220",
    "smart180_handface_vel286",
)
EXTRA_SCHEMA_NAMES = ("smart180_face1584",) + FACE_REF_SCHEMA_NAMES
FULL_SCHEMA_NAMES = BASE_SCHEMA_NAMES + EXTRA_SCHEMA_NAMES
FACE_ENABLED_SCHEMA_NAMES = ("smart180_face1584", "khukuh1629", "adi1662") + FACE_REF_SCHEMA_NAMES


SCHEMAS: dict[str, FeatureSchema] = {
    "smart180": FeatureSchema(
        name="smart180",
        display_name="Smart V8 180-D (hands + shoulders, no face)",
        feature_schema=sc.FEATURE_SCHEMA,
        feature_mode=sc.FEATURE_MODE,
        feature_dim=sc.FEATURE_DIM,
        dataset_subdir="smart180",
        model_subdir="smart180",
        extractor="smart_v8",
        target_fps=sc.TARGET_FPS,
        uses_face=False,
        base_schema="smart180",
        notes="Fast compact Smart V8 feature: shoulders + selected hand points + local geometry + angles + metadata.",
    ),
    "khukuh1629": FeatureSchema(
        name="khukuh1629",
        display_name="Khukuh Holistic 1629-D (includes face)",
        feature_schema="bisindo_khukuh_holistic_1629_10fps",
        feature_mode="khukuh_holistic_1629",
        feature_dim=1629,
        dataset_subdir="khukuh1629",
        model_subdir="khukuh1629",
        extractor="holistic",
        target_fps=10.0,
        uses_face=True,
        base_schema="khukuh1629",
        notes="Right hand xyz + left hand xyz + pose xyz + face xyz. Face is already part of the schema.",
    ),
    "adi1662": FeatureSchema(
        name="adi1662",
        display_name="Adi Holistic 1662-D (includes face)",
        feature_schema="bisindo_adi_holistic_1662_10fps",
        feature_mode="adi_holistic_1662",
        feature_dim=1662,
        dataset_subdir="adi1662",
        model_subdir="adi1662",
        extractor="holistic",
        target_fps=10.0,
        uses_face=True,
        base_schema="adi1662",
        notes="Pose xyzw + face xyz + left hand xyz + right hand xyz. Face is already part of the schema.",
    ),
    "smart180_face1584": FeatureSchema(
        name="smart180_face1584",
        display_name="Smart V8 180-D + Face 1404-D = 1584-D",
        feature_schema="bisindo_smart_v8_180_plus_face468xyz_1584_10fps",
        feature_mode="smart180_plus_face468xyz",
        feature_dim=1584,
        dataset_subdir="smart180_face1584",
        model_subdir="smart180_face1584",
        extractor="holistic",
        target_fps=10.0,
        uses_face=True,
        base_schema="smart180",
        notes="Opt-in feature: Smart180-compatible compact hand/shoulder vector plus MediaPipe face xyz landmarks.",
    ),
    "smart180_mouthdyn214": FeatureSchema(
        name="smart180_mouthdyn214",
        display_name="Smart180 + Mouth Dynamics = 214-D",
        feature_schema="bisindo_smart180_mouthdyn_214_10fps",
        feature_mode="smart180_mouthdyn",
        feature_dim=214,
        dataset_subdir="smart180_mouthdyn214",
        model_subdir="smart180_mouthdyn214",
        extractor="holistic",
        target_fps=10.0,
        uses_face=True,
        base_schema="smart180",
        notes="Smart180 + 9 face points (rel) + mouth openness/width/aspect + eye_dist + mouth velocity (per detik) + face_present. Stateful (single worker).",
    ),
    "smart180_mouthstat206": FeatureSchema(
        name="smart180_mouthstat206",
        display_name="Smart180 + Mouth Static = 206-D",
        feature_schema="bisindo_smart180_mouthstat_206_10fps",
        feature_mode="smart180_mouthstat",
        feature_dim=206,
        dataset_subdir="smart180_mouthstat206",
        model_subdir="smart180_mouthstat206",
        extractor="holistic",
        target_fps=10.0,
        uses_face=True,
        base_schema="smart180",
        notes="Smart180 + nose + mouth line (corner kiri/kanan/center) + 4 sudut mata + eye_dist + face_present. Stateless.",
    ),
    "smart180_handface220": FeatureSchema(
        name="smart180_handface220",
        display_name="Smart180 + Hand-Face Relations = 220-D",
        feature_schema="bisindo_smart180_handface_220_10fps",
        feature_mode="smart180_handface",
        feature_dim=220,
        dataset_subdir="smart180_handface220",
        model_subdir="smart180_handface220",
        extractor="holistic",
        target_fps=10.0,
        uses_face=True,
        base_schema="smart180",
        notes="Smart180 + face anchor rel (nose/mouth/eye) + vektor wrist/palm->wajah per tangan + jarak skalar + flag. Wajah sebagai penanda posisi tangan. Stateless.",
    ),
    "smart180_handface_vel286": FeatureSchema(
        name="smart180_handface_vel286",
        display_name="Smart180 + Hand-Face + Velocity = 286-D",
        feature_schema="bisindo_smart180_handface_vel_286_10fps",
        feature_mode="smart180_handface_vel",
        feature_dim=286,
        dataset_subdir="smart180_handface_vel286",
        model_subdir="smart180_handface_vel286",
        extractor="holistic",
        target_fps=10.0,
        uses_face=True,
        base_schema="smart180",
        notes="Blok handface220 + velocity per detik slice global kiri/kanan smart180. Stateful (single worker).",
    ),
}
SCHEMA_NAMES = tuple(SCHEMAS.keys())


SMART180_SELECTED_POINTS = (
    "wrist",
    "palm_center",
    "thumb_tip",
    "index_mcp",
    "index_tip",
    "middle_mcp",
    "middle_tip",
    "ring_mcp",
    "ring_tip",
    "pinky_mcp",
    "pinky_tip",
)
SMART180_META_NAMES = (
    "left_present",
    "right_present",
    "left_detected",
    "right_detected",
    "left_held",
    "right_held",
    "shoulder_ok",
    "shoulder_scale",
    "left_score",
    "right_score",
)
ANGLE_NAMES = (
    "thumb_cmc",
    "thumb_mcp",
    "thumb_ip",
    "thumb_tip_chain",
    "index_mcp",
    "index_pip",
    "index_dip",
    "middle_mcp",
    "middle_pip",
    "middle_dip",
    "ring_mcp",
    "ring_pip",
    "ring_dip",
    "pinky_mcp",
    "pinky_pip",
    "pinky_dip",
)


def normalize_schema_name(schema: str | None = None) -> str:
    value = str(schema or DEFAULT_SCHEMA).strip().lower().replace("-", "_")
    aliases = {
        "default": DEFAULT_SCHEMA,
        "smart": "smart180",
        "v8": "smart180",
        "180": "smart180",
        "khukuh": "khukuh1629",
        "khukuh_face": "khukuh1629",
        "khukuh1629_face": "khukuh1629",
        "1629": "khukuh1629",
        "1629_face": "khukuh1629",
        "adi": "adi1662",
        "adhi": "adi1662",
        "adi_face": "adi1662",
        "adhi_face": "adi1662",
        "adi1662_face": "adi1662",
        "1662": "adi1662",
        "1662_face": "adi1662",
        "smart_face": "smart180_face1584",
        "smart180_face": "smart180_face1584",
        "smart180face": "smart180_face1584",
        "180_face": "smart180_face1584",
        "180face": "smart180_face1584",
        "face180": "smart180_face1584",
        "compact_face": "smart180_face1584",
        "mouthdyn": "smart180_mouthdyn214",
        "smart180_mouthdyn": "smart180_mouthdyn214",
        "mouthdyn214": "smart180_mouthdyn214",
        "mouthstat": "smart180_mouthstat206",
        "smart180_mouthstat": "smart180_mouthstat206",
        "mouthstat206": "smart180_mouthstat206",
        "handface": "smart180_handface220",
        "smart180_handface": "smart180_handface220",
        "handface220": "smart180_handface220",
        "handface_vel": "smart180_handface_vel286",
        "handfacevel": "smart180_handface_vel286",
        "smart180_handface_vel": "smart180_handface_vel286",
        "handface_vel286": "smart180_handface_vel286",
    }
    value = aliases.get(value, value)
    if value not in SCHEMAS:
        raise ValueError(f"Unknown feature schema '{schema}'. Pilih: {', '.join(SCHEMA_NAMES)}")
    return value


def _expand_schema_token(value: str) -> tuple[str, ...]:
    if value in {"base", "original", "originals", "asli", "ketiganya"}:
        return BASE_SCHEMA_NAMES
    # faceref = only the 4 smart180 + face-position-reference schemas.
    if value in {"faceref", "face_ref", "facerefs", "wajah_ref", "smart_faceref"}:
        return FACE_REF_SCHEMA_NAMES
    # face = the optional 180+face schema plus the 4 face-reference schemas.
    # Adi/Khukuh already contain face and stay in `base`/`all`.
    if value in {"face", "faces", "wajah", "extra", "extras", "new", "tambahan"}:
        return EXTRA_SCHEMA_NAMES
    # all/full explicitly includes every maintained schema.
    if value in {"all", "full", "all_face", "all_with_face", "semua", "semua_plus_face", "base_plus_face"}:
        return FULL_SCHEMA_NAMES
    return (normalize_schema_name(value),)


def expand_schema_names(schema: str | Iterable[str] | None = None) -> tuple[str, ...]:
    raw_values: list[str | None]
    if schema is None or isinstance(schema, str):
        raw_values = [schema]
    else:
        raw_values = list(schema)
    if not raw_values:
        raw_values = [DEFAULT_SCHEMA]

    names: list[str] = []
    for raw in raw_values:
        text = str(raw or DEFAULT_SCHEMA).strip().lower().replace("-", "_").replace(";", ",")
        for part in (item.strip() for item in text.split(",")):
            if not part:
                continue
            for name in _expand_schema_token(part):
                if name not in names:
                    names.append(name)
    return tuple(names or (DEFAULT_SCHEMA,))


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


def _axis_names(prefix: str, count: int, axes: tuple[str, ...]) -> list[str]:
    return [f"{prefix}_{idx:03d}_{axis}" for idx in range(count) for axis in axes]


def smart180_feature_column_names() -> list[str]:
    names: list[str] = []
    names.extend([f"shoulder_{side}_{axis}" for side in ("left", "right") for axis in ("x", "y", "z")])
    for hand in ("left", "right"):
        for point in SMART180_SELECTED_POINTS:
            for axis in ("global_x", "global_y", "global_z"):
                names.append(f"{hand}_{point}_{axis}")
    for hand in ("left", "right"):
        for point in SMART180_SELECTED_POINTS:
            for axis in ("local_x", "local_y", "local_z"):
                names.append(f"{hand}_{point}_{axis}")
    for hand in ("left", "right"):
        names.extend([f"{hand}_angle_{name}" for name in ANGLE_NAMES])
    names.extend([f"meta_{name}" for name in SMART180_META_NAMES])
    return names


def face_feature_column_names(prefix: str = "face") -> list[str]:
    return _axis_names(prefix, 468, ("x", "y", "z"))


def feature_column_names(schema: str | FeatureSchema | None = None) -> list[str]:
    spec = get_schema(schema)
    if spec.name == "smart180":
        return smart180_feature_column_names()
    if spec.name == "smart180_face1584":
        return smart180_feature_column_names() + face_feature_column_names()
    if spec.name in FACE_REF_SCHEMA_NAMES:
        import face_reference as fr  # lazy: avoids pulling cv2/mediapipe for light imports
        return smart180_feature_column_names() + fr.face_block_column_names(spec.name)
    if spec.name == "khukuh1629":
        return (
            _axis_names("right_hand", 21, ("x", "y", "z"))
            + _axis_names("left_hand", 21, ("x", "y", "z"))
            + _axis_names("pose", 33, ("x", "y", "z"))
            + face_feature_column_names()
        )
    if spec.name == "adi1662":
        return (
            _axis_names("pose", 33, ("x", "y", "z", "visibility"))
            + face_feature_column_names()
            + _axis_names("left_hand", 21, ("x", "y", "z"))
            + _axis_names("right_hand", 21, ("x", "y", "z"))
        )
    return [f"f{idx}" for idx in range(spec.feature_dim)]


def full_raw_feature_column_names() -> list[str]:
    return (
        _axis_names("right_hand", 21, ("x", "y", "z"))
        + _axis_names("left_hand", 21, ("x", "y", "z"))
        + _axis_names("pose", 33, ("x", "y", "z", "visibility"))
        + face_feature_column_names()
        + [f"shoulder_{side}_{axis}" for side in ("left", "right") for axis in ("x", "y", "z", "visibility")]
    )


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

    if spec.base_schema == "smart180":
        # smart180 occupies [0:180] for every smart180-based schema, so the meta
        # slice [170:180] is identical regardless of any appended face block.
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
    if spec.base_schema == "smart180":
        # Motion is read from the smart180 hand slices ([6:72]) shared by every
        # smart180-based schema; the appended face block does not affect it.
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
