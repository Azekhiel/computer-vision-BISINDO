"""Convert archived full MediaPipe landmark NPZ files into GRU parquet datasets."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path
import shutil
from typing import Any, Iterable

import numpy as np
import pandas as pd

import face_reference as fr
import feature_schemas as fs
from smart_extract import contract as sc
from smart_extract.live_bisindo_mp_real_shoulder_v6 import build_feature


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT_DIR / "dataset_full_mediapipe"
DEFAULT_DATASET_DIR = ROOT_DIR / "dataset_parquets"
DEFAULT_MODEL_DIR = ROOT_DIR / "models"
DEFAULT_BACKUP_ROOT = ROOT_DIR / "backups"
SPLITS = {"train", "val", "test"}
MASK_COLUMNS = (
    "pose_landmarks",
    "pose_world_landmarks",
    "left_hand_landmarks",
    "right_hand_landmarks",
    "face_landmarks",
    "face_blendshapes",
    "facial_transformation_matrixes",
    "left_handedness",
    "right_handedness",
)


@dataclass(frozen=True)
class FullMediaPipeItem:
    npz_path: Path
    report_path: Path | None
    split: str
    label: str
    video_id: str


@dataclass
class FullExtractResult:
    scanned: int = 0
    converted: int = 0
    failed: int = 0
    rows: int = 0
    labels: list[str] = field(default_factory=list)
    schemas: list[str] = field(default_factory=list)
    backup_dir: Path | None = None
    manifest_path: Path | None = None
    errors: list[dict[str, str]] = field(default_factory=list)
    counts: dict[str, dict[str, dict[str, int]]] = field(default_factory=dict)


def clean_label(value: str) -> str:
    return str(value or "").strip().replace(" ", "_").lower()


def _normalize_vocab_filter(vocab: str | Iterable[str] | None = None) -> tuple[str, ...]:
    if vocab is None or isinstance(vocab, str):
        raw_values = [vocab]
    else:
        raw_values = list(vocab)

    labels: list[str] = []
    for raw in raw_values:
        text = str(raw or "").strip().replace(";", ",")
        for part in (item.strip() for item in text.split(",")):
            label = clean_label(part)
            if label and label not in labels:
                labels.append(label)
    return tuple(labels)


def _feature_root(source: str | Path) -> Path:
    root = Path(source).expanduser()
    if (root / "features_holistic" / "fps_10").exists():
        return root / "features_holistic" / "fps_10"
    if root.name == "fps_10" and root.exists():
        return root
    return root


def discover_full_mediapipe_items(source: str | Path = DEFAULT_SOURCE) -> list[FullMediaPipeItem]:
    root = _feature_root(source)
    items: list[FullMediaPipeItem] = []
    for npz_path in sorted(root.rglob("*.landmarks.npz")):
        try:
            rel = npz_path.relative_to(root)
        except ValueError:
            continue
        if len(rel.parts) < 3:
            continue
        split = rel.parts[0].lower()
        if split not in SPLITS:
            continue
        label = clean_label(rel.parts[1])
        video_id = npz_path.name.replace(".landmarks.npz", "")
        report_path = Path(str(npz_path).replace(".landmarks.npz", ".report.json"))
        items.append(
            FullMediaPipeItem(
                npz_path=npz_path,
                report_path=report_path if report_path.exists() else None,
                split=split,
                label=label,
                video_id=video_id,
            )
        )
    return items


def _scalar(value: Any, default: Any = "") -> Any:
    arr = np.asarray(value)
    if arr.size == 0:
        return default
    try:
        return arr.reshape(-1)[0].item()
    except Exception:
        return arr.reshape(-1)[0]


def _mask_at(masks: np.ndarray | None, frame_idx: int, name: str, fallback: bool = True) -> bool:
    if masks is None or masks.ndim != 2:
        return bool(fallback)
    try:
        col = MASK_COLUMNS.index(name)
    except ValueError:
        return bool(fallback)
    if frame_idx >= masks.shape[0] or col >= masks.shape[1]:
        return bool(fallback)
    return bool(masks[frame_idx, col])


def _xyz_frame(arr: np.ndarray, frame_idx: int, count: int, available: bool) -> np.ndarray:
    out = np.zeros((count, 3), dtype=np.float32)
    if not available or arr.ndim < 3 or frame_idx >= arr.shape[0]:
        return out
    take = min(count, arr.shape[1])
    dims = min(3, arr.shape[2])
    out[:take, :dims] = arr[frame_idx, :take, :dims].astype(np.float32, copy=False)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _pose_xyzw_frame(arr: np.ndarray, frame_idx: int, available: bool) -> np.ndarray:
    out = np.zeros((33, 4), dtype=np.float32)
    if not available or arr.ndim < 3 or frame_idx >= arr.shape[0]:
        return out
    take = min(33, arr.shape[1])
    dims = min(4, arr.shape[2])
    out[:take, :dims] = arr[frame_idx, :take, :dims].astype(np.float32, copy=False)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _shoulders_from_pose_xyzw(pose_xyzw: np.ndarray, available: bool) -> np.ndarray:
    if not available or pose_xyzw.shape[0] < 13:
        return np.full((2, 4), np.nan, dtype=np.float32)
    return pose_xyzw[[11, 12], :4].astype(np.float32, copy=True)


def _score_frame(arr: np.ndarray, frame_idx: int, available: bool) -> float:
    if not available or arr.ndim < 2 or frame_idx >= arr.shape[0] or arr.shape[1] < 1:
        return 0.0
    return float(arr[frame_idx, 0])


def _frame_meta(npz: dict[str, np.ndarray], frame_idx: int, item: FullMediaPipeItem, report: dict[str, Any]) -> dict[str, Any]:
    timestamps = np.asarray(npz.get("timestamp_ms", []), dtype=np.float32)
    frame_index = np.asarray(npz.get("frame_index", []), dtype=np.int32)
    image_size = np.asarray(npz.get("image_size", []), dtype=np.int32)
    source_fps = float(_scalar(npz.get("source_fps", np.asarray([10.0], dtype=np.float32)), 10.0) or 10.0)
    time_sec = float(timestamps[frame_idx] / 1000.0) if frame_idx < len(timestamps) else float(frame_idx / max(source_fps, 1e-6))
    source_frame = int(frame_index[frame_idx]) if frame_idx < len(frame_index) else int(frame_idx)
    width = int(report.get("width", 0) or 0)
    height = int(report.get("height", 0) or 0)
    if image_size.ndim == 2 and frame_idx < image_size.shape[0] and image_size.shape[1] >= 2:
        width = int(image_size[frame_idx, 0])
        height = int(image_size[frame_idx, 1])
    return {
        "out_index": int(frame_idx),
        "time_sec": time_sec,
        "target_source_frame": source_frame,
        "chosen_source_frame": source_frame,
        "source_sample_index": int(frame_idx),
        "source_width": width,
        "source_height": height,
        "source_video_path": str(_scalar(npz.get("source_video_path", np.asarray([""])), "")),
        "sample_id": str(report.get("sample_id", "")),
        "mediapipe_version": str(report.get("mediapipe_version", "")),
        "source_npz_path": str(item.npz_path),
        "source_report_path": str(item.report_path or ""),
    }


def convert_npz_to_schema_sequences(
    npz_path: str | Path,
    schemas: Iterable[str],
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    """Convert one full MediaPipe NPZ into requested schema arrays."""

    schema_specs = [fs.get_schema(schema) for schema in schemas]
    with np.load(Path(npz_path), allow_pickle=True) as loaded:
        npz = {key: loaded[key] for key in loaded.files}

    frame_count = int(np.asarray(npz.get("frame_index", [])).shape[0])
    if frame_count <= 0:
        for key in ("pose_landmarks", "left_hand_landmarks", "right_hand_landmarks", "face_landmarks"):
            if key in npz and np.asarray(npz[key]).ndim >= 1:
                frame_count = int(np.asarray(npz[key]).shape[0])
                break
    if frame_count <= 0:
        raise ValueError(f"NPZ tanpa frame: {npz_path}")

    masks = np.asarray(npz["availability_masks"], dtype=bool) if "availability_masks" in npz else None
    pose_arr = np.asarray(npz.get("pose_landmarks", np.zeros((frame_count, 33, 5), dtype=np.float32)), dtype=np.float32)
    left_arr = np.asarray(npz.get("left_hand_landmarks", np.zeros((frame_count, 21, 5), dtype=np.float32)), dtype=np.float32)
    right_arr = np.asarray(npz.get("right_hand_landmarks", np.zeros((frame_count, 21, 5), dtype=np.float32)), dtype=np.float32)
    face_arr = np.asarray(npz.get("face_landmarks", np.zeros((frame_count, 468, 5), dtype=np.float32)), dtype=np.float32)
    left_handed = np.asarray(npz.get("left_handedness", np.zeros((frame_count, 2), dtype=np.float32)), dtype=np.float32)
    right_handed = np.asarray(npz.get("right_handedness", np.zeros((frame_count, 2), dtype=np.float32)), dtype=np.float32)

    timestamps_ms = np.asarray(npz.get("timestamp_ms", []), dtype=np.float32)
    source_fps = float(_scalar(npz.get("source_fps", np.asarray([10.0], dtype=np.float32)), 10.0) or 10.0)
    # Per-schema temporal state for face-reference schemas (velocity history).
    face_states: dict[str, dict] = {
        spec.name: {} for spec in schema_specs if fr.is_face_ref_schema(spec.name)
    }

    out: dict[str, list[np.ndarray]] = {spec.name: [] for spec in schema_specs}
    metas: list[dict[str, Any]] = []
    dummy_item = FullMediaPipeItem(Path(npz_path), None, "train", "", Path(npz_path).name.replace(".landmarks.npz", ""))
    for frame_idx in range(frame_count):
        if frame_idx > 0 and frame_idx < len(timestamps_ms) and timestamps_ms[frame_idx] > timestamps_ms[frame_idx - 1]:
            dt = float((timestamps_ms[frame_idx] - timestamps_ms[frame_idx - 1]) / 1000.0)
        else:
            dt = 1.0 / max(source_fps, 1e-6)
        pose_ok = _mask_at(masks, frame_idx, "pose_landmarks", fallback=np.linalg.norm(pose_arr[frame_idx]) > 1e-6)
        left_ok = _mask_at(masks, frame_idx, "left_hand_landmarks", fallback=np.linalg.norm(left_arr[frame_idx]) > 1e-6)
        right_ok = _mask_at(masks, frame_idx, "right_hand_landmarks", fallback=np.linalg.norm(right_arr[frame_idx]) > 1e-6)
        face_ok = _mask_at(masks, frame_idx, "face_landmarks", fallback=np.linalg.norm(face_arr[frame_idx]) > 1e-6)

        pose_xyz = _xyz_frame(pose_arr, frame_idx, 33, pose_ok)
        pose_xyzw = _pose_xyzw_frame(pose_arr, frame_idx, pose_ok)
        left_xyz = _xyz_frame(left_arr, frame_idx, 21, left_ok)
        right_xyz = _xyz_frame(right_arr, frame_idx, 21, right_ok)
        face_xyz = _xyz_frame(face_arr, frame_idx, 468, face_ok)
        shoulders = _shoulders_from_pose_xyzw(pose_xyzw, pose_ok)
        present = np.array([float(left_ok), float(right_ok)], dtype=np.float32)
        detected = present.copy()
        held = np.zeros(2, dtype=np.float32)
        scores = np.array(
            [
                _score_frame(left_handed, frame_idx, _mask_at(masks, frame_idx, "left_handedness", fallback=False)) or float(left_ok),
                _score_frame(right_handed, frame_idx, _mask_at(masks, frame_idx, "right_handedness", fallback=False)) or float(right_ok),
            ],
            dtype=np.float32,
        )

        for spec in schema_specs:
            if spec.name == "smart180":
                vector = build_feature(
                    sc.FEATURE_MODE,
                    left_xyz if left_ok else None,
                    right_xyz if right_ok else None,
                    shoulders,
                    present,
                    detected,
                    held,
                    scores,
                )
            elif spec.name == "smart268":
                vector = build_feature(
                    "268",
                    left_xyz if left_ok else None,
                    right_xyz if right_ok else None,
                    shoulders,
                    present,
                    detected,
                    held,
                    scores,
                )
            elif spec.name == "khukuh1629":
                vector = np.concatenate((right_xyz.reshape(-1), left_xyz.reshape(-1), pose_xyz.reshape(-1), face_xyz.reshape(-1))).astype(np.float32)
            elif spec.name == "adi1662":
                vector = np.concatenate((pose_xyzw.reshape(-1), face_xyz.reshape(-1), left_xyz.reshape(-1), right_xyz.reshape(-1))).astype(np.float32)
            elif spec.name == "smart180_face1584":
                smart = build_feature(
                    sc.FEATURE_MODE,
                    left_xyz if left_ok else None,
                    right_xyz if right_ok else None,
                    shoulders,
                    present,
                    detected,
                    held,
                    scores,
                )
                vector = np.concatenate((smart, face_xyz.reshape(-1))).astype(np.float32)
            elif fr.is_face_ref_schema(spec.name):
                smart = build_feature(
                    sc.FEATURE_MODE,
                    left_xyz if left_ok else None,
                    right_xyz if right_ok else None,
                    shoulders,
                    present,
                    detected,
                    held,
                    scores,
                )
                vector = fr.build_face_schema_vector(
                    spec.name,
                    smart,
                    face_xyz,
                    shoulders,
                    left_xyz if left_ok else None,
                    right_xyz if right_ok else None,
                    face_present=bool(face_ok),
                    state=face_states[spec.name],
                    dt=dt,
                )
            else:
                raise ValueError(f"Schema tidak didukung converter full-mediapipe: {spec.name}")
            if vector.shape[0] != spec.feature_dim:
                raise RuntimeError(f"{spec.name}: dim {vector.shape[0]} != {spec.feature_dim}")
            out[spec.name].append(np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0))

        metas.append(
            {
                **_frame_meta(npz, frame_idx, dummy_item, {}),
                "enhance_mode": "full_mediapipe_archive",
                "left_present": float(left_ok),
                "right_present": float(right_ok),
                "left_detected": float(left_ok),
                "right_detected": float(right_ok),
                "left_held": 0.0,
                "right_held": 0.0,
                "left_score": float(scores[0]),
                "right_score": float(scores[1]),
                "pose_present": float(pose_ok),
                "face_present": float(face_ok),
            }
        )

    arrays = {name: fs.ensure_feature_dim(np.stack(rows).astype(np.float32), name) for name, rows in out.items()}
    return arrays, metas


def _rows_for_sequence(
    item: FullMediaPipeItem,
    schema: str,
    sequence: np.ndarray,
    metas: list[dict[str, Any]],
    report: dict[str, Any],
) -> list[dict[str, Any]]:
    spec = fs.get_schema(schema)
    rows: list[dict[str, Any]] = []
    source_fps = float(report.get("fps", 10.0) or 10.0)
    source_duration = float(report.get("duration_sec", 0.0) or 0.0)
    for frame_num, vec in enumerate(fs.ensure_feature_dim(sequence, spec)):
        meta = dict(metas[frame_num]) if frame_num < len(metas) else {}
        rows.append(
            {
                "video_id": item.video_id,
                "label": item.label,
                "frame_num": int(frame_num),
                "split": item.split,
                "schema": spec.name,
                "feature_version": spec.feature_schema,
                "feature_mode": spec.feature_mode,
                "feature_dim": spec.feature_dim,
                "target_fps": spec.target_fps,
                "extract_profile": "full_mediapipe_archive_10fps",
                "source_media_path": str(meta.get("source_video_path", "")),
                "source_npz_path": str(item.npz_path),
                "source_report_path": str(item.report_path or ""),
                "source_frame_num": int(meta.get("target_source_frame", frame_num)),
                "chosen_source_frame": int(meta.get("chosen_source_frame", frame_num)),
                "time_sec": float(meta.get("time_sec", frame_num / max(source_fps, 1e-6))),
                "smart_mode": "full_mediapipe",
                "enhance_mode": str(meta.get("enhance_mode", "full_mediapipe_archive")),
                "left_present": float(meta.get("left_present", 0.0)),
                "right_present": float(meta.get("right_present", 0.0)),
                "left_detected": float(meta.get("left_detected", 0.0)),
                "right_detected": float(meta.get("right_detected", 0.0)),
                "left_held": float(meta.get("left_held", 0.0)),
                "right_held": float(meta.get("right_held", 0.0)),
                "left_score": float(meta.get("left_score", 0.0)),
                "right_score": float(meta.get("right_score", 0.0)),
                "pose_present": float(meta.get("pose_present", 0.0)),
                "face_present": float(meta.get("face_present", 0.0)),
                "source_width": int(meta.get("source_width", report.get("width", 0) or 0)),
                "source_height": int(meta.get("source_height", report.get("height", 0) or 0)),
                "source_fps": source_fps,
                "source_duration_sec": source_duration,
                "sample_id": str(report.get("sample_id", "")),
                "mediapipe_version": str(report.get("mediapipe_version", "")),
                "features": sc.format_feature_value(vec),
            }
        )
    return rows


def _read_report(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def cleanup_generated_outputs(
    dataset_dir: str | Path = DEFAULT_DATASET_DIR,
    model_dir: str | Path = DEFAULT_MODEL_DIR,
    backup_root: str | Path = DEFAULT_BACKUP_ROOT,
) -> Path | None:
    """Move generated dataset/model outputs into a timestamped backup folder."""

    backup_dir = Path(backup_root) / f"cleanup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    moved = False
    targets = [Path(dataset_dir)]
    model_root = Path(model_dir)
    targets.append(model_root if model_root.name == "gru" else model_root / "gru")
    for target in targets:
        if not target.exists():
            continue
        rel = target.relative_to(ROOT_DIR) if target.is_relative_to(ROOT_DIR) else Path(target.name)
        destination = backup_dir / rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            destination = destination.with_name(f"{destination.name}_{int(datetime.now().timestamp())}")
        shutil.move(str(target), str(destination))
        moved = True
    return backup_dir if moved else None


def _backup_relative_path(path: Path, dataset_root: Path) -> Path:
    try:
        return path.relative_to(ROOT_DIR)
    except ValueError:
        try:
            return Path(dataset_root.name) / path.relative_to(dataset_root)
        except ValueError:
            return Path(path.name)


def backup_selected_parquets(
    schema_names: Iterable[str],
    labels: Iterable[str],
    dataset_dir: str | Path = DEFAULT_DATASET_DIR,
    backup_root: str | Path = DEFAULT_BACKUP_ROOT,
) -> Path | None:
    """Copy only selected schema/vocab parquet targets into a backup folder."""

    dataset_root = Path(dataset_dir)
    backup_dir = Path(backup_root) / f"cleanup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    copied = False
    for schema_name in schema_names:
        spec = fs.get_schema(schema_name)
        schema_dir = fs.dataset_dir_for(spec, dataset_root)
        for raw_label in labels:
            label = clean_label(raw_label)
            if not label:
                continue
            parquet_path = schema_dir / f"{label}.parquet"
            if not parquet_path.exists():
                continue
            rel = _backup_relative_path(parquet_path, dataset_root)
            destination = backup_dir / rel
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                destination = destination.with_name(f"{destination.stem}_{int(datetime.now().timestamp())}{destination.suffix}")
            shutil.copy2(parquet_path, destination)
            copied = True
    return backup_dir if copied else None


def extract_full_mediapipe_dataset(
    source: str | Path = DEFAULT_SOURCE,
    dataset_dir: str | Path = DEFAULT_DATASET_DIR,
    model_dir: str | Path = DEFAULT_MODEL_DIR,
    backup_root: str | Path = DEFAULT_BACKUP_ROOT,
    schema: str | Iterable[str] = "full",
    clean: str = "none",
    limit_per_class: int | None = None,
    vocab: str | Iterable[str] | None = None,
) -> FullExtractResult:
    schema_names = fs.expand_schema_names(schema)
    vocab_filter = _normalize_vocab_filter(vocab)
    if clean not in {"none", "backup"}:
        raise ValueError("--clean harus none atau backup")

    result = FullExtractResult(schemas=list(schema_names))

    items = discover_full_mediapipe_items(source)
    if not items:
        raise FileNotFoundError(f"Tidak ada *.landmarks.npz di {source}")
    if vocab_filter:
        allowed = set(vocab_filter)
        items = [item for item in items if item.label in allowed]
    result.scanned = len(items)
    if not items:
        vocab_text = ",".join(vocab_filter) if vocab_filter else "-"
        raise FileNotFoundError(f"Tidak ada *.landmarks.npz untuk vocab {vocab_text} di {source}")

    by_label: dict[str, list[FullMediaPipeItem]] = {}
    for item in items:
        by_label.setdefault(item.label, []).append(item)
    result.labels = sorted(by_label)

    full_unscoped = not vocab_filter and tuple(schema_names) == tuple(fs.FULL_SCHEMA_NAMES)
    if clean == "backup":
        if full_unscoped:
            result.backup_dir = cleanup_generated_outputs(dataset_dir=dataset_dir, model_dir=model_dir, backup_root=backup_root)
        else:
            result.backup_dir = backup_selected_parquets(
                schema_names=schema_names,
                labels=result.labels,
                dataset_dir=dataset_dir,
                backup_root=backup_root,
            )

    dataset_root = Path(dataset_dir)
    dataset_root.mkdir(parents=True, exist_ok=True)

    for label, label_items in sorted(by_label.items()):
        if limit_per_class is not None:
            per_split: dict[str, int] = {}
            filtered: list[FullMediaPipeItem] = []
            for item in label_items:
                count = per_split.get(item.split, 0)
                if count >= int(limit_per_class):
                    continue
                per_split[item.split] = count + 1
                filtered.append(item)
            label_items = filtered

        rows_by_schema: dict[str, list[dict[str, Any]]] = {name: [] for name in schema_names}
        for item in label_items:
            try:
                report = _read_report(item.report_path)
                sequences, metas = convert_npz_to_schema_sequences(item.npz_path, schema_names)
                for meta in metas:
                    meta["source_report_path"] = str(item.report_path or "")
                    meta["source_npz_path"] = str(item.npz_path)
                for schema_name in schema_names:
                    rows = _rows_for_sequence(item, schema_name, sequences[schema_name], metas, report)
                    rows_by_schema[schema_name].extend(rows)
                    split_counts = result.counts.setdefault(schema_name, {}).setdefault(label, {"train": 0, "val": 0, "test": 0})
                    split_counts[item.split] = split_counts.get(item.split, 0) + 1
                    result.rows += len(rows)
                result.converted += 1
                print(f"[FULL] {item.split}/{item.label}/{item.video_id} frames={len(metas)}", flush=True)
            except Exception as exc:
                result.failed += 1
                result.errors.append({"path": str(item.npz_path), "error": str(exc)})
                print(f"[FULL][ERR] {item.npz_path}: {exc}", flush=True)

        for schema_name, rows in rows_by_schema.items():
            if not rows:
                continue
            spec = fs.get_schema(schema_name)
            out_dir = fs.dataset_dir_for(spec, dataset_root)
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{label}.parquet"
            pd.DataFrame(rows).to_parquet(out_path, index=False)
            print(f"[WRITE] {out_path} rows={len(rows)}", flush=True)

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source": str(Path(source)),
        "dataset_dir": str(Path(dataset_dir)),
        "schemas": result.schemas,
        "scanned": result.scanned,
        "converted": result.converted,
        "failed": result.failed,
        "rows": result.rows,
        "labels": result.labels,
        "vocab_filter": list(vocab_filter),
        "counts": result.counts,
        "backup_dir": str(result.backup_dir or ""),
        "errors": result.errors,
    }
    result.manifest_path = dataset_root / "full_mediapipe_extract_manifest.json"
    result.manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[MANIFEST] {result.manifest_path}", flush=True)
    return result
