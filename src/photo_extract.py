"""Photo-to-feature staging pipeline for BISINDO datasets.

The extractor treats every image as one live-like frame, then duplicates the
resulting feature vector into a short 10-frame sequence so it can be appended to
the same parquet datasets used by live/video samples.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd

import feature_schemas as fs
from smart_extract import contract as sc


ROOT_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT_DIR / "dataset_parquets"
BACKUP_ROOT = ROOT_DIR / "backups"
GIF_DIR = ROOT_DIR / "assets" / "gifs"
TEMP_ROOT = ROOT_DIR / "tmp" / "photo_extract"
SPLITS = {"train", "val", "test"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
DEFAULT_DUPLICATE_FRAMES = 10


@dataclass(frozen=True)
class PhotoItem:
    image_path: Path
    label: str
    split: str
    video_id: str


@dataclass(frozen=True)
class PhotoTask:
    item: PhotoItem
    schemas: tuple[str, ...]


@dataclass
class PhotoCommitResult:
    scanned: int = 0
    committed: int = 0
    skipped: int = 0
    failed: int = 0
    rows: int = 0
    gif_paths: list[Path] | None = None
    backup_dir: Path | None = None
    dry_run: bool = False


TaskProcessor = Callable[[PhotoItem, Sequence[str], int, Path, str], list[dict]]


def clean_label(value: str) -> str:
    return str(value or "").strip().replace(" ", "_").lower()


def safe_name(value: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in str(value))


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTS


def _image_files(folder: Path) -> list[Path]:
    return sorted(path for path in folder.iterdir() if is_image_file(path)) if folder.exists() else []


def _split_from_path(path: Path, default_split: str) -> str:
    for part in reversed(path.parts):
        value = part.lower()
        if value in SPLITS:
            return value
    return default_split


def stable_photo_video_id(split: str, label: str, image_path: str | Path) -> str:
    path = Path(image_path).expanduser()
    resolved = path.resolve()
    digest = hashlib.sha1(str(resolved).encode("utf-8")).hexdigest()[:10]
    return f"{split}_photo_{clean_label(label)}_{safe_name(path.stem)}_{digest}"


def scan_photo_items(
    paths: Sequence[str | Path],
    default_split: str = "train",
    label: str | None = None,
) -> list[PhotoItem]:
    """Scan supported image layouts into concrete photo extraction tasks."""

    default_split = str(default_split or "train").lower()
    if default_split not in SPLITS:
        raise ValueError("split harus salah satu: train, val, test")
    fixed_label = clean_label(label) if label else None
    items: list[PhotoItem] = []
    seen: set[tuple[str, str, str]] = set()

    def add(image_path: Path, item_label: str, item_split: str) -> None:
        item_label = clean_label(item_label)
        item_split = str(item_split or default_split).lower()
        if item_split not in SPLITS:
            item_split = default_split
        if not item_label:
            raise ValueError(f"Label kosong untuk {image_path}")
        resolved = image_path.resolve()
        key = (str(resolved), item_label, item_split)
        if key in seen:
            return
        seen.add(key)
        items.append(
            PhotoItem(
                image_path=image_path,
                label=item_label,
                split=item_split,
                video_id=stable_photo_video_id(item_split, item_label, resolved),
            )
        )

    for raw_path in paths:
        path = Path(raw_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Path tidak ditemukan: {path}")
        if is_image_file(path):
            if not fixed_label:
                raise ValueError("Import single image butuh --label.")
            add(path, fixed_label, default_split)
            continue
        if not path.is_dir():
            continue

        direct_images = _image_files(path)
        if direct_images:
            split = path.parent.name.lower() if path.parent.name.lower() in SPLITS else default_split
            item_label = fixed_label or path.name
            for image_path in direct_images:
                add(image_path, item_label, split)
            continue

        if fixed_label:
            for image_path in sorted(path.rglob("*")):
                if is_image_file(image_path):
                    add(image_path, fixed_label, _split_from_path(image_path, default_split))
            continue

        child_dirs = [child for child in sorted(path.iterdir()) if child.is_dir()]
        split_dirs = [child for child in child_dirs if child.name.lower() in SPLITS]
        if split_dirs and len(split_dirs) == len(child_dirs):
            for split_dir in split_dirs:
                split = split_dir.name.lower()
                for vocab_dir in sorted(child for child in split_dir.iterdir() if child.is_dir()):
                    for image_path in _image_files(vocab_dir):
                        add(image_path, vocab_dir.name, split)
            continue

        fallback_split = path.name.lower() if path.name.lower() in SPLITS else default_split
        for vocab_dir in child_dirs:
            for image_path in _image_files(vocab_dir):
                add(image_path, vocab_dir.name, fallback_split)

    return items


def expand_schema_args(values: Sequence[str] | str | None) -> tuple[str, ...]:
    if values is None:
        values = (fs.DEFAULT_SCHEMA,)
    if isinstance(values, str):
        values = (values,)
    out: list[str] = []
    for value in values:
        for name in fs.expand_schema_names(value):
            if name not in out:
                out.append(name)
    if not out:
        raise ValueError("Minimal pilih satu schema.")
    return tuple(out)


def entry_id(schema: str, split: str, label: str, video_id: str) -> str:
    return f"{fs.normalize_schema_name(schema)}|{split}|{clean_label(label)}|{video_id}"


def _dataset_root_for(schema: str, dataset_dir: str | Path) -> Path:
    return fs.dataset_dir_for(fs.get_schema(schema), dataset_dir)


def _existing_video_ids(parquet_path: Path) -> set[str]:
    if not parquet_path.exists():
        return set()
    try:
        df = pd.read_parquet(parquet_path, columns=["video_id"])
    except TypeError:
        df = pd.read_parquet(parquet_path)
    except Exception:
        return set()
    if "video_id" not in df.columns:
        return set()
    return {str(value) for value in df["video_id"].dropna().unique()}


def _existing_video_ids_for_label(dataset_root: Path, label: str) -> set[str]:
    label = clean_label(label)
    if not dataset_root.exists():
        return set()
    ids: set[str] = set()
    for parquet_path in sorted(dataset_root.glob("*.parquet")):
        try:
            df = pd.read_parquet(parquet_path, columns=["label", "video_id"])
        except Exception:
            try:
                df = pd.read_parquet(parquet_path)
            except Exception:
                continue
        if "video_id" not in df.columns:
            continue
        if "label" in df.columns:
            df = df[df["label"].astype(str).map(clean_label) == label]
        elif clean_label(parquet_path.stem) != label:
            continue
        ids.update(str(value) for value in df["video_id"].dropna().unique())
    return ids


def collect_existing_ids(
    schema_names: Sequence[str],
    labels: Iterable[str],
    dataset_dir: str | Path = DATASET_DIR,
) -> dict[tuple[str, str], set[str]]:
    out: dict[tuple[str, str], set[str]] = {}
    label_names = sorted({clean_label(value) for value in labels})
    for schema_name in schema_names:
        spec = fs.get_schema(schema_name)
        root = _dataset_root_for(spec.name, dataset_dir)
        for label in label_names:
            out[(spec.name, label)] = _existing_video_ids_for_label(root, label)
    return out


def _feature_metadata_from_result(res, frame_num: int, time_sec: float, profile: str) -> dict:
    return {
        "out_index": int(frame_num),
        "time_sec": float(time_sec),
        "target_source_frame": 0,
        "chosen_source_frame": 0,
        "enhance_mode": "photo_live",
        "left_present": float(res.present[0]) if getattr(res, "present", None) is not None else 0.0,
        "right_present": float(res.present[1]) if getattr(res, "present", None) is not None else 0.0,
        "left_detected": float(res.detected[0]) if getattr(res, "detected", None) is not None else 0.0,
        "right_detected": float(res.detected[1]) if getattr(res, "detected", None) is not None else 0.0,
        "left_held": float(res.held[0]) if getattr(res, "held", None) is not None else 0.0,
        "right_held": float(res.held[1]) if getattr(res, "held", None) is not None else 0.0,
        "left_score": float(res.scores[0]) if getattr(res, "scores", None) is not None else 0.0,
        "right_score": float(res.scores[1]) if getattr(res, "scores", None) is not None else 0.0,
        "hand_ms": float(getattr(res, "hand_ms", 0.0)),
        "pose_ms": float(getattr(res, "pose_ms", 0.0)),
        "profile": str(profile),
    }


def _build_duplicate_frames_meta(res, duplicate_frames: int, profile: str, target_fps: float = sc.TARGET_FPS) -> list[dict]:
    return [
        _feature_metadata_from_result(res, frame_num, frame_num / max(float(target_fps), 1e-6), profile)
        for frame_num in range(int(duplicate_frames))
    ]


def stage_schema_result(
    *,
    item: PhotoItem,
    schema: str,
    features: np.ndarray,
    frames: list[dict],
    session_dir: str | Path,
    profile: str,
) -> dict:
    spec = fs.get_schema(schema)
    arr = fs.ensure_feature_dim(features, spec).astype(np.float32, copy=False)
    out_dir = Path(session_dir) / "samples" / spec.name / item.label
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / f"{safe_name(item.video_id)}.npz"
    np.savez_compressed(
        npz_path,
        features=arr,
        frames_json=np.asarray(json.dumps(frames)),
        source_image_path=np.asarray(str(item.image_path.resolve())),
        label=np.asarray(item.label),
        split=np.asarray(item.split),
        video_id=np.asarray(item.video_id),
        schema=np.asarray(spec.name),
        feature_version=np.asarray(spec.feature_schema),
        feature_mode=np.asarray(spec.feature_mode),
        feature_dim=np.asarray(spec.feature_dim, dtype=np.int32),
        target_fps=np.asarray(spec.target_fps, dtype=np.float32),
        extract_profile=np.asarray(f"photo_live_{profile}"),
    )
    return {
        "entry_id": entry_id(spec.name, item.split, item.label, item.video_id),
        "status": "success",
        "accepted": True,
        "duplicate_existing": False,
        "schema": spec.name,
        "feature_version": spec.feature_schema,
        "feature_mode": spec.feature_mode,
        "feature_dim": int(spec.feature_dim),
        "frames": int(arr.shape[0]),
        "label": item.label,
        "split": item.split,
        "video_id": item.video_id,
        "source_image_path": str(item.image_path.resolve()),
        "npz_path": str(npz_path),
        "gif_temp_path": "",
        "error": "",
    }


def skipped_existing_entry(item: PhotoItem, schema: str) -> dict:
    spec = fs.get_schema(schema)
    return {
        "entry_id": entry_id(spec.name, item.split, item.label, item.video_id),
        "status": "skipped_existing",
        "accepted": False,
        "duplicate_existing": True,
        "schema": spec.name,
        "feature_version": spec.feature_schema,
        "feature_mode": spec.feature_mode,
        "feature_dim": int(spec.feature_dim),
        "frames": 0,
        "label": item.label,
        "split": item.split,
        "video_id": item.video_id,
        "source_image_path": str(item.image_path.resolve()),
        "npz_path": "",
        "gif_temp_path": "",
        "error": "video_id already exists",
    }


def failed_entry(item: PhotoItem, schema: str, error: str) -> dict:
    spec = fs.get_schema(schema)
    return {
        "entry_id": entry_id(spec.name, item.split, item.label, item.video_id),
        "status": "failed",
        "accepted": False,
        "duplicate_existing": False,
        "schema": spec.name,
        "feature_version": spec.feature_schema,
        "feature_mode": spec.feature_mode,
        "feature_dim": int(spec.feature_dim),
        "frames": 0,
        "label": item.label,
        "split": item.split,
        "video_id": item.video_id,
        "source_image_path": str(item.image_path.resolve()),
        "npz_path": "",
        "gif_temp_path": "",
        "error": str(error),
    }


class PhotoExtractorContext:
    """Per-worker live-like MediaPipe extractor bundle."""

    def __init__(self, schemas: Sequence[str], profile: str) -> None:
        import cv2  # noqa: F401 - imported here so import-time CLI stays light.
        import holistic_features
        import live_gru_fast
        from smart_extract.live_bisindo_mp_real_shoulder_v6 import (
            HandTrack,
            UltraMediaPipeExtractor,
            blank_shoulders,
        )

        self.cv2 = cv2
        self.live_gru_fast = live_gru_fast
        self.HandTrack = HandTrack
        self.blank_shoulders = blank_shoulders
        self.schemas = tuple(fs.normalize_schema_name(name) for name in schemas)
        self.profile = live_gru_fast.normalize_profile(profile)
        self.profile_cfg = live_gru_fast.LIVE_PROFILES[self.profile]
        self.smart = None
        self.holistic = None

        if "smart180" in self.schemas:
            self.smart = UltraMediaPipeExtractor(
                feature_mode=sc.FEATURE_MODE,
                shoulder_backend=str(self.profile_cfg["shoulder_backend"]),
                proc_width=int(self.profile_cfg["proc_width"]),
                pose_proc_width=int(self.profile_cfg["pose_proc_width"]),
                pose_every=int(self.profile_cfg["pose_every"]),
                hand_every=int(self.profile_cfg["hand_every"]),
                hand_model_complexity=0,
                pose_model_complexity=0,
                det_conf=float(self.profile_cfg["det_conf"]),
                track_conf=float(self.profile_cfg["track_conf"]),
                smooth_alpha=float(self.profile_cfg["smooth_alpha"]),
                hold_frames=int(self.profile_cfg["hold_frames"]),
                mirror_input=False,
                mirror_handedness=True,
                shoulder_smooth_alpha=float(self.profile_cfg["shoulder_smooth_alpha"]),
            )
        holistic_names = [name for name in self.schemas if fs.get_schema(name).extractor == "holistic"]
        if holistic_names:
            self.holistic = holistic_features.MultiSchemaHolisticExtractor(
                holistic_names,
                proc_width=int(self.profile_cfg["proc_width"]),
                det_conf=float(self.profile_cfg["det_conf"]),
                track_conf=float(self.profile_cfg["track_conf"]),
                model_complexity=0,
                smooth_landmarks=True,
                refine_face_landmarks=False,
            )

    def close(self) -> None:
        if self.smart is not None:
            self.smart.close()
        if self.holistic is not None:
            self.holistic.close()

    def _reset_smart_state(self) -> None:
        if self.smart is None:
            return
        self.smart.left = self.HandTrack()
        self.smart.right = self.HandTrack()
        self.smart.shoulders = self.blank_shoulders()
        self.smart.shoulder_age = 10000
        self.smart.shoulder_real_seen = False
        self.smart.frame_i = 0
        self.smart.last_hand_ms = 0.0
        self.smart.last_pose_ms = 0.0

    def _read_image(self, image_path: Path) -> np.ndarray:
        frame = self.cv2.imread(str(image_path))
        if frame is None:
            raise RuntimeError(f"Gagal baca image: {image_path}")
        width = int(self.profile_cfg["width"])
        height = int(self.profile_cfg["height"])
        if frame.shape[1] != width or frame.shape[0] != height:
            frame = self.cv2.resize(frame, (width, height), interpolation=self.cv2.INTER_AREA)
        return frame

    def extract(
        self,
        item: PhotoItem,
        schemas: Sequence[str],
        duplicate_frames: int,
        session_dir: Path,
    ) -> list[dict]:
        schema_names = tuple(fs.normalize_schema_name(name) for name in schemas)
        work = self._read_image(item.image_path)
        results_by_schema = {}
        if "smart180" in schema_names:
            self._reset_smart_state()
            results_by_schema["smart180"] = self.smart.process(work)
        holistic_names = [name for name in schema_names if fs.get_schema(name).extractor == "holistic"]
        if holistic_names:
            results_by_schema.update(self.holistic.process(work))

        entries: list[dict] = []
        for schema_name in schema_names:
            spec = fs.get_schema(schema_name)
            res = results_by_schema[schema_name]
            vector = np.asarray(res.vector, dtype=np.float32).reshape(1, -1)
            features = np.repeat(vector, int(duplicate_frames), axis=0)
            frames = _build_duplicate_frames_meta(res, int(duplicate_frames), self.profile, spec.target_fps)
            entries.append(
                stage_schema_result(
                    item=item,
                    schema=spec.name,
                    features=features,
                    frames=frames,
                    session_dir=session_dir,
                    profile=self.profile,
                )
            )
        return entries


def _process_chunk(tasks: Sequence[PhotoTask], profile: str, duplicate_frames: int, session_dir: Path) -> list[dict]:
    schema_names: list[str] = []
    for task in tasks:
        for schema_name in task.schemas:
            if schema_name not in schema_names:
                schema_names.append(schema_name)
    context = PhotoExtractorContext(schema_names, profile)
    out: list[dict] = []
    try:
        for task in tasks:
            try:
                out.extend(context.extract(task.item, task.schemas, duplicate_frames, session_dir))
            except Exception as exc:
                out.extend(failed_entry(task.item, schema_name, str(exc)) for schema_name in task.schemas)
    finally:
        context.close()
    return out


def _chunk_tasks(tasks: Sequence[PhotoTask], workers: int) -> list[list[PhotoTask]]:
    n = max(1, min(int(workers), len(tasks) or 1))
    chunks: list[list[PhotoTask]] = [[] for _ in range(n)]
    for idx, task in enumerate(tasks):
        chunks[idx % n].append(task)
    return [chunk for chunk in chunks if chunk]


def _run_group_tasks(
    tasks: Sequence[PhotoTask],
    *,
    profile: str,
    duplicate_frames: int,
    session_dir: Path,
    workers: int,
    task_processor: TaskProcessor | None = None,
) -> list[dict]:
    if not tasks:
        return []
    if task_processor is not None:
        entries: list[dict] = []
        for task in tasks:
            try:
                entries.extend(task_processor(task.item, task.schemas, duplicate_frames, session_dir, profile))
            except Exception as exc:
                entries.extend(failed_entry(task.item, schema_name, str(exc)) for schema_name in task.schemas)
        return entries

    chunks = _chunk_tasks(tasks, workers)
    entries: list[dict] = []
    with ThreadPoolExecutor(max_workers=len(chunks)) as executor:
        future_map = {
            executor.submit(_process_chunk, chunk, profile, int(duplicate_frames), session_dir): chunk
            for chunk in chunks
        }
        for future in as_completed(future_map):
            entries.extend(future.result())
    return entries


def _summarize_entries(entries: Sequence[dict]) -> dict[str, int]:
    summary = {"success": 0, "failed": 0, "skipped_existing": 0}
    for entry in entries:
        status = str(entry.get("status", ""))
        if status in summary:
            summary[status] += 1
    summary["total"] = len(entries)
    return summary


def write_manifest(session_dir: str | Path, manifest: dict) -> Path:
    path = Path(session_dir) / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def read_manifest(session_dir: str | Path) -> dict:
    return json.loads((Path(session_dir) / "manifest.json").read_text(encoding="utf-8"))


def write_source_log(session_dir: str | Path, entries: Sequence[dict]) -> Path:
    path = Path(session_dir) / "source_log.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry, ensure_ascii=True) + "\n")
    return path


def extract_photo_session(
    *,
    paths: Sequence[str | Path],
    schemas: Sequence[str] | str | None = None,
    default_split: str = "train",
    label: str | None = None,
    profile: str = "fast10",
    workers: int = 1,
    duplicate_frames: int = DEFAULT_DUPLICATE_FRAMES,
    dataset_dir: str | Path = DATASET_DIR,
    session_root: str | Path = TEMP_ROOT,
    task_processor: TaskProcessor | None = None,
    overwrite_existing: bool = False,
) -> dict:
    schema_names = expand_schema_args(schemas)
    items = scan_photo_items(paths, default_split=default_split, label=label)
    if not items:
        raise ValueError("Tidak ada image yang bisa diekstrak.")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = Path(session_root) / stamp
    suffix = 1
    while session_dir.exists():
        session_dir = Path(session_root) / f"{stamp}_{suffix:02d}"
        suffix += 1
    session_dir.mkdir(parents=True, exist_ok=True)

    existing = collect_existing_ids(schema_names, (item.label for item in items), dataset_dir=dataset_dir)
    entries: list[dict] = []
    groups: dict[tuple[str, str], list[PhotoItem]] = {}
    for item in items:
        groups.setdefault((item.split, item.label), []).append(item)

    total_groups = len(groups)
    for group_idx, ((split, label_name), group_items) in enumerate(sorted(groups.items()), start=1):
        print(
            f"[PHOTO] vocab {group_idx}/{total_groups} label={label_name} split={split} samples={len(group_items)}",
            flush=True,
        )
        tasks: list[PhotoTask] = []
        for item in group_items:
            missing: list[str] = []
            for schema_name in schema_names:
                if item.video_id in existing.get((schema_name, item.label), set()) and not overwrite_existing:
                    print(f"[SKIP existing video_id] {schema_name}/{item.label}/{item.video_id}", flush=True)
                    entries.append(skipped_existing_entry(item, schema_name))
                else:
                    missing.append(schema_name)
            if missing:
                tasks.append(PhotoTask(item=item, schemas=tuple(missing)))

        group_entries = _run_group_tasks(
            tasks,
            profile=profile,
            duplicate_frames=int(duplicate_frames),
            session_dir=session_dir,
            workers=max(1, int(workers)),
            task_processor=task_processor,
        )
        entries.extend(group_entries)
        ok = sum(1 for entry in group_entries if entry.get("status") == "success")
        fail = sum(1 for entry in group_entries if entry.get("status") == "failed")
        skipped = sum(
            1
            for entry in entries
            if entry.get("status") == "skipped_existing" and entry.get("label") == label_name and entry.get("split") == split
        )
        print(f"[PHOTO] vocab {group_idx}/{total_groups} done ok={ok} fail={fail} skipped_existing={skipped}", flush=True)

    manifest = {
        "session_id": session_dir.name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "session_dir": str(session_dir),
        "source_paths": [str(Path(path).expanduser()) for path in paths],
        "profile": str(profile),
        "schemas": list(schema_names),
        "duplicate_frames": int(duplicate_frames),
        "dataset_dir": str(dataset_dir),
        "summary": _summarize_entries(entries),
        "entries": sorted(entries, key=lambda e: (e.get("schema", ""), e.get("split", ""), e.get("label", ""), e.get("video_id", ""))),
    }
    write_manifest(session_dir, manifest)
    write_source_log(session_dir, manifest["entries"])
    print(
        f"photo-extract: session={session_dir} success={manifest['summary']['success']} "
        f"failed={manifest['summary']['failed']} skipped_existing={manifest['summary']['skipped_existing']}",
        flush=True,
    )
    return manifest


def load_staged_result(entry: dict) -> tuple[np.ndarray, list[dict]]:
    npz_path = Path(str(entry.get("npz_path", "")))
    if not npz_path.exists():
        raise FileNotFoundError(f"Staged npz tidak ditemukan: {npz_path}")
    payload = np.load(npz_path, allow_pickle=False)
    features = np.asarray(payload["features"], dtype=np.float32)
    frames_raw = str(np.asarray(payload["frames_json"]).item())
    frames = json.loads(frames_raw)
    return fs.ensure_feature_dim(features, entry["schema"]), list(frames)


def _format_feature_value(features: Iterable[float]) -> str:
    return sc.format_feature_value(np.asarray(features, dtype=np.float32))


def rows_from_staged_entry(entry: dict) -> list[dict]:
    spec = fs.get_schema(str(entry["schema"]))
    features, frames = load_staged_result(entry)
    rows: list[dict] = []
    for frame_num, vec in enumerate(features):
        meta = frames[frame_num] if frame_num < len(frames) else {}
        rows.append(
            {
                "video_id": str(entry["video_id"]),
                "label": clean_label(str(entry["label"])),
                "frame_num": int(frame_num),
                "split": str(entry["split"]).lower(),
                "schema": spec.name,
                "feature_version": spec.feature_schema,
                "feature_mode": spec.feature_mode,
                "feature_dim": spec.feature_dim,
                "target_fps": spec.target_fps,
                "extract_profile": f"photo_live_{meta.get('profile', '')}".rstrip("_"),
                "source_media_path": str(entry["source_image_path"]),
                "source_frame_num": int(meta.get("target_source_frame", 0)),
                "chosen_source_frame": int(meta.get("chosen_source_frame", 0)),
                "time_sec": float(meta.get("time_sec", frame_num / max(spec.target_fps, 1e-6))),
                "smart_mode": "photo_live",
                "enhance_mode": str(meta.get("enhance_mode", "photo_live")),
                "left_present": float(meta.get("left_present", 0.0)),
                "right_present": float(meta.get("right_present", 0.0)),
                "left_detected": float(meta.get("left_detected", 0.0)),
                "right_detected": float(meta.get("right_detected", 0.0)),
                "left_held": float(meta.get("left_held", 0.0)),
                "right_held": float(meta.get("right_held", 0.0)),
                "left_score": float(meta.get("left_score", 0.0)),
                "right_score": float(meta.get("right_score", 0.0)),
                "features": _format_feature_value(vec),
            }
        )
    return rows


def _backup_existing(parquet_path: Path, backup_dir: Path, copied: set[Path]) -> None:
    if not parquet_path.exists() or parquet_path in copied:
        return
    backup_dir.mkdir(parents=True, exist_ok=True)
    try:
        rel = parquet_path.resolve().relative_to(ROOT_DIR.resolve())
    except Exception:
        rel = Path(parquet_path.parent.name) / parquet_path.name
    dest = backup_dir / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(parquet_path, dest)
    copied.add(parquet_path)
    print(f"[BACKUP] {parquet_path} -> {dest}", flush=True)


def _feature_gif_dest(entry: dict, gif_dir: str | Path = GIF_DIR) -> Path:
    spec = fs.get_schema(str(entry["schema"]))
    return (
        Path(gif_dir)
        / "samples"
        / spec.name
        / clean_label(str(entry["label"]))
        / f"{safe_name(str(entry['split']))}_{safe_name(str(entry['video_id']))}_feature.gif"
    )


def commit_photo_session(
    *,
    session_dir: str | Path,
    accepted_entry_ids: Sequence[str] | None = None,
    dataset_dir: str | Path = DATASET_DIR,
    backup_root: str | Path = BACKUP_ROOT,
    gif_dir: str | Path = GIF_DIR,
    dry_run: bool = False,
    overwrite_existing: bool = False,
) -> PhotoCommitResult:
    manifest = read_manifest(session_dir)
    accepted = set(accepted_entry_ids or [])
    result = PhotoCommitResult(gif_paths=[], dry_run=bool(dry_run))
    copied: set[Path] = set()
    backup_dir = Path(backup_root) / f"photo_import_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    rows_by_parquet: dict[Path, list[dict]] = {}
    replace_ids_by_parquet: dict[Path, set[str]] = {}
    known_ids_by_key: dict[tuple[str, str], set[str]] = {}

    for entry in manifest.get("entries", []):
        if entry.get("status") != "success":
            continue
        if accepted and entry.get("entry_id") not in accepted:
            continue
        if not accepted and not bool(entry.get("accepted", True)):
            continue
        result.scanned += 1
        spec = fs.get_schema(str(entry["schema"]))
        label = clean_label(str(entry["label"]))
        dataset_root = fs.dataset_dir_for(spec, dataset_dir)
        parquet_path = dataset_root / f"{label}.parquet"
        video_id = str(entry["video_id"])
        key = (spec.name, label)
        known_ids = known_ids_by_key.setdefault(key, _existing_video_ids_for_label(dataset_root, label))
        if video_id in known_ids and not overwrite_existing:
            result.skipped += 1
            entry["commit_status"] = "skipped_existing"
            print(f"[SKIP existing video_id] {spec.name}/{label}/{video_id}", flush=True)
            continue
        if video_id in known_ids:
            replace_ids_by_parquet.setdefault(parquet_path, set()).add(video_id)
            print(f"[OVERWRITE existing video_id] {spec.name}/{label}/{video_id}", flush=True)
        try:
            rows = rows_from_staged_entry(entry)
        except Exception as exc:
            result.failed += 1
            entry["commit_status"] = "failed"
            entry["commit_error"] = str(exc)
            continue
        rows_by_parquet.setdefault(parquet_path, []).extend(rows)
        known_ids.add(video_id)
        result.rows += len(rows)
        result.committed += 1
        entry["commit_status"] = "pending" if dry_run else "committed"

        gif_temp = str(entry.get("gif_temp_path") or "")
        if gif_temp and Path(gif_temp).exists():
            dest = _feature_gif_dest(entry, gif_dir=gif_dir)
            result.gif_paths.append(dest)
            if not dry_run:
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(gif_temp, dest)
                entry["gif_asset_path"] = str(dest)

    if not dry_run:
        for parquet_path, rows in rows_by_parquet.items():
            parquet_path.parent.mkdir(parents=True, exist_ok=True)
            _backup_existing(parquet_path, backup_dir, copied)
            df_new = pd.DataFrame(rows)
            if parquet_path.exists():
                df_old = pd.read_parquet(parquet_path)
                replace_ids = replace_ids_by_parquet.get(parquet_path, set())
                if replace_ids and "video_id" in df_old.columns:
                    mask = df_old["video_id"].astype(str).isin(replace_ids)
                    if "label" in df_old.columns:
                        mask = mask & df_old["label"].astype(str).map(clean_label).eq(clean_label(parquet_path.stem))
                    df_old = df_old.loc[~mask].copy()
                df_out = pd.concat([df_old, df_new], ignore_index=True)
            else:
                df_out = df_new
            df_out.to_parquet(parquet_path, index=False)
            print(f"[APPEND parquet +{len(rows)} rows] {parquet_path}", flush=True)
        manifest["last_commit"] = {
            "committed_at": datetime.now().isoformat(timespec="seconds"),
            "committed": result.committed,
            "skipped": result.skipped,
            "failed": result.failed,
            "rows": result.rows,
            "dry_run": False,
        }
        write_manifest(session_dir, manifest)

    result.backup_dir = backup_dir if copied else None
    return result
