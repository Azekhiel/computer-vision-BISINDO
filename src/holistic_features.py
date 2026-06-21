"""MediaPipe Holistic feature extraction for paper-sized BISINDO schemas."""

from __future__ import annotations

from argparse import Namespace
from dataclasses import dataclass
import math
from pathlib import Path
import time
from typing import Any, Sequence

import cv2
import mediapipe as mp
import numpy as np

import face_reference as fr
import feature_schemas as fs
from smart_extract.live_bisindo_mp_real_shoulder_v6 import (
    center_crop,
    draw_shoulders,
    draw_simple_hand,
    resize_width,
    build_feature,
)


mp_holistic = mp.solutions.holistic


@dataclass
class HolisticFrameResult:
    vector: np.ndarray
    left: np.ndarray | None
    right: np.ndarray | None
    shoulders: np.ndarray
    detected: np.ndarray
    held: np.ndarray
    present: np.ndarray
    scores: np.ndarray
    fps_infer: float
    hand_ms: float
    pose_ms: float
    feature_mode: str
    raw_full: dict[str, Any] | None = None


def make_holistic_args(**overrides) -> Namespace:
    values = {
        "target_fps": 10.0,
        "width": 640,
        "height": 480,
        "center_crop": 1.0,
        "proc_width": 384,
        "det_conf": 0.40,
        "track_conf": 0.45,
        "model_complexity": 0,
        "smooth_landmarks": True,
        "refine_face_landmarks": False,
        "quiet": False,
        "gif_width": 420,
    }
    values.update(overrides)
    return Namespace(**values)


def _landmarks_xyz(landmarks, count: int) -> np.ndarray:
    out = np.zeros((count, 3), dtype=np.float32)
    if landmarks is None:
        return out
    for idx, landmark in enumerate(landmarks.landmark[:count]):
        out[idx] = (float(landmark.x), float(landmark.y), float(landmark.z))
    return out


def _pose_xyzw(landmarks) -> np.ndarray:
    out = np.zeros((33, 4), dtype=np.float32)
    if landmarks is None:
        return out
    for idx, landmark in enumerate(landmarks.landmark[:33]):
        out[idx] = (
            float(landmark.x),
            float(landmark.y),
            float(landmark.z),
            float(getattr(landmark, "visibility", 0.0)),
        )
    return out


def _shoulders_from_pose(pose_landmarks) -> np.ndarray:
    out = np.full((2, 4), np.nan, dtype=np.float32)
    if pose_landmarks is None:
        return out
    pts = pose_landmarks.landmark
    for row, idx in enumerate((11, 12)):
        if idx < len(pts):
            lm = pts[idx]
            out[row] = (
                float(lm.x),
                float(lm.y),
                float(lm.z),
                float(getattr(lm, "visibility", 0.0)),
            )
    return out


def build_full_landmark_feature(results: Any) -> tuple[np.ndarray, dict[str, Any]]:
    """Return full raw Holistic landmark vector for archive/debug.

    Layout: right hand xyz, left hand xyz, pose xyzw, face xyz, shoulders xyzw.
    This is used only when saving full feature archives. It does not add another
    MediaPipe pass; it reuses the same Holistic result already computed for
    khukuh/adi/smart180_face.
    """
    right = _landmarks_xyz(results.right_hand_landmarks, 21)
    left = _landmarks_xyz(results.left_hand_landmarks, 21)
    pose = _pose_xyzw(results.pose_landmarks)
    face = _landmarks_xyz(results.face_landmarks, 468)
    shoulders = np.nan_to_num(_shoulders_from_pose(results.pose_landmarks), nan=0.0)
    vector = np.concatenate([
        right.reshape(-1),
        left.reshape(-1),
        pose.reshape(-1),
        face.reshape(-1),
        shoulders.reshape(-1),
    ]).astype(np.float32)
    meta = {
        "feature_mode": "mediapipe_holistic_full_raw",
        "feature_dim": int(vector.shape[0]),
        "columns": fs.full_raw_feature_column_names(),
        "left_present": float(results.left_hand_landmarks is not None),
        "right_present": float(results.right_hand_landmarks is not None),
        "pose_present": float(results.pose_landmarks is not None),
        "face_present": float(results.face_landmarks is not None),
        "shoulders": shoulders,
    }
    return vector, meta


def build_paper_feature(
    schema: str | fs.FeatureSchema,
    results: Any,
    state: dict | None = None,
    dt: float = 0.1,
) -> tuple[np.ndarray, dict[str, Any]]:
    spec = fs.get_schema(schema)
    right = _landmarks_xyz(results.right_hand_landmarks, 21)
    left = _landmarks_xyz(results.left_hand_landmarks, 21)
    pose_xyz = _landmarks_xyz(results.pose_landmarks, 33)
    pose_xyzw = _pose_xyzw(results.pose_landmarks)
    face = _landmarks_xyz(results.face_landmarks, 468)

    present = np.array([float(results.left_hand_landmarks is not None), float(results.right_hand_landmarks is not None)], dtype=np.float32)
    detected = present.copy()
    held = np.zeros(2, dtype=np.float32)
    scores = present.copy()
    shoulders = _shoulders_from_pose(results.pose_landmarks)

    if spec.name == "smart180":
        vector = build_feature(
            "btj_global_local",
            left if results.left_hand_landmarks is not None else None,
            right if results.right_hand_landmarks is not None else None,
            shoulders,
            present,
            detected,
            held,
            scores,
        )
    elif spec.name == "smart268":
        vector = build_feature(
            "268",
            left if results.left_hand_landmarks is not None else None,
            right if results.right_hand_landmarks is not None else None,
            shoulders,
            present,
            detected,
            held,
            scores,
        )
    elif spec.name == "khukuh1629":
        vector = np.concatenate(
            [
                right.reshape(-1),
                left.reshape(-1),
                pose_xyz.reshape(-1),
                face.reshape(-1),
            ]
        ).astype(np.float32)
    elif spec.name == "adi1662":
        vector = np.concatenate(
            [
                pose_xyzw.reshape(-1),
                face.reshape(-1),
                left.reshape(-1),
                right.reshape(-1),
            ]
        ).astype(np.float32)
    elif spec.name == "smart180_face1584":
        smart = build_feature(
            "btj_global_local",
            left if results.left_hand_landmarks is not None else None,
            right if results.right_hand_landmarks is not None else None,
            shoulders,
            present,
            detected,
            held,
            scores,
        )
        vector = np.concatenate([smart, face.reshape(-1)]).astype(np.float32)
    elif fr.is_face_ref_schema(spec.name):
        smart = build_feature(
            "btj_global_local",
            left if results.left_hand_landmarks is not None else None,
            right if results.right_hand_landmarks is not None else None,
            shoulders,
            present,
            detected,
            held,
            scores,
        )
        vector = fr.build_face_schema_vector(
            spec.name,
            smart,
            face,
            shoulders,
            left if results.left_hand_landmarks is not None else None,
            right if results.right_hand_landmarks is not None else None,
            face_present=bool(results.face_landmarks is not None),
            state=state,
            dt=dt,
        )
    else:
        raise ValueError(f"Paper Holistic extractor tidak mendukung schema {spec.name}")

    if vector.shape[0] != spec.feature_dim:
        raise RuntimeError(f"Feature dim mismatch: {vector.shape[0]} != {spec.feature_dim}")

    meta = {
        "left": left if results.left_hand_landmarks is not None else None,
        "right": right if results.right_hand_landmarks is not None else None,
        "shoulders": shoulders,
        "left_present": float(results.left_hand_landmarks is not None),
        "right_present": float(results.right_hand_landmarks is not None),
        "left_detected": float(results.left_hand_landmarks is not None),
        "right_detected": float(results.right_hand_landmarks is not None),
        "left_score": float(results.left_hand_landmarks is not None),
        "right_score": float(results.right_hand_landmarks is not None),
    }
    return vector, meta


class HolisticLiveExtractor:
    def __init__(
        self,
        schema: str | fs.FeatureSchema,
        proc_width: int = 384,
        det_conf: float = 0.50,
        track_conf: float = 0.50,
        model_complexity: int = 1,
        smooth_landmarks: bool = True,
        refine_face_landmarks: bool = True,
    ) -> None:
        self.spec = fs.get_schema(schema)
        supported = {"smart180", "smart268", "khukuh1629", "adi1662", "smart180_face1584"} | set(fs.FACE_REF_SCHEMA_NAMES)
        if self.spec.name not in supported:
            raise ValueError(f"HolisticLiveExtractor tidak mendukung schema {self.spec.name}")
        self.proc_width = int(proc_width)
        self.feature_mode = self.spec.feature_mode
        self.backend_name = "studio_holistic"
        self.backend_detail = "mp.solutions.holistic.Holistic"
        # Temporal state + last-frame timestamp for face-reference velocity terms.
        self._face_state: dict = {}
        self._last_t: float | None = None
        self.holistic = mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=int(model_complexity),
            smooth_landmarks=bool(smooth_landmarks),
            refine_face_landmarks=bool(refine_face_landmarks),
            enable_segmentation=False,
            min_detection_confidence=float(det_conf),
            min_tracking_confidence=float(track_conf),
        )

    def close(self) -> None:
        self.holistic.close()

    def process(self, frame_bgr: np.ndarray) -> HolisticFrameResult:
        work = resize_width(frame_bgr, self.proc_width)
        rgb = cv2.cvtColor(work, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        t0 = time.perf_counter()
        results = self.holistic.process(rgb)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        now = time.perf_counter()
        dt = (now - self._last_t) if self._last_t is not None else (1.0 / max(self.spec.target_fps, 1e-6))
        self._last_t = now
        vector, meta = build_paper_feature(self.spec, results, state=self._face_state, dt=dt)
        raw_vector, raw_meta = build_full_landmark_feature(results)
        raw_full = {"vector": raw_vector, "meta": raw_meta}
        present = np.array([meta["left_present"], meta["right_present"]], dtype=np.float32)
        detected = np.array([meta["left_detected"], meta["right_detected"]], dtype=np.float32)
        scores = np.array([meta["left_score"], meta["right_score"]], dtype=np.float32)
        return HolisticFrameResult(
            vector=vector,
            left=meta["left"],
            right=meta["right"],
            shoulders=meta["shoulders"],
            detected=detected,
            held=np.zeros(2, dtype=np.float32),
            present=present,
            scores=scores,
            fps_infer=1000.0 / max(elapsed_ms, 1e-6),
            hand_ms=elapsed_ms,
            pose_ms=0.0,
            feature_mode=self.spec.feature_mode,
            raw_full=raw_full,
        )


class _FakeLandmarks:
    """Minimal landmark container so stabilized hand arrays look like MediaPipe output."""

    __slots__ = ("landmark",)

    def __init__(self, points: np.ndarray) -> None:
        from types import SimpleNamespace

        self.landmark = [
            SimpleNamespace(x=float(p[0]), y=float(p[1]), z=float(p[2])) for p in points
        ]


class _StabilizedResults:
    """Holistic result with the two hands replaced by stabilized landmarks."""

    __slots__ = ("pose_landmarks", "pose_world_landmarks", "face_landmarks", "left_hand_landmarks", "right_hand_landmarks")

    def __init__(self, results: Any, left_pts: np.ndarray | None, right_pts: np.ndarray | None) -> None:
        self.pose_landmarks = results.pose_landmarks
        self.pose_world_landmarks = getattr(results, "pose_world_landmarks", None)
        self.face_landmarks = results.face_landmarks
        self.left_hand_landmarks = _FakeLandmarks(left_pts) if left_pts is not None else None
        self.right_hand_landmarks = _FakeLandmarks(right_pts) if right_pts is not None else None


class StabilizedHolisticLiveExtractor(HolisticLiveExtractor):
    """Forward-only stabilized variant: 1€ smoothing + jump rejection + hold on hands.

    Live-safe (never looks ahead). Reuses :func:`build_paper_feature` by swapping in
    stabilized hand landmarks, so it works for every smart180/face-ref schema.
    """

    def __init__(self, *args, stabilizer_config=None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        import landmark_stabilizer as lstab

        cfg = stabilizer_config or lstab.StabilizerConfig()
        self._stab_left = lstab.LandmarkStreamStabilizer(cfg)
        self._stab_right = lstab.LandmarkStreamStabilizer(cfg)
        self.backend_name = "studio_holistic_stabilized"
        self.backend_detail = "mp.solutions.holistic.Holistic + landmark_stabilizer"

    def process(self, frame_bgr: np.ndarray) -> HolisticFrameResult:
        work = resize_width(frame_bgr, self.proc_width)
        rgb = cv2.cvtColor(work, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        t0 = time.perf_counter()
        results = self.holistic.process(rgb)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        now = time.perf_counter()
        dt = (now - self._last_t) if self._last_t is not None else (1.0 / max(self.spec.target_fps, 1e-6))
        self._last_t = now

        shoulders = _shoulders_from_pose(results.pose_landmarks)
        scale = 0.28
        if np.isfinite(shoulders[:, :2]).all():
            width = float(np.linalg.norm(shoulders[0, :2] - shoulders[1, :2]))
            if width > 1e-4:
                scale = width

        left_arr = _landmarks_xyz(results.left_hand_landmarks, 21) if results.left_hand_landmarks is not None else None
        right_arr = _landmarks_xyz(results.right_hand_landmarks, 21) if results.right_hand_landmarks is not None else None
        step_l = self._stab_left.update(left_arr, results.left_hand_landmarks is not None, scale, dt)
        step_r = self._stab_right.update(right_arr, results.right_hand_landmarks is not None, scale, dt)
        proxy = _StabilizedResults(
            results,
            step_l.points if step_l.present else None,
            step_r.points if step_r.present else None,
        )

        vector, meta = build_paper_feature(self.spec, proxy, state=self._face_state, dt=dt)
        raw_vector, raw_meta = build_full_landmark_feature(proxy)
        raw_full = {"vector": raw_vector, "meta": raw_meta}
        present = np.array([meta["left_present"], meta["right_present"]], dtype=np.float32)
        detected = np.array([meta["left_detected"], meta["right_detected"]], dtype=np.float32)
        scores = np.array([meta["left_score"], meta["right_score"]], dtype=np.float32)
        return HolisticFrameResult(
            vector=vector,
            left=meta["left"],
            right=meta["right"],
            shoulders=meta["shoulders"],
            detected=detected,
            held=np.array([float(step_l.held), float(step_r.held)], dtype=np.float32),
            present=present,
            scores=scores,
            fps_infer=1000.0 / max(elapsed_ms, 1e-6),
            hand_ms=elapsed_ms,
            pose_ms=0.0,
            feature_mode=self.spec.feature_mode,
            raw_full=raw_full,
        )


class MultiSchemaHolisticExtractor:
    """One MediaPipe Holistic pass that emits multiple paper feature schemas.

    This is used by live dataset recording so khukuh1629 and adi1662 can be
    captured from the same camera frame without running Holistic twice.
    """

    def __init__(
        self,
        schemas: Sequence[str | fs.FeatureSchema],
        proc_width: int = 384,
        det_conf: float = 0.40,
        track_conf: float = 0.45,
        model_complexity: int = 0,
        smooth_landmarks: bool = True,
        refine_face_landmarks: bool = False,
    ) -> None:
        self.specs = [fs.get_schema(schema) for schema in schemas]
        if not self.specs:
            raise ValueError("schemas tidak boleh kosong")
        for spec in self.specs:
            if spec.extractor != "holistic":
                raise ValueError(f"MultiSchemaHolisticExtractor hanya untuk schema holistic, got {spec.name}")
        self.proc_width = int(proc_width)
        # Per-schema temporal state + last-frame timestamp for face-ref velocity terms.
        self._face_states: dict[str, dict] = {spec.name: {} for spec in self.specs}
        self._last_t: float | None = None
        self.holistic = mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=int(model_complexity),
            smooth_landmarks=bool(smooth_landmarks),
            refine_face_landmarks=bool(refine_face_landmarks),
            enable_segmentation=False,
            min_detection_confidence=float(det_conf),
            min_tracking_confidence=float(track_conf),
        )

    def close(self) -> None:
        self.holistic.close()

    def process(self, frame_bgr: np.ndarray) -> dict[str, HolisticFrameResult]:
        work = resize_width(frame_bgr, self.proc_width)
        rgb = cv2.cvtColor(work, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        t0 = time.perf_counter()
        results = self.holistic.process(rgb)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        now = time.perf_counter()
        dt = (now - self._last_t) if self._last_t is not None else 0.1
        self._last_t = now
        raw_vector, raw_meta = build_full_landmark_feature(results)
        raw_full = {"vector": raw_vector, "meta": raw_meta}
        out: dict[str, HolisticFrameResult] = {}
        for spec in self.specs:
            vector, meta = build_paper_feature(spec, results, state=self._face_states[spec.name], dt=dt)
            present = np.array([meta["left_present"], meta["right_present"]], dtype=np.float32)
            detected = np.array([meta["left_detected"], meta["right_detected"]], dtype=np.float32)
            scores = np.array([meta["left_score"], meta["right_score"]], dtype=np.float32)
            out[spec.name] = HolisticFrameResult(
                vector=vector,
                left=meta["left"],
                right=meta["right"],
                shoulders=meta["shoulders"],
                detected=detected,
                held=np.zeros(2, dtype=np.float32),
                present=present,
                scores=scores,
                fps_infer=1000.0 / max(elapsed_ms, 1e-6),
                hand_ms=elapsed_ms,
                pose_ms=0.0,
                feature_mode=spec.feature_mode,
                raw_full=raw_full,
            )
        return out


def _read_frame_at(cap: cv2.VideoCapture, frame_idx_1based: int) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(frame_idx_1based) - 1))
    ok, frame = cap.read()
    return frame if ok else None


def extract_video_arrays(
    video_path: str | Path,
    schema: str | fs.FeatureSchema,
    args: Namespace | None = None,
    include_frames: bool = False,
) -> dict[str, Any] | None:
    spec = fs.get_schema(schema)
    args = args or make_holistic_args(target_fps=spec.target_fps)
    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERR] Cannot open video: {video_path}")
        return None

    src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_sec = total_frames / max(src_fps, 1e-6) if total_frames > 0 else 0.0
    target_fps = float(getattr(args, "target_fps", spec.target_fps))
    target_dt = 1.0 / max(target_fps, 1e-6)
    n_targets = int(math.floor(duration_sec * target_fps + 1e-6)) + 1 if duration_sec > 0 else total_frames

    extractor = HolisticLiveExtractor(
        spec,
        proc_width=int(getattr(args, "proc_width", 384)),
        det_conf=float(getattr(args, "det_conf", 0.40)),
        track_conf=float(getattr(args, "track_conf", 0.45)),
        model_complexity=int(getattr(args, "model_complexity", 0)),
        smooth_landmarks=bool(getattr(args, "smooth_landmarks", True)),
        refine_face_landmarks=bool(getattr(args, "refine_face_landmarks", False)),
    )

    features: list[np.ndarray] = []
    meta_rows: list[dict[str, Any]] = []
    overlay_frames: list[np.ndarray] = []
    skeleton_frames: list[np.ndarray] = []

    try:
        for out_i in range(max(0, n_targets)):
            t_sec = out_i * target_dt
            base_idx = int(round(t_sec * src_fps)) + 1
            if total_frames and base_idx > total_frames:
                break
            raw = _read_frame_at(cap, base_idx)
            if raw is None:
                break
            if getattr(args, "width", 0) and getattr(args, "height", 0):
                raw = cv2.resize(raw, (int(args.width), int(args.height)))
            work = center_crop(raw, float(getattr(args, "center_crop", 1.0)))
            res = extractor.process(work)
            features.append(res.vector.astype(np.float32))
            meta_rows.append(
                {
                    "out_index": int(out_i),
                    "time_sec": float(t_sec),
                    "target_source_frame": int(base_idx),
                    "chosen_source_frame": int(base_idx),
                    "enhance_mode": "holistic",
                    "left_present": float(res.present[0]),
                    "right_present": float(res.present[1]),
                    "left_detected": float(res.detected[0]),
                    "right_detected": float(res.detected[1]),
                    "left_held": 0.0,
                    "right_held": 0.0,
                    "left_score": float(res.scores[0]),
                    "right_score": float(res.scores[1]),
                    "hand_ms": float(res.hand_ms),
                    "pose_ms": float(res.pose_ms),
                }
            )
            if include_frames:
                overlay = work.copy()
                draw_shoulders(overlay, res.shoulders)
                draw_simple_hand(overlay, res.left, (0, 255, 0))
                draw_simple_hand(overlay, res.right, (0, 180, 255))
                cv2.putText(
                    overlay,
                    f"{video_path.name} | {spec.name}:{spec.feature_dim} | {target_fps:g}fps",
                    (8, 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (245, 245, 245),
                    1,
                    cv2.LINE_AA,
                )
                skeleton = np.zeros_like(work)
                draw_shoulders(skeleton, res.shoulders)
                draw_simple_hand(skeleton, res.left, (0, 255, 0))
                draw_simple_hand(skeleton, res.right, (0, 180, 255))
                overlay_frames.append(overlay)
                skeleton_frames.append(skeleton)
    finally:
        extractor.close()
        cap.release()

    if not features:
        print(f"[ERR] No features extracted: {video_path}")
        return None

    arr = np.stack(features).astype(np.float32)
    return {
        "features": arr,
        "frames": meta_rows,
        "overlay_frames": overlay_frames,
        "skeleton_frames": skeleton_frames,
        "input_video": str(video_path),
        "source_fps": float(src_fps),
        "source_frame_count": int(total_frames),
        "source_duration_sec": float(duration_sec),
        "target_fps": target_fps,
        "feature_mode": spec.feature_mode,
        "feature_dim": spec.feature_dim,
        "feature_version": spec.feature_schema,
        "extract_profile": "mediapipe_holistic_10fps",
        "smart_mode": "holistic",
        "settings": vars(args),
    }
