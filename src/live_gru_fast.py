"""Low-overhead live inference for the GRU BISINDO models."""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass, field
import queue
import threading
import time
from typing import Any

import cv2
import numpy as np
import torch

import feature_schemas as fs
import gru_manager as gm
import jetson_runtime as jr
from smart_extract import contract as sc
from smart_extract.live_bisindo_mp_real_shoulder_v6 import (
    LatestFrameCamera,
    center_crop,
    draw_shoulders,
    draw_simple_hand,
)


LIVE_PROFILES: dict[str, dict[str, float | int | str]] = {
    "lossless1080_10": {
        "width": 1920,
        "height": 1080,
        "camera_fps": 30,
        "display_width": 960,
        "proc_width": 960,
        "pose_proc_width": 960,
        "pose_every": 3,
        "hand_every": 1,
        "shoulder_backend": "mp-pose",
        "hold_frames": 5,
        "model_complexity": 1,
        "refine_face_landmarks": 1,
        "det_conf": 0.50,
        "track_conf": 0.50,
        "smooth_alpha": 0.78,
        "shoulder_smooth_alpha": 0.35,
        "predict_interval": 0.0,
        "append_interval": 1.0 / sc.TARGET_FPS,
        "status_interval": 0.12,
        "reset_after": 0.60,
        "debounce_hits": 1,
        "torch_threads": 1,
        "use_gstreamer": 0,
        "min_segment_frames": 8,
        "max_segment_frames": 55,
        "end_idle_samples": 5,
        "end_still_samples": 5,
        "motion_start": 0.010,
        "motion_end": 0.006,
    },
    "accurate10": {
        "width": 640,
        "height": 480,
        "camera_fps": 30,
        "proc_width": 384,
        "pose_proc_width": 256,
        "pose_every": 3,
        "hand_every": 1,
        "shoulder_backend": "mp-pose",
        "hold_frames": 5,
        "det_conf": 0.40,
        "track_conf": 0.45,
        "smooth_alpha": 0.78,
        "shoulder_smooth_alpha": 0.35,
        "predict_interval": 0.0,
        "append_interval": 1.0 / sc.TARGET_FPS,
        "status_interval": 0.12,
        "reset_after": 0.60,
        "debounce_hits": 1,
        "torch_threads": 1,
        "use_gstreamer": 0,
        "min_segment_frames": 8,
        "max_segment_frames": 55,
        "end_idle_samples": 5,
        "end_still_samples": 5,
        "motion_start": 0.010,
        "motion_end": 0.006,
    },
    "fast10": {
        "width": 480,
        "height": 360,
        "camera_fps": 30,
        "proc_width": 256,
        "pose_proc_width": 160,
        "pose_every": 8,
        "hand_every": 1,
        "shoulder_backend": "mp-pose",
        "hold_frames": 4,
        "det_conf": 0.40,
        "track_conf": 0.45,
        "smooth_alpha": 0.80,
        "shoulder_smooth_alpha": 0.35,
        "predict_interval": 0.0,
        "append_interval": 1.0 / sc.TARGET_FPS,
        "status_interval": 0.12,
        "reset_after": 0.60,
        "debounce_hits": 1,
        "torch_threads": 1,
        "use_gstreamer": 0,
        "min_segment_frames": 8,
        "max_segment_frames": 55,
        "end_idle_samples": 5,
        "end_still_samples": 5,
        "motion_start": 0.010,
        "motion_end": 0.006,
    },
    "jetson10": {
        "width": 424,
        "height": 240,
        "camera_fps": 30,
        "proc_width": 160,
        "pose_proc_width": 96,
        "pose_every": 9999,
        "hand_every": 2,
        "shoulder_backend": "none",
        "hold_frames": 4,
        "det_conf": 0.40,
        "track_conf": 0.45,
        "smooth_alpha": 0.84,
        "shoulder_smooth_alpha": 0.35,
        "predict_interval": 0.55,
        "append_interval": 0.08,
        "status_interval": 0.12,
        "reset_after": 0.50,
        "debounce_hits": 2,
        "torch_threads": 1,
        "use_gstreamer": 0,
    },
    "lite": {
        "width": 424,
        "height": 240,
        "camera_fps": 30,
        "proc_width": 160,
        "pose_proc_width": 96,
        "pose_every": 9999,
        "hand_every": 2,
        "shoulder_backend": "none",
        "hold_frames": 4,
        "det_conf": 0.40,
        "track_conf": 0.45,
        "smooth_alpha": 0.84,
        "shoulder_smooth_alpha": 0.35,
        "predict_interval": 0.55,
        "append_interval": 0.08,
        "status_interval": 0.12,
        "reset_after": 0.50,
        "debounce_hits": 2,
        "torch_threads": 1,
        "use_gstreamer": 0,
    },
    "ultra": {
        "width": 480,
        "height": 360,
        "camera_fps": 30,
        "proc_width": 192,
        "pose_proc_width": 128,
        "pose_every": 12,
        "hand_every": 2,
        "shoulder_backend": "mp-pose",
        "hold_frames": 4,
        "det_conf": 0.40,
        "track_conf": 0.45,
        "smooth_alpha": 0.82,
        "shoulder_smooth_alpha": 0.35,
        "predict_interval": 0.45,
        "append_interval": 0.08,
        "status_interval": 0.12,
        "reset_after": 0.50,
        "debounce_hits": 2,
        "torch_threads": 1,
        "use_gstreamer": 0,
    },
    "fast": {
        "width": 640,
        "height": 480,
        "camera_fps": 30,
        "proc_width": 256,
        "pose_proc_width": 160,
        "pose_every": 8,
        "hand_every": 2,
        "shoulder_backend": "mp-pose",
        "hold_frames": 4,
        "det_conf": 0.40,
        "track_conf": 0.45,
        "smooth_alpha": 0.80,
        "shoulder_smooth_alpha": 0.35,
        "predict_interval": 0.30,
        "append_interval": 0.08,
        "status_interval": 0.12,
        "reset_after": 0.50,
        "debounce_hits": 2,
        "torch_threads": 1,
        "use_gstreamer": 0,
    },
    "quality": {
        "width": 640,
        "height": 480,
        "camera_fps": 30,
        "proc_width": 384,
        "pose_proc_width": 256,
        "pose_every": 3,
        "hand_every": 1,
        "shoulder_backend": "mp-pose",
        "hold_frames": 5,
        "det_conf": 0.40,
        "track_conf": 0.45,
        "smooth_alpha": 0.78,
        "shoulder_smooth_alpha": 0.35,
        "predict_interval": 0.35,
        "append_interval": 1.0 / sc.TARGET_FPS,
        "status_interval": 0.12,
        "reset_after": 0.50,
        "debounce_hits": 2,
        "torch_threads": 1,
        "use_gstreamer": 0,
    },
}
DEFAULT_LIVE_PROFILE = "lossless1080_10"
PROFILE_ALIASES = {
    "accurate": "accurate10",
    "best": DEFAULT_LIVE_PROFILE,
    "balanced": DEFAULT_LIVE_PROFILE,
    "1080": DEFAULT_LIVE_PROFILE,
    "lossless": DEFAULT_LIVE_PROFILE,
    "turbo": "fast10",
    "tflite": "fast10",
    "rt-lite": "fast10",
    "jetson": "fast10",
}


def normalize_profile(profile: str | None) -> str:
    value = str(profile or DEFAULT_LIVE_PROFILE).strip().lower()
    value = PROFILE_ALIASES.get(value, value)
    if value not in LIVE_PROFILES:
        raise ValueError(f"Unknown live profile '{profile}'. Pilih: {', '.join(sorted(LIVE_PROFILES))}")
    return value


def live_display_size(frame_width: int, frame_height: int, display_width: int | None) -> tuple[int, int]:
    frame_width = max(1, int(frame_width))
    frame_height = max(1, int(frame_height))
    display_width = int(display_width or 0)
    if display_width <= 0 or frame_width <= display_width:
        return frame_width, frame_height
    scale = display_width / float(frame_width)
    return display_width, max(1, int(round(frame_height * scale)))


def resize_live_preview(frame: np.ndarray, display_width: int | None) -> np.ndarray:
    target_width, target_height = live_display_size(frame.shape[1], frame.shape[0], display_width)
    if target_width == frame.shape[1] and target_height == frame.shape[0]:
        return frame
    return cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)


def live_profile_status(profile: str | None) -> dict[str, int]:
    profile_name = normalize_profile(profile)
    cfg = LIVE_PROFILES[profile_name]
    display_width, display_height = live_display_size(
        int(cfg["width"]),
        int(cfg["height"]),
        int(cfg.get("display_width", cfg["width"])),
    )
    return {
        "capture_width": int(cfg["width"]),
        "capture_height": int(cfg["height"]),
        "camera_fps": int(cfg.get("camera_fps", 30)),
        "proc_width": int(cfg["proc_width"]),
        "pose_proc_width": int(cfg.get("pose_proc_width", cfg["proc_width"])),
        "display_width": int(display_width),
        "display_height": int(display_height),
    }


def normalize_live_variant(variant: str) -> str:
    value = str(variant or "auto").strip().lower()
    if value in {"auto", "best"}:
        return "auto"
    return gm.normalize_variant_name(value)


def configure_torch_for_live(num_threads: int = 1) -> None:
    try:
        torch.set_num_threads(max(1, int(num_threads)))
    except Exception:
        pass
    try:
        torch.set_num_interop_threads(1)
    except Exception:
        pass


def maybe_trace_model(
    model: torch.nn.Module,
    target_frames: int,
    device: torch.device,
    enabled: bool = True,
    feature_dim: int = sc.FEATURE_DIM,
) -> torch.nn.Module:
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


@dataclass
class PredictionResult:
    request_id: int = 0
    label: str = "-"
    confidence: float = 0.0
    top: list[tuple[str, float]] = field(default_factory=list)
    model_ms: float = 0.0
    fps_predict: float = 0.0
    error: str | None = None


@dataclass
class ExtractedFrameResult:
    request_id: int = 0
    frame: np.ndarray | None = None
    result: Any | None = None
    extract_ms: float = 0.0
    submitted_at: float = 0.0
    completed_at: float = 0.0
    error: str | None = None


class AsyncMediaPipePool:
    """Latest-only MediaPipe worker pool.

    The pending slot is replaced by new frames, so slow extraction never builds
    an old-frame backlog. Multiple workers can consume pending frames, but there
    is still only one latest pending frame by design.
    """

    def __init__(self, extractor_factory, workers: int = 1) -> None:
        self.extractor_factory = extractor_factory
        self.workers = max(1, int(workers))
        self._stop_event = threading.Event()
        self._condition = threading.Condition()
        self._pending: tuple[int, np.ndarray, float] | None = None
        self._pending_id = 0
        self._latest = ExtractedFrameResult()
        self._latest_lock = threading.Lock()
        self._threads: list[threading.Thread] = []

    def start(self) -> "AsyncMediaPipePool":
        for idx in range(self.workers):
            thread = threading.Thread(target=self._run_worker, name=f"mp-worker-{idx}", daemon=True)
            thread.start()
            self._threads.append(thread)
        return self

    def submit(self, frame: np.ndarray) -> int:
        with self._condition:
            self._pending_id += 1
            request_id = self._pending_id
            self._pending = (request_id, frame.copy(), time.perf_counter())
            self._condition.notify()
        return request_id

    def latest(self) -> ExtractedFrameResult:
        with self._latest_lock:
            return ExtractedFrameResult(
                request_id=self._latest.request_id,
                frame=None if self._latest.frame is None else self._latest.frame.copy(),
                result=self._latest.result,
                extract_ms=self._latest.extract_ms,
                submitted_at=self._latest.submitted_at,
                completed_at=self._latest.completed_at,
                error=self._latest.error,
            )

    def stop(self) -> None:
        self._stop_event.set()
        with self._condition:
            self._pending = None
            self._condition.notify_all()

    def join(self, timeout: float = 1.0) -> bool:
        deadline = time.perf_counter() + max(0.0, float(timeout))
        for thread in self._threads:
            remaining = max(0.0, deadline - time.perf_counter())
            thread.join(timeout=remaining)
        return any(thread.is_alive() for thread in self._threads)

    def _run_worker(self) -> None:
        extractor = None
        try:
            extractor = self.extractor_factory()
            while not self._stop_event.is_set():
                with self._condition:
                    while self._pending is None and not self._stop_event.is_set():
                        self._condition.wait(timeout=0.10)
                    if self._stop_event.is_set():
                        break
                    request_id, frame, submitted_at = self._pending
                    self._pending = None
                t0 = time.perf_counter()
                try:
                    result = extractor.process(frame)
                    item = ExtractedFrameResult(
                        request_id=request_id,
                        frame=frame,
                        result=result,
                        extract_ms=(time.perf_counter() - t0) * 1000.0,
                        submitted_at=submitted_at,
                        completed_at=time.perf_counter(),
                    )
                except Exception as exc:
                    item = ExtractedFrameResult(request_id=request_id, error=str(exc), submitted_at=submitted_at, completed_at=time.perf_counter())
                with self._latest_lock:
                    if item.request_id >= self._latest.request_id:
                        self._latest = item
        finally:
            if extractor is not None:
                extractor.close()


class AsyncGRUPredictor(threading.Thread):
    """Runs model inference off the camera/extractor thread.

    Submit replaces any older pending request, so a slow GRU never makes the
    camera wait through stale windows.
    """

    def __init__(
        self,
        model: torch.nn.Module | None,
        labels: dict[int, str],
        target_frames: int,
        device: torch.device,
        feature_dim: int = sc.FEATURE_DIM,
        predict_fn=None,
    ) -> None:
        super().__init__(daemon=True)
        self.model = model
        self.labels = labels
        self.target_frames = int(target_frames)
        self.device = device
        self.feature_dim = int(feature_dim)
        self.predict_fn = predict_fn
        self._stop_event = threading.Event()
        self._condition = threading.Condition()
        self._pending: np.ndarray | None = None
        self._pending_id = 0
        self._latest = PredictionResult()
        self._latest_lock = threading.Lock()
        self._done_times: deque[float] = deque(maxlen=12)

    def submit(self, sequence: np.ndarray) -> int:
        arr = sc.ensure_feature_dim(sequence, self.feature_dim).astype(np.float32, copy=True)
        with self._condition:
            self._pending_id += 1
            request_id = self._pending_id
            self._pending = arr
            self._condition.notify()
        return request_id

    def latest(self) -> PredictionResult:
        with self._latest_lock:
            return PredictionResult(
                request_id=self._latest.request_id,
                label=self._latest.label,
                confidence=self._latest.confidence,
                top=list(self._latest.top),
                model_ms=self._latest.model_ms,
                fps_predict=self._latest.fps_predict,
                error=self._latest.error,
            )

    def stop(self) -> None:
        self._stop_event.set()
        with self._condition:
            self._pending = None
            self._condition.notify_all()

    def run(self) -> None:
        while not self._stop_event.is_set():
            with self._condition:
                while self._pending is None and not self._stop_event.is_set():
                    self._condition.wait(timeout=0.10)
                if self._stop_event.is_set():
                    break
                sequence = self._pending
                request_id = self._pending_id
                self._pending = None
            if sequence is None:
                continue
            t0 = time.perf_counter()
            try:
                if self.predict_fn is not None:
                    label, confidence, top = self.predict_fn(sequence)
                else:
                    if self.model is None:
                        raise RuntimeError("Predictor model belum di-set")
                    label, confidence, top = gm.predict_sequence(
                        self.model,
                        sequence,
                        self.labels,
                        self.target_frames,
                        self.device,
                        feature_dim=self.feature_dim,
                    )
                model_ms = (time.perf_counter() - t0) * 1000.0
                now = time.perf_counter()
                self._done_times.append(now)
                fps_predict = 0.0
                if len(self._done_times) >= 2:
                    fps_predict = (len(self._done_times) - 1) / max(self._done_times[-1] - self._done_times[0], 1e-6)
                result = PredictionResult(
                    request_id=request_id,
                    label=label,
                    confidence=confidence,
                    top=top,
                    model_ms=model_ms,
                    fps_predict=fps_predict,
                )
            except Exception as exc:
                result = PredictionResult(request_id=request_id, error=str(exc))
            with self._latest_lock:
                self._latest = result


class AsyncGRUPredictorPool:
    """Latest-only predictor pool with the same public API as AsyncGRUPredictor."""

    def __init__(
        self,
        model: torch.nn.Module | None,
        labels: dict[int, str],
        target_frames: int,
        device: torch.device,
        feature_dim: int = sc.FEATURE_DIM,
        predict_fn=None,
        workers: int = 1,
    ) -> None:
        self.model = model
        self.labels = labels
        self.target_frames = int(target_frames)
        self.device = device
        self.feature_dim = int(feature_dim)
        self.predict_fn = predict_fn
        self.workers = max(1, int(workers))
        self._stop_event = threading.Event()
        self._condition = threading.Condition()
        self._pending: np.ndarray | None = None
        self._pending_id = 0
        self._latest = PredictionResult()
        self._latest_lock = threading.Lock()
        self._done_times: deque[float] = deque(maxlen=12)
        self._threads: list[threading.Thread] = []

    def start(self) -> "AsyncGRUPredictorPool":
        for idx in range(self.workers):
            thread = threading.Thread(target=self._run_worker, name=f"infer-worker-{idx}", daemon=True)
            thread.start()
            self._threads.append(thread)
        return self

    def submit(self, sequence: np.ndarray) -> int:
        arr = sc.ensure_feature_dim(sequence, self.feature_dim).astype(np.float32, copy=True)
        with self._condition:
            self._pending_id += 1
            request_id = self._pending_id
            self._pending = arr
            self._condition.notify()
        return request_id

    def latest(self) -> PredictionResult:
        with self._latest_lock:
            return PredictionResult(
                request_id=self._latest.request_id,
                label=self._latest.label,
                confidence=self._latest.confidence,
                top=list(self._latest.top),
                model_ms=self._latest.model_ms,
                fps_predict=self._latest.fps_predict,
                error=self._latest.error,
            )

    def stop(self) -> None:
        self._stop_event.set()
        with self._condition:
            self._pending = None
            self._condition.notify_all()

    def join(self, timeout: float = 1.0) -> bool:
        deadline = time.perf_counter() + max(0.0, float(timeout))
        for thread in self._threads:
            remaining = max(0.0, deadline - time.perf_counter())
            thread.join(timeout=remaining)
        return any(thread.is_alive() for thread in self._threads)

    def _predict(self, sequence: np.ndarray) -> tuple[str, float, list[tuple[str, float]]]:
        if self.predict_fn is not None:
            return self.predict_fn(sequence)
        if self.model is None:
            raise RuntimeError("Predictor model belum di-set")
        return gm.predict_sequence(
            self.model,
            sequence,
            self.labels,
            self.target_frames,
            self.device,
            feature_dim=self.feature_dim,
        )

    def _run_worker(self) -> None:
        while not self._stop_event.is_set():
            with self._condition:
                while self._pending is None and not self._stop_event.is_set():
                    self._condition.wait(timeout=0.10)
                if self._stop_event.is_set():
                    break
                sequence = self._pending
                request_id = self._pending_id
                self._pending = None
            if sequence is None:
                continue
            t0 = time.perf_counter()
            try:
                label, confidence, top = self._predict(sequence)
                model_ms = (time.perf_counter() - t0) * 1000.0
                now = time.perf_counter()
                self._done_times.append(now)
                fps_predict = 0.0
                if len(self._done_times) >= 2:
                    fps_predict = (len(self._done_times) - 1) / max(self._done_times[-1] - self._done_times[0], 1e-6)
                result = PredictionResult(
                    request_id=request_id,
                    label=label,
                    confidence=confidence,
                    top=top,
                    model_ms=model_ms,
                    fps_predict=fps_predict,
                )
            except Exception as exc:
                result = PredictionResult(request_id=request_id, error=str(exc))
            with self._latest_lock:
                if result.request_id >= self._latest.request_id:
                    self._latest = result


class LiveSequenceBuffer:
    """Collects visible hand frames and exposes complete model windows only."""

    def __init__(self, target_frames: int, append_interval: float, reset_after: float = 0.50, feature_dim: int = sc.FEATURE_DIM) -> None:
        self.target_frames = int(target_frames)
        self.append_interval = float(append_interval)
        self.reset_after = float(reset_after)
        self.feature_dim = int(feature_dim)
        self.buffer: deque[np.ndarray] = deque(maxlen=self.target_frames)
        self.last_append = -1e9
        self.last_visible: float | None = None
        self.was_reset = False

    def update(self, vector: np.ndarray, visible: bool, now: float) -> bool:
        self.was_reset = False
        if visible:
            self.last_visible = float(now)
            if now - self.last_append >= self.append_interval:
                self.buffer.append(sc.ensure_feature_dim(vector, self.feature_dim)[0].astype(np.float32, copy=True))
                self.last_append = float(now)
                return True
            return False

        if self.last_visible is not None and now - self.last_visible >= self.reset_after:
            self.reset()
        return False

    def reset(self) -> None:
        self.buffer.clear()
        self.last_visible = None
        self.last_append = -1e9
        self.was_reset = True

    @property
    def ready(self) -> bool:
        return len(self.buffer) >= self.target_frames

    def window(self) -> np.ndarray:
        return np.asarray(self.buffer, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.buffer)


@dataclass
class SegmentUpdate:
    sampled: bool = False
    finalized: np.ndarray | None = None
    reason: str = ""
    motion: float = 0.0


class LiveGestureSegmenter:
    """Samples live features at 10 FPS and emits trimmed gesture segments."""

    def __init__(
        self,
        append_interval: float,
        schema: str = fs.DEFAULT_SCHEMA,
        min_frames: int = 8,
        max_frames: int = 55,
        end_idle_samples: int = 5,
        end_still_samples: int = 5,
        motion_start: float = 0.010,
        motion_end: float = 0.006,
    ) -> None:
        self.append_interval = float(append_interval)
        self.schema = fs.normalize_schema_name(schema)
        self.feature_dim = fs.get_schema(self.schema).feature_dim
        self.min_frames = max(1, int(min_frames))
        self.max_frames = max(self.min_frames, int(max_frames))
        self.end_idle_samples = max(1, int(end_idle_samples))
        self.end_still_samples = max(1, int(end_still_samples))
        self.motion_start = float(motion_start)
        self.motion_end = float(motion_end)
        self.samples: list[np.ndarray] = []
        self.sample_times: list[float] = []
        self.active = False
        self.last_sample = -1e9
        self.last_vector: np.ndarray | None = None
        self.idle_samples = 0
        self.still_samples = 0
        self.last_reason = ""
        self.last_segment_len = 0
        self.last_segment_ms = 0.0
        self.was_reset = False

    def reset(self, reason: str = "reset") -> None:
        self.samples.clear()
        self.sample_times.clear()
        self.active = False
        self.last_vector = None
        self.idle_samples = 0
        self.still_samples = 0
        self.last_reason = reason
        self.was_reset = True

    def _duration_ms(self, times: list[float] | None = None) -> float:
        values = self.sample_times if times is None else times
        if len(values) < 2:
            return 0.0
        return max(0.0, (values[-1] - values[0]) * 1000.0)

    def _finalize(self, reason: str) -> np.ndarray | None:
        if len(self.samples) < self.min_frames:
            self.reset(f"drop_short:{reason}")
            return None
        arr = np.asarray(self.samples, dtype=np.float32)
        times = list(self.sample_times)
        self.last_segment_len = len(arr)
        self.last_segment_ms = self._duration_ms(times)
        self.reset(reason)
        return arr

    def update(self, vector: np.ndarray, visible: bool, now: float) -> SegmentUpdate:
        self.was_reset = False
        now = float(now)
        if now - self.last_sample < self.append_interval:
            return SegmentUpdate(motion=0.0)
        self.last_sample = now

        arr = sc.ensure_feature_dim(vector, self.feature_dim)[0].astype(np.float32, copy=True)
        motion, _ = fs.motion_score(self.schema, self.last_vector, arr)
        self.last_vector = arr.copy()
        moving = motion >= self.motion_start if not self.active else motion >= self.motion_end

        if not self.active:
            if not visible:
                return SegmentUpdate(sampled=True, motion=motion)
            self.active = True
            self.samples = [arr]
            self.sample_times = [now]
            self.idle_samples = 0
            self.still_samples = 0
            return SegmentUpdate(sampled=True, motion=motion, reason="start")

        if visible:
            self.samples.append(arr)
            self.sample_times.append(now)
            self.idle_samples = 0
            self.still_samples = 0 if moving else self.still_samples + 1
        else:
            self.idle_samples += 1
            self.still_samples += 1

        reason = ""
        finalized = None
        if len(self.samples) >= self.max_frames:
            reason = "max_len"
        elif len(self.samples) >= self.min_frames and self.idle_samples >= self.end_idle_samples:
            reason = "idle"
        elif len(self.samples) >= self.min_frames and self.still_samples >= self.end_still_samples:
            reason = "still"
        elif len(self.samples) < self.min_frames and self.idle_samples >= self.end_idle_samples:
            self.reset("drop_short:idle")
            return SegmentUpdate(sampled=True, reason=self.last_reason, motion=motion)

        if reason:
            finalized = self._finalize(reason)
        return SegmentUpdate(sampled=True, finalized=finalized, reason=reason, motion=motion)

    @property
    def current_len(self) -> int:
        return len(self.samples)

    @property
    def current_ms(self) -> float:
        return self._duration_ms()

    @property
    def sample_fps(self) -> float:
        duration = self.current_ms / 1000.0
        if duration <= 1e-6 or len(self.samples) < 2:
            return 0.0
        return (len(self.samples) - 1) / duration


class LabelDebouncer:
    def __init__(self, threshold: float, hits_required: int = 2) -> None:
        self.threshold = float(threshold)
        self.hits_required = max(1, int(hits_required))
        self.candidate = "-"
        self.hits = 0
        self.stable_label = "-"
        self.stable_confidence = 0.0

    def reset(self) -> tuple[str, float]:
        self.candidate = "-"
        self.hits = 0
        self.stable_label = "-"
        self.stable_confidence = 0.0
        return self.stable_label, self.stable_confidence

    def update(self, label: str, confidence: float) -> tuple[str, float]:
        confidence = float(confidence)
        if label == "-" or confidence < self.threshold:
            return self.reset()
        if label == self.candidate:
            self.hits += 1
        else:
            self.candidate = label
            self.hits = 1
        if self.hits >= self.hits_required:
            self.stable_label = label
            self.stable_confidence = confidence
        return self.stable_label, self.stable_confidence


class FastGRULiveWorker(threading.Thread):
    """Threaded OpenCV live runner used by both CLI and Tkinter UI."""

    def __init__(
        self,
        variant: str = "auto",
        mp_device: str = "CPU",
        status_queue: queue.Queue | None = None,
        performance_mode: str = DEFAULT_LIVE_PROFILE,
        profile: str | None = None,
        device: str = "auto",
        camera_index: int = 0,
        model_dir: str | None = None,
        confidence_threshold: float = 0.25,
        show_window: bool = True,
        use_jit: bool = True,
        segment_mode: str = "auto",
        schema: str = fs.DEFAULT_SCHEMA,
        route: str = "main",
        stream_workers: int = 1,
        mp_workers: int = 1,
        inference_workers: int = 1,
        include_sequences: bool = False,
        mp_method: str = "holistic",
    ) -> None:
        super().__init__(daemon=True)
        self.requested_variant = normalize_live_variant(variant)
        self.variant = self.requested_variant
        self.mp_device = str(mp_device or "CPU")
        self.status_queue = status_queue
        self.profile = normalize_profile(profile or performance_mode)
        self.device_name = str(device or "auto").lower()
        self.camera_index = int(camera_index)
        self.model_dir = model_dir or str(gm.MODEL_DIR)
        self.confidence_threshold = float(confidence_threshold)
        self.show_window = bool(show_window)
        self.use_jit = bool(use_jit)
        self.segment_mode = str(segment_mode or "auto").lower()
        self.schema = fs.normalize_schema_name(schema)
        self.schema_spec = fs.get_schema(self.schema)
        import gru_experts as ge

        self.route = ge.normalize_route_name(route)
        self.stream_workers = max(1, int(stream_workers))
        self.mp_workers = max(1, int(mp_workers))
        self.inference_workers = max(1, int(inference_workers))
        self.include_sequences = bool(include_sequences)
        self.mp_method = str(mp_method or "holistic").strip().lower()
        if self.mp_method not in {"holistic", "holistic_stabilized"}:
            raise ValueError("mp_method harus 'holistic' atau 'holistic_stabilized'")
        # Velocity (stateful) face-ref schemas keep per-frame history inside the
        # extractor; splitting frames across MediaPipe workers would corrupt that
        # temporal state, so force a single worker for them.
        import face_reference as _fr
        if self.schema in _fr.FACE_REF_STATEFUL_SCHEMAS and self.mp_workers > 1:
            print(
                f"[live] schema {self.schema} bersifat stateful (velocity) -> "
                f"mp_workers dipaksa 1 (dari {self.mp_workers})",
                flush=True,
            )
            self.mp_workers = 1
        if self.segment_mode not in {"auto", "rolling"}:
            raise ValueError("segment_mode harus auto atau rolling")
        self._stop_event = threading.Event()
        self.finished_event = threading.Event()
        self.last_error: str | None = None
        self.device_reason = ""
        self.runtime_diag: dict[str, Any] = {}
        self.window_name = f"BISINDO GRU Live {time.strftime('%H%M%S')}-{id(self):x}"
        self.window_closed = False
        self.quit_requested = False
        self._window_ready = False
        self._camera_lock = threading.Lock()
        self._camera: LatestFrameCamera | None = None
        self._mp_pool: AsyncMediaPipePool | None = None
        self._predictor: AsyncGRUPredictorPool | None = None

    def stop(self) -> None:
        """Signal the worker to stop; resource release belongs to the worker thread.

        Releasing the camera here (caller thread) races the worker's finally block
        and can leave the V4L2/GStreamer device unable to stream on the next open.
        """
        self._stop_event.set()
        if self._predictor is not None:
            self._predictor.stop()
        if self._mp_pool is not None:
            self._mp_pool.stop()
        with self._camera_lock:
            cap = self._camera
        if cap is None:
            return
        if hasattr(cap, "request_stop"):
            try:
                cap.request_stop()
                return
            except Exception:
                pass
        try:
            cap.running = False
        except Exception:
            pass

    def force_cleanup(self) -> None:
        """Last-resort cleanup for UI watchdogs when the worker looks stuck."""
        self.stop()
        try:
            self.join(timeout=0.5)
        except RuntimeError:
            pass
        if self.finished_event.is_set():
            self._destroy_window_best_effort()
            return
        with self._camera_lock:
            cap = self._camera
        if cap is not None:
            if hasattr(cap, "release"):
                try:
                    cap.release(join_timeout=0.5)
                except TypeError:
                    try:
                        cap.release()
                    except Exception:
                        pass
                except Exception:
                    pass
            raw_cap = getattr(cap, "cap", None)
            if raw_cap is not None:
                try:
                    raw_cap.release()
                except Exception:
                    pass
        self._destroy_window_best_effort()

    def _destroy_window_best_effort(self) -> None:
        if self.show_window and self._window_ready:
            try:
                cv2.destroyWindow(self.window_name)
            except cv2.error:
                pass
            for _ in range(3):
                try:
                    cv2.waitKey(1)
                except cv2.error:
                    break

    @property
    def uses_routed_predictor(self) -> bool:
        return self.route != "main"

    def _push(self, payload: dict[str, Any]) -> None:
        if self.status_queue is not None:
            self.status_queue.put(payload)

    def _resolve_variant(self) -> str:
        if self.requested_variant == "auto":
            return gm.select_best_available_variant(self.model_dir, schema=self.schema)
        return gm.normalize_variant_name(self.requested_variant)

    def _load_runtime(self):
        profile = LIVE_PROFILES[self.profile]
        configure_torch_for_live(int(profile.get("torch_threads", 1)))
        variant = self._resolve_variant()
        actual_device, reason = jr.select_live_device(variant, requested=self.device_name, torch_module=torch)
        self.device_reason = reason
        self.runtime_diag = jr.diagnostics(torch_module=torch, cv2_module=cv2)
        model, labels, metadata, device = gm.load_checkpoint(
            variant,
            model_dir=self.model_dir,
            device=actual_device,
            schema=self.schema,
        )
        spec = gm.variant_spec(variant)
        model = maybe_trace_model(
            model,
            spec.target_frames,
            device,
            enabled=self.use_jit,
            feature_dim=self.schema_spec.feature_dim,
        )
        dummy = torch.zeros(1, spec.target_frames, self.schema_spec.feature_dim, device=device)
        with torch.inference_mode():
            model(dummy)
        self.variant = variant
        return model, labels, metadata, device, spec

    def _open_camera_with_retry(self, profile: dict[str, float | int | str], attempts: int = 3):
        """Open the camera with retries; the device may need a beat after a previous session."""
        backoffs = (0.5, 1.0, 2.0)
        last_exc: Exception | None = None
        for attempt in range(1, attempts + 1):
            if self._stop_event.is_set():
                return None
            cap = None
            try:
                cap = LatestFrameCamera(
                    src=self.camera_index,
                    width=int(profile["width"]),
                    height=int(profile["height"]),
                    fps=int(profile.get("camera_fps", 30)),
                    use_gstreamer=bool(int(profile.get("use_gstreamer", 0))),
                    fourcc="MJPG",
                )
                cap.start(require_frame=True, timeout=float(profile.get("camera_start_timeout", 3.0)))
                return cap
            except Exception as exc:
                last_exc = exc
                if cap is not None:
                    try:
                        cap.release(join_timeout=1.0)
                    except Exception:
                        pass
                if attempt >= attempts:
                    break
                delay = backoffs[min(attempt - 1, len(backoffs) - 1)]
                self._push(
                    {
                        "event": "camera_retry",
                        "attempt": attempt,
                        "attempts": attempts,
                        "message": f"Camera open attempt {attempt}/{attempts} failed ({exc}); retry in {delay:.1f}s",
                    }
                )
                if self._stop_event.wait(delay):
                    return None
        raise RuntimeError(f"Camera src={self.camera_index} failed after {attempts} attempts: {last_exc}")

    def _build_extractor(self, profile: dict[str, float | int | str]):
        import holistic_features

        extractor_cls = holistic_features.HolisticLiveExtractor
        if self.mp_method == "holistic_stabilized":
            extractor_cls = holistic_features.StabilizedHolisticLiveExtractor
        return extractor_cls(
            self.schema,
            proc_width=int(profile["proc_width"]),
            det_conf=float(profile["det_conf"]),
            track_conf=float(profile["track_conf"]),
            model_complexity=int(profile.get("model_complexity", 1)),
            smooth_landmarks=True,
            refine_face_landmarks=bool(int(profile.get("refine_face_landmarks", 1))),
        )

    def run(self) -> None:
        cap = None
        mp_pool: AsyncMediaPipePool | None = None
        predictor: AsyncGRUPredictorPool | None = None
        camera_backend = "unknown"
        mp_backend = "studio_holistic_stabilized" if self.mp_method == "holistic_stabilized" else "studio_holistic"
        mp_backend_detail = "mp.solutions.holistic.Holistic"
        mp_thread_alive_after_stop = False
        predictor_thread_alive_after_stop = False
        camera_release_info: dict[str, Any] = {
            "camera_released": False,
            "camera_thread_alive_after_release": False,
            "camera_release_ms": 0.0,
        }
        try:
            profile = LIVE_PROFILES[self.profile]
            profile_status = live_profile_status(self.profile)
            model, labels, metadata, device, spec = self._load_runtime()
            if not self.uses_routed_predictor:
                predictor = AsyncGRUPredictorPool(
                    model,
                    labels,
                    spec.target_frames,
                    device,
                    feature_dim=self.schema_spec.feature_dim,
                    workers=self.inference_workers,
                )
            else:
                import gru_experts as ge

                routed = ge.RoutedGRUPredictor(
                    self.variant,
                    self.schema,
                    self.model_dir,
                    device=device,
                    route=self.route,
                )
                predictor = AsyncGRUPredictorPool(
                    None,
                    labels,
                    spec.target_frames,
                    device,
                    feature_dim=self.schema_spec.feature_dim,
                    predict_fn=routed.predict,
                    workers=self.inference_workers,
            )
            predictor.start()
            self._predictor = predictor

            cap = self._open_camera_with_retry(profile)
            if cap is None:
                return  # stop requested while opening; finally still reports "stopped"
            camera_backend = getattr(cap, "backend", "unknown")
            with self._camera_lock:
                self._camera = cap

            mp_pool = AsyncMediaPipePool(lambda: self._build_extractor(profile), workers=self.mp_workers).start()
            self._mp_pool = mp_pool
            mp_backend = "studio_holistic_stabilized" if self.mp_method == "holistic_stabilized" else "studio_holistic"
            mp_backend_detail = "mp.solutions.holistic.Holistic"

            rolling_buffer = LiveSequenceBuffer(
                target_frames=spec.target_frames,
                append_interval=float(profile["append_interval"]),
                reset_after=float(profile["reset_after"]),
                feature_dim=self.schema_spec.feature_dim,
            )
            segmenter = LiveGestureSegmenter(
                append_interval=float(profile["append_interval"]),
                schema=self.schema,
                min_frames=int(profile.get("min_segment_frames", 8)),
                max_frames=int(profile.get("max_segment_frames", 55)),
                end_idle_samples=int(profile.get("end_idle_samples", 5)),
                end_still_samples=int(profile.get("end_still_samples", 5)),
                motion_start=float(profile.get("motion_start", 0.010)),
                motion_end=float(profile.get("motion_end", 0.006)),
            )
            debouncer = LabelDebouncer(
                threshold=self.confidence_threshold,
                hits_required=int(profile["debounce_hits"]),
            )
            prev_vector = None
            last_submit = -1e9
            last_status = -1e9
            last_seen_request = 0
            display_prediction_id = 0
            display_label = "-"
            display_conf = 0.0
            raw_label = "-"
            raw_conf = 0.0
            last_top: list[tuple[str, float]] = []
            model_ms = 0.0
            fps_predict = 0.0
            fps_t0 = time.perf_counter()
            frame_counter = 0
            display_fps = 0.0
            motion = 0.0
            buffer_len = 0
            segment_len = 0
            segment_ms = 0.0
            sample_fps = 0.0
            segment_reason = ""
            shoulder_ok = False
            left_present = 0.0
            right_present = 0.0
            last_extract_request = 0
            extract_ms = 0.0
            visible = False
            mp_ready = False
            last_result = None
            submitted_sequences: dict[int, np.ndarray] = {}
            last_prediction_sequence: np.ndarray | None = None

            val_acc = metadata.get("best_val_acc")
            warning = ""
            if isinstance(val_acc, (int, float)) and float(val_acc) < 0.70:
                warning = f"low validation accuracy {float(val_acc):.3f}"
            self._push(
                {
                    "event": "started",
                    "schema": self.schema,
                    "feature_dim": self.schema_spec.feature_dim,
                    "variant": self.variant,
                    "requested_variant": self.requested_variant,
                    "route": self.route,
                    "profile": self.profile,
                    "performance_mode": self.profile,
                    **profile_status,
                    "window_closed": False,
                    "quit_requested": False,
                    "camera_backend": camera_backend,
                    "mp_backend": mp_backend,
                    "mp_backend_detail": mp_backend_detail,
                    "mp_ready": False,
                    "camera_opened": True,
                    "first_frame_ready": True,
                    "segment_mode": self.segment_mode,
                    "stream_workers": self.stream_workers,
                    "mp_workers": self.mp_workers,
                    "inference_workers": self.inference_workers,
                    "device": str(device),
                    "requested_device": self.device_name,
                    "device_reason": self.device_reason,
                    "runtime": (
                        "mediapipe-tflite/xnnpack + torch-cuda"
                        if device.type == "cuda"
                        else "mediapipe-tflite/xnnpack + torchscript"
                        if self.use_jit
                        else "mediapipe-tflite/xnnpack + torch"
                    ),
                    "runtime_diag": self.runtime_diag,
                    "warning": warning,
                    "message": f"Live {self.variant} aktif",
                }
            )

            while not self._stop_event.is_set():
                ok, frame = cap.read()
                if not ok or frame is None:
                    time.sleep(0.003)
                    continue
                now = time.perf_counter()
                if frame.shape[1] != int(profile["width"]) or frame.shape[0] != int(profile["height"]):
                    frame = cv2.resize(frame, (int(profile["width"]), int(profile["height"])))
                work = center_crop(frame, 1.0)
                if self._stop_event.is_set():
                    break

                if mp_pool is not None:
                    mp_pool.submit(work)
                    extracted = mp_pool.latest()
                    if extracted.error:
                        raise RuntimeError(extracted.error)
                    if extracted.request_id and extracted.request_id != last_extract_request and extracted.result is not None:
                        last_extract_request = extracted.request_id
                        last_result = extracted.result
                        extract_ms = float(extracted.extract_ms)
                        result_time = float(extracted.completed_at or now)
                        mp_ready = True

                        motion, visible = fs.motion_score(self.schema, prev_vector, last_result.vector)
                        prev_vector = last_result.vector.copy()
                        presence = fs.presence_from_vector(self.schema, last_result.vector)
                        left_present = float(presence["left_present"])
                        right_present = float(presence["right_present"])
                        shoulder_ok = bool(presence["shoulder_ok"])

                        if self.segment_mode == "rolling":
                            rolling_buffer.update(last_result.vector, visible, result_time)
                            buffer_len = len(rolling_buffer)
                            segment_len = buffer_len
                            segment_ms = 0.0
                            sample_fps = 0.0
                            if rolling_buffer.was_reset:
                                display_label, display_conf = debouncer.reset()
                                display_prediction_id = 0
                                raw_label = "-"
                                raw_conf = 0.0

                            if (
                                visible
                                and rolling_buffer.ready
                                and result_time - last_submit >= float(profile["predict_interval"])
                                and predictor is not None
                            ):
                                sequence = rolling_buffer.window()
                                request_id = predictor.submit(sequence)
                                if self.include_sequences:
                                    submitted_sequences[request_id] = sequence.copy()
                                    while len(submitted_sequences) > 8:
                                        submitted_sequences.pop(min(submitted_sequences), None)
                                last_submit = result_time
                                segment_reason = "rolling"
                        else:
                            update = segmenter.update(last_result.vector, visible, result_time)
                            motion = update.motion if update.sampled else motion
                            buffer_len = segmenter.current_len
                            segment_len = segmenter.current_len
                            segment_ms = segmenter.current_ms
                            sample_fps = segmenter.sample_fps
                            if segmenter.was_reset and update.finalized is None:
                                display_label, display_conf = debouncer.reset()
                                display_prediction_id = 0
                                raw_label = "-"
                                raw_conf = 0.0
                            if update.reason:
                                segment_reason = update.reason
                            if update.finalized is not None and predictor is not None:
                                sequence = update.finalized
                                request_id = predictor.submit(sequence)
                                if self.include_sequences:
                                    submitted_sequences[request_id] = sequence.copy()
                                    while len(submitted_sequences) > 8:
                                        submitted_sequences.pop(min(submitted_sequences), None)
                                last_submit = result_time
                                segment_len = segmenter.last_segment_len
                                segment_ms = segmenter.last_segment_ms
                                sample_fps = 0.0 if segment_ms <= 0.0 else max(0.0, (segment_len - 1) / (segment_ms / 1000.0))

                if predictor is not None:
                    prediction = predictor.latest()
                    if prediction.error:
                        raise RuntimeError(prediction.error)
                    if prediction.request_id and prediction.request_id != last_seen_request:
                        last_seen_request = prediction.request_id
                        raw_label = prediction.label
                        raw_conf = prediction.confidence
                        last_top = prediction.top
                        model_ms = prediction.model_ms
                        fps_predict = prediction.fps_predict
                        if self.include_sequences:
                            last_prediction_sequence = submitted_sequences.pop(prediction.request_id, last_prediction_sequence)
                            for old_id in list(submitted_sequences):
                                if old_id < int(prediction.request_id) - 8:
                                    submitted_sequences.pop(old_id, None)
                        display_label, display_conf = debouncer.update(raw_label, raw_conf)
                        if display_label == "-":
                            display_prediction_id = 0
                        elif raw_label == display_label and debouncer.hits >= debouncer.hits_required:
                            display_prediction_id = int(prediction.request_id)

                frame_counter += 1
                if now - fps_t0 >= 0.5:
                    display_fps = frame_counter / max(now - fps_t0, 1e-6)
                    fps_t0 = now
                    frame_counter = 0

                if self.show_window:
                    vis = resize_live_preview(work.copy(), profile_status["display_width"])
                    if last_result is not None:
                        draw_shoulders(vis, last_result.shoulders)
                        draw_simple_hand(vis, last_result.left, (0, 255, 0))
                        draw_simple_hand(vis, last_result.right, (0, 180, 255))
                    self._draw_overlay(
                        vis,
                        label=display_label,
                        confidence=display_conf,
                        raw_label=raw_label,
                        raw_confidence=raw_conf,
                        fps_camera=display_fps,
                        fps_predict=fps_predict,
                        extract_ms=extract_ms,
                        model_ms=model_ms,
                        visible=visible,
                        buffer_len=buffer_len,
                        target_frames=spec.target_frames,
                        segment_mode=self.segment_mode,
                        segment_reason=segment_reason if mp_ready else "mediapipe warming",
                    )
                    if not self._window_ready:
                        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
                        cv2.resizeWindow(self.window_name, profile_status["display_width"], profile_status["display_height"])
                        self._window_ready = True
                    cv2.imshow(self.window_name, vis)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        self.quit_requested = True
                        break
                    try:
                        if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
                            self.window_closed = True
                            break
                    except cv2.error:
                        self.window_closed = True
                        break

                if now - last_status >= float(profile["status_interval"]):
                    last_status = now
                    status_payload = {
                        "event": "status",
                        "schema": self.schema,
                        "feature_dim": self.schema_spec.feature_dim,
                        "variant": self.variant,
                        "requested_variant": self.requested_variant,
                        "route": self.route,
                        "prediction_id": int(display_prediction_id),
                        "raw_prediction_id": int(last_seen_request),
                        "prediction": display_label,
                        "confidence": float(display_conf),
                        "raw_prediction": raw_label,
                        "raw_confidence": float(raw_conf),
                        "top": last_top,
                        "visible": bool(visible),
                        "motion": float(motion),
                        "fps": float(display_fps),
                        "fps_camera": float(display_fps),
                        "fps_predict": float(fps_predict),
                        "extract_ms": float(extract_ms),
                        "model_ms": float(model_ms),
                        "buffer": int(buffer_len),
                        "target_frames": spec.target_frames,
                        "segment_mode": self.segment_mode,
                        "segment_len": int(segment_len),
                        "segment_ms": float(segment_ms),
                        "segment_reason": segment_reason,
                        "sample_fps": float(sample_fps),
                        "shoulder_ok": bool(shoulder_ok),
                        "left_present": float(left_present),
                        "right_present": float(right_present),
                        "profile": self.profile,
                        "performance_mode": self.profile,
                        **profile_status,
                        "mp_backend": mp_backend,
                        "mp_backend_detail": mp_backend_detail,
                        "mp_ready": bool(mp_ready),
                        "window_closed": self.window_closed,
                        "quit_requested": self.quit_requested,
                        "stream_workers": self.stream_workers,
                        "mp_workers": self.mp_workers,
                        "inference_workers": self.inference_workers,
                        "device": str(device),
                        "requested_device": self.device_name,
                        "device_reason": self.device_reason,
                    }
                    if self.include_sequences and last_prediction_sequence is not None:
                        status_payload["sequence_id"] = int(last_seen_request)
                        status_payload["sequence"] = last_prediction_sequence.copy()
                    self._push(status_payload)
        except Exception as exc:
            self.last_error = str(exc)
            self._push(
                {
                    "event": "error",
                    "message": self.last_error,
                    "schema": self.schema,
                    "variant": self.variant,
                    "profile": self.profile,
                    **live_profile_status(self.profile),
                    "mp_backend": mp_backend,
                    "mp_backend_detail": mp_backend_detail,
                    "window_closed": self.window_closed,
                    "quit_requested": self.quit_requested,
                    "camera_backend": camera_backend,
                    "camera_opened": cap is not None,
                    "first_frame_ready": bool(getattr(cap, "frame", None) is not None) if cap is not None else False,
                }
            )
        finally:
            if cap is not None:
                camera_release_info = cap.release(join_timeout=3.0)
                cap = None
            with self._camera_lock:
                self._camera = None
            if predictor is not None:
                predictor.stop()
                predictor_thread_alive_after_stop = predictor.join(timeout=1.0)
                self._predictor = None
            if mp_pool is not None:
                mp_pool.stop()
                mp_thread_alive_after_stop = mp_pool.join(timeout=1.0)
                self._mp_pool = None
            if self.show_window:
                try:
                    cv2.destroyWindow(self.window_name)
                except cv2.error:
                    pass
                # HighGUI (GTK) needs a few event-loop pumps to actually process the destroy.
                for _ in range(3):
                    try:
                        cv2.waitKey(1)
                    except cv2.error:
                        break
            self.finished_event.set()
            self._push(
                {
                    "event": "stopped",
                    "message": self.last_error or "Live selesai",
                    "schema": self.schema,
                    "feature_dim": self.schema_spec.feature_dim,
                    "variant": self.variant,
                    "route": self.route,
                    "profile": self.profile,
                    **live_profile_status(self.profile),
                    "mp_backend": mp_backend,
                    "mp_backend_detail": mp_backend_detail,
                    "mp_thread_alive_after_stop": bool(mp_thread_alive_after_stop),
                    "predictor_thread_alive_after_stop": bool(predictor_thread_alive_after_stop),
                    "window_closed": self.window_closed,
                    "quit_requested": self.quit_requested,
                    "camera_backend": camera_backend,
                    **camera_release_info,
                }
            )

    @staticmethod
    def _draw_overlay(
        frame: np.ndarray,
        label: str,
        confidence: float,
        raw_label: str,
        raw_confidence: float,
        fps_camera: float,
        fps_predict: float,
        extract_ms: float,
        model_ms: float,
        visible: bool,
        buffer_len: int,
        target_frames: int,
        segment_mode: str = "auto",
        segment_reason: str = "",
    ) -> None:
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 96), (20, 20, 20), -1)
        color = (42, 220, 90) if label != "-" else (80, 180, 255)
        cv2.putText(frame, f"{label.upper()}  {confidence:.2f}", (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.88, color, 2)
        cv2.putText(
            frame,
            f"raw {raw_label}:{raw_confidence:.2f} | {segment_mode} {buffer_len}/{target_frames} {segment_reason} | visible {int(visible)}",
            (16, 62),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (235, 235, 235),
            1,
        )
        cv2.putText(
            frame,
            f"cam {fps_camera:.1f} fps | pred {fps_predict:.1f} fps | extract {extract_ms:.1f} ms | model {model_ms:.1f} ms",
            (16, 86),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (235, 235, 235),
            1,
        )


def start_live_inference(
    model_type: str = "auto",
    mp_device: str = "CPU",
    status_queue: queue.Queue | None = None,
    performance_mode: str = DEFAULT_LIVE_PROFILE,
    profile: str | None = None,
    device: str = "auto",
    use_jit: bool = True,
    **kwargs,
) -> FastGRULiveWorker:
    worker = FastGRULiveWorker(
        variant=model_type,
        mp_device=mp_device,
        status_queue=status_queue,
        performance_mode=performance_mode,
        profile=profile,
        device=device,
        use_jit=use_jit,
        **kwargs,
    )
    worker.start()
    return worker


def _print_startup_probe(args: argparse.Namespace) -> tuple[str | None, str | None]:
    diag = jr.diagnostics(torch_module=torch, cv2_module=cv2)
    print(jr.diagnostics_text(diag))
    schema = fs.normalize_schema_name(args.schema)
    try:
        variant = gm.select_best_available_variant(args.model_dir, schema=schema) if args.variant in {"auto", "best"} else gm.normalize_variant_name(args.variant)
    except Exception as exc:
        print(f"variant: unavailable ({exc})")
        return None, None
    try:
        selected_device, reason = jr.select_live_device(variant, requested=args.device, torch_module=torch)
    except Exception as exc:
        print(f"device: unavailable ({exc})")
        return variant, None
    print(f"schema: {schema}")
    print(f"variant: {variant}")
    print(f"profile: {normalize_profile(args.profile)}")
    print(f"selected_device: {selected_device} ({reason})")
    return variant, selected_device


def run_probe(args: argparse.Namespace) -> int:
    variant, selected_device = _print_startup_probe(args)
    if variant is None or selected_device is None:
        return 1
    if not gm.checkpoint_exists(variant, args.model_dir, schema=args.schema):
        print(f"checkpoint missing: {args.schema}/gru_{variant}")
        return 1

    seconds = max(0.0, float(args.probe_seconds))
    if seconds <= 0.0:
        return 0

    status_queue: queue.Queue = queue.Queue()
    worker = FastGRULiveWorker(
        variant=args.variant,
        camera_index=args.camera,
        profile=args.profile,
        device=args.device,
        model_dir=args.model_dir,
        confidence_threshold=args.threshold,
        show_window=False,
        use_jit=not args.no_jit,
        segment_mode=args.segment_mode,
        schema=args.schema,
        status_queue=status_queue,
        mp_method=getattr(args, "mp_method", "holistic"),
    )
    worker.start()
    deadline = time.perf_counter() + seconds
    last_status: dict[str, Any] = {}
    try:
        while time.perf_counter() < deadline and worker.is_alive():
            try:
                event = status_queue.get(timeout=0.10)
            except queue.Empty:
                continue
            if event.get("event") == "status":
                last_status = event
            elif event.get("event") == "error":
                print(f"probe error: {event.get('message')}")
    finally:
        worker.stop()
        worker.join(timeout=3.0)

    if last_status:
        print(
            "probe: "
            f"fps_camera={last_status.get('fps_camera', 0.0):.1f} "
            f"fps_predict={last_status.get('fps_predict', 0.0):.1f} "
            f"extract_ms={last_status.get('extract_ms', 0.0):.1f} "
            f"model_ms={last_status.get('model_ms', 0.0):.1f} "
            f"buffer={last_status.get('buffer', 0)}/{last_status.get('target_frames', 0)}"
        )
    if worker.last_error:
        print(worker.last_error)
        return 1
    return 0


def capture_one_gesture(
    variant: str = "auto",
    schema: str = fs.DEFAULT_SCHEMA,
    profile: str = DEFAULT_LIVE_PROFILE,
    device: str = "auto",
    camera_index: int = 0,
    model_dir: str | None = None,
    out_dir: str | None = None,
    timeout: float = 12.0,
    show_window: bool = False,
    use_jit: bool = True,
) -> dict[str, Any]:
    """Capture one segmented live gesture, save features/GIF, and predict it."""

    from smart_extract.extract_video_smart_v8 import save_gif

    worker = FastGRULiveWorker(
        variant=variant,
        schema=schema,
        profile=profile,
        device=device,
        camera_index=camera_index,
        model_dir=model_dir,
        show_window=show_window,
        use_jit=use_jit,
        segment_mode="auto",
    )
    profile_cfg = LIVE_PROFILES[worker.profile]
    model, labels, metadata, selected_device, spec = worker._load_runtime()
    cap = None
    extractor = None
    frames: list[np.ndarray] = []
    sequence: np.ndarray | None = None
    segmenter = LiveGestureSegmenter(
        append_interval=float(profile_cfg["append_interval"]),
        schema=worker.schema,
        min_frames=int(profile_cfg.get("min_segment_frames", 8)),
        max_frames=int(profile_cfg.get("max_segment_frames", 55)),
        end_idle_samples=int(profile_cfg.get("end_idle_samples", 5)),
        end_still_samples=int(profile_cfg.get("end_still_samples", 5)),
        motion_start=float(profile_cfg.get("motion_start", 0.010)),
        motion_end=float(profile_cfg.get("motion_end", 0.006)),
    )

    try:
        cap = LatestFrameCamera(
            src=int(camera_index),
            width=int(profile_cfg["width"]),
            height=int(profile_cfg["height"]),
            fps=int(profile_cfg.get("camera_fps", 30)),
            use_gstreamer=bool(int(profile_cfg.get("use_gstreamer", 0))),
            fourcc="MJPG",
        ).start()
        extractor = worker._build_extractor(profile_cfg)

        deadline = time.perf_counter() + max(1.0, float(timeout))
        while time.perf_counter() < deadline:
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.003)
                continue
            if frame.shape[1] != int(profile_cfg["width"]) or frame.shape[0] != int(profile_cfg["height"]):
                frame = cv2.resize(frame, (int(profile_cfg["width"]), int(profile_cfg["height"])))
            work = center_crop(frame, 1.0)
            result = extractor.process(work)
            visible = bool(result.present[0] >= 0.5 or result.present[1] >= 0.5)
            update = segmenter.update(result.vector, visible, time.perf_counter())

            if update.sampled and (segmenter.active or update.finalized is not None):
                vis = work.copy()
                draw_shoulders(vis, result.shoulders)
                draw_simple_hand(vis, result.left, (0, 255, 0))
                draw_simple_hand(vis, result.right, (0, 180, 255))
                cv2.putText(
                    vis,
                    f"diagnose {segmenter.current_len} | visible {int(visible)} | motion {update.motion:.3f}",
                    (8, 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (245, 245, 245),
                    1,
                    cv2.LINE_AA,
                )
                frames.append(vis)
                if len(frames) > int(profile_cfg.get("max_segment_frames", 55)) + 8:
                    frames = frames[-int(profile_cfg.get("max_segment_frames", 55)) :]

            if show_window:
                preview = work.copy()
                draw_shoulders(preview, result.shoulders)
                draw_simple_hand(preview, result.left, (0, 255, 0))
                draw_simple_hand(preview, result.right, (0, 180, 255))
                cv2.imshow("BISINDO live diagnose", preview)
                if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                    break

            if update.finalized is not None:
                sequence = update.finalized
                break
    finally:
        if extractor is not None:
            extractor.close()
        if cap is not None:
            cap.release()
        if show_window:
            cv2.destroyAllWindows()

    if sequence is None:
        raise RuntimeError("Tidak ada segment gesture yang selesai sebelum timeout.")

    label, confidence, top = gm.predict_sequence(
        model,
        sequence,
        labels,
        spec.target_frames,
        selected_device,
        feature_dim=worker.schema_spec.feature_dim,
    )
    root = Path(out_dir) if out_dir else gm.ROOT_DIR / "assets" / "gifs" / "live_diagnose"
    root.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    stem = f"live_{worker.schema}_{worker.variant}_{worker.profile}_{stamp}"
    npz_path = root / f"{stem}.npz"
    gif_path = root / f"{stem}.gif"
    np.savez_compressed(
        npz_path,
        features=sequence.astype(np.float32),
        schema=worker.schema,
        feature_schema=worker.schema_spec.feature_schema,
        feature_dim=worker.schema_spec.feature_dim,
        target_fps=worker.schema_spec.target_fps,
        variant=worker.variant,
        profile=worker.profile,
        prediction=label,
        confidence=float(confidence),
    )
    if frames:
        save_gif(frames, gif_path, worker.schema_spec.target_fps, int(profile_cfg.get("gif_width", 420)))

    return {
        "schema": worker.schema,
        "feature_dim": worker.schema_spec.feature_dim,
        "variant": worker.variant,
        "profile": worker.profile,
        "device": str(selected_device),
        "label": label,
        "confidence": float(confidence),
        "top": top,
        "sequence_frames": int(len(sequence)),
        "segment_ms": float(segmenter.last_segment_ms),
        "npz": str(npz_path),
        "gif": str(gif_path) if frames else "",
        "metadata": metadata,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fast live GRU BISINDO inference")
    variant_choices = ["auto", *gm.VARIANT_NAMES, *(f"gru_{variant}" for variant in gm.VARIANT_NAMES)]
    parser.add_argument("--schema", default=fs.DEFAULT_SCHEMA, choices=list(fs.SCHEMA_NAMES))
    parser.add_argument("--variant", default="auto", choices=variant_choices)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--profile", default=DEFAULT_LIVE_PROFILE, choices=sorted(LIVE_PROFILES))
    parser.add_argument("--mode", dest="profile", choices=[*sorted(LIVE_PROFILES), *PROFILE_ALIASES], help=argparse.SUPPRESS)
    parser.add_argument("--segment-mode", default="auto", choices=["auto", "rolling"])
    parser.add_argument(
        "--route",
        default="main",
        choices=["main", "chunk10", "threshold", "vote_all", "main_chunk10", "main_threshold", "boosted_stack"],
    )
    parser.add_argument("--stream-workers", type=int, default=1)
    parser.add_argument("--mp-workers", type=int, default=1)
    parser.add_argument("--inference-workers", type=int, default=1)
    parser.add_argument("--mp-method", default="holistic", choices=["holistic", "holistic_stabilized"])
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--model-dir", default=str(gm.MODEL_DIR))
    parser.add_argument("--threshold", type=float, default=0.65)
    parser.add_argument("--probe", action="store_true", help="Print Jetson diagnostics and run a short no-window camera probe")
    parser.add_argument("--probe-seconds", type=float, default=5.0)
    parser.add_argument("--no-jit", action="store_true", help="Disable TorchScript trace optimization on CPU")
    parser.add_argument("--no-window", action="store_true", help="Run camera loop without OpenCV display window")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.probe:
        return run_probe(args)
    worker = FastGRULiveWorker(
        variant=args.variant,
        camera_index=args.camera,
        profile=args.profile,
        device=args.device,
        model_dir=args.model_dir,
        confidence_threshold=args.threshold,
        show_window=not args.no_window,
        use_jit=not args.no_jit,
        segment_mode=args.segment_mode,
        schema=args.schema,
        route=args.route,
        stream_workers=args.stream_workers,
        mp_workers=args.mp_workers,
        inference_workers=args.inference_workers,
        mp_method=args.mp_method,
    )
    worker.start()
    try:
        while worker.is_alive():
            worker.join(timeout=0.2)
    except KeyboardInterrupt:
        worker.stop()
        worker.join(timeout=3.0)
    if worker.last_error:
        print(worker.last_error)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
