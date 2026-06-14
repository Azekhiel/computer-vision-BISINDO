import os
import queue
import sys
import threading
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import time


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import gru_manager as gm
import gru_experts as ge
import feature_schemas as fs
import jetson_runtime as jr
import live_gru_fast as lgf
from smart_extract import contract as sc


def _feature(value):
    return sc.format_feature_value(np.full(sc.FEATURE_DIM, value, dtype=np.float32))


def _write_tiny_dataset(root: Path) -> None:
    rows = []
    for label_idx, label in enumerate(["aku", "kamu"]):
        for split in ["train", "val", "test"]:
            for vid in range(2):
                for frame in range(5):
                    rows.append(
                        {
                            "video_id": f"{label}_{split}_{vid}",
                            "label": label,
                            "frame_num": frame,
                            "split": split,
                            "feature_version": sc.FEATURE_SCHEMA,
                            "feature_dim": sc.FEATURE_DIM,
                            "features": _feature(label_idx + frame / 10.0),
                        }
                    )
    for frame in range(5):
        rows.append(
            {
                "video_id": "idle_train_0",
                "label": "idle",
                "frame_num": frame,
                "split": "train",
                "feature_version": sc.FEATURE_SCHEMA,
                "feature_dim": sc.FEATURE_DIM,
                "features": _feature(0.0),
            }
        )
        rows.append(
            {
                "video_id": "old_schema_0",
                "label": "lama",
                "frame_num": frame,
                "split": "train",
                "feature_version": "old",
                "feature_dim": 179,
                "features": "0",
            }
        )
    pd.DataFrame(rows).to_parquet(root / "tiny.parquet", index=False)


def _write_tiny_schema_dataset(root: Path, schema: str) -> None:
    schema_spec = fs.get_schema(schema)
    schema_dir = root / schema
    schema_dir.mkdir(parents=True)
    rows = []
    for label_idx, label in enumerate(["aku", "kamu"]):
        for frame in range(3):
            rows.append(
                {
                    "video_id": f"{label}_train_0",
                    "label": label,
                    "frame_num": frame,
                    "split": "train",
                    "feature_version": schema_spec.feature_schema,
                    "feature_dim": schema_spec.feature_dim,
                    "features": sc.format_feature_value(np.full(schema_spec.feature_dim, label_idx + 0.1, dtype=np.float32)),
                }
            )
    pd.DataFrame(rows).to_parquet(schema_dir / "tiny.parquet", index=False)


def test_gru_models_forward_shapes():
    for variant in gm.VARIANT_NAMES:
        for schema_name in fs.SCHEMA_NAMES:
            variant_spec = gm.variant_spec(variant)
            schema_spec = fs.get_schema(schema_name)
            model = gm.build_model(variant, input_dim=schema_spec.feature_dim, num_classes=4)
            model.eval()
            x = torch.zeros(2, variant_spec.target_frames, schema_spec.feature_dim)
            with torch.inference_mode():
                out = model(x)
            assert out.shape == (2, 4)
            assert torch.isfinite(out).all()


def test_all_variants_under_param_budget():
    """Every variant must stay under 2M params even on the largest schema input."""
    worst_dim = max(fs.get_schema(name).feature_dim for name in fs.SCHEMA_NAMES)
    for variant in gm.BASE_VARIANT_NAMES:
        model = gm.build_model(variant, input_dim=worst_dim, num_classes=56)
        n_params = sum(p.numel() for p in model.parameters())
        assert n_params < 2_000_000, f"{variant} has {n_params} params at input_dim={worst_dim}"


def test_new_variants_registered():
    for variant in ("biattn", "convfront", "tcn", "transformer"):
        assert variant in gm.BASE_VARIANT_NAMES
        assert gm.normalize_variant_name(variant) == variant
        assert gm.variant_spec(variant).target_frames == 60
    assert len(gm.VARIANT_NAMES) == 14


def test_artifact_paths_are_schema_isolated(tmp_path):
    paths = {
        schema_name: gm.artifact_paths("adi", model_dir=tmp_path, schema=schema_name)["weights"]
        for schema_name in fs.SCHEMA_NAMES
    }
    assert len(set(paths.values())) == len(fs.SCHEMA_NAMES)
    assert paths["smart180"] == tmp_path / "gru" / "smart180" / "gru_adi.pth"
    assert paths["khukuh1629"] == tmp_path / "gru" / "khukuh1629" / "gru_adi.pth"
    assert paths["adi1662"] == tmp_path / "gru" / "adi1662" / "gru_adi.pth"
    assert paths["smart180_face1584"] == tmp_path / "gru" / "smart180_face1584" / "gru_adi.pth"
    aug_paths = gm.artifact_paths("adi_dengan_augmentasi", model_dir=tmp_path, schema="smart180")
    assert aug_paths["weights"] == tmp_path / "gru" / "smart180" / "gru_adi_dengan_augmentasi.pth"


def test_train_variant_skips_existing_checkpoint_unless_overwrite(tmp_path):
    data_dir = tmp_path / "data"
    model_dir = tmp_path / "models"
    data_dir.mkdir()
    _write_tiny_dataset(data_dir)

    paths = gm.artifact_paths("adi", model_dir=model_dir, schema="smart180")
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"old-checkpoint")

    old_weights = paths["weights"].read_bytes()
    ok, msg = gm.train_variant("adi", dataset_dir=data_dir, model_dir=model_dir, schema="smart180", epochs=1, batch_size=3, device="cpu")

    assert ok
    assert "[SKIP checkpoint exists]" in msg
    assert paths["weights"].read_bytes() == old_weights

    ok, msg = gm.train_variant(
        "adi",
        dataset_dir=data_dir,
        model_dir=model_dir,
        schema="smart180",
        epochs=1,
        batch_size=3,
        patience=1,
        device="cpu",
        overwrite_existing=True,
        backup_root=tmp_path / "backups",
    )

    assert ok, msg
    assert paths["weights"].read_bytes() != old_weights
    backed_up = list((tmp_path / "backups").rglob("gru_adi.pth"))
    assert backed_up
    assert backed_up[0].read_bytes() == old_weights


def test_train_group_suite_skips_existing_expert_without_overwrite(tmp_path):
    model_dir = tmp_path / "models"
    root = ge.suite_root(model_dir, "smart180", "chunk10", "adi")
    root.mkdir(parents=True)
    (root / "suite_metadata.json").write_text("{}", encoding="utf-8")

    result = ge.train_group_suite(
        suite="chunk10",
        variant="adi",
        schema="smart180",
        dataset_dir=tmp_path / "missing_dataset",
        model_dir=model_dir,
        epochs=1,
        batch_size=2,
        lr=None,
        patience=1,
        device="cpu",
    )

    assert result["skipped"] is True
    assert "[SKIP checkpoint exists]" in str(result["message"])


def test_resample_sequence_keeps_feature_dim():
    seq = np.random.default_rng(42).normal(size=(5, sc.FEATURE_DIM)).astype(np.float32)
    out = gm.resample_sequence(seq, 30)
    assert out.shape == (30, sc.FEATURE_DIM)
    np.testing.assert_allclose(out[0], seq[0])
    np.testing.assert_allclose(out[-1], seq[-1])


def test_load_sequences_filters_idle_and_stale_schema(tmp_path):
    _write_tiny_dataset(tmp_path)
    samples = gm.load_sequences(dataset_dir=tmp_path, include_idle=False)
    labels = {sample.label for sample in samples}
    assert labels == {"aku", "kamu"}
    assert all(sample.sequence.shape[1] == sc.FEATURE_DIM for sample in samples)


def test_load_sequences_uses_requested_schema_folder(tmp_path):
    _write_tiny_schema_dataset(tmp_path, "adi1662")
    samples = gm.load_sequences(dataset_dir=tmp_path, schema="adi1662")
    assert {sample.label for sample in samples} == {"aku", "kamu"}
    assert all(sample.sequence.shape[1] == 1662 for sample in samples)


def test_load_sequences_filters_augmented_samples(tmp_path):
    rows = []
    for video_id, is_augmented, augmented_from in [
        ("train_manual_aku.mp4", False, ""),
        ("train_aku_augmented_001", False, ""),
        ("train_aku_augmentation_002", True, "train_manual_aku.mp4"),
    ]:
        for frame in range(2):
            rows.append(
                {
                    "video_id": video_id,
                    "label": "aku",
                    "frame_num": frame,
                    "split": "train",
                    "feature_version": sc.FEATURE_SCHEMA,
                    "feature_dim": sc.FEATURE_DIM,
                    "is_augmented": is_augmented,
                    "augmented_from": augmented_from,
                    "features": sc.format_feature_value(np.full(sc.FEATURE_DIM, frame, dtype=np.float32)),
                }
            )
    pd.DataFrame(rows).to_parquet(tmp_path / "aku.parquet", index=False)

    original = gm.load_sequences(dataset_dir=tmp_path, augmentation_filter="exclude")
    augmented = gm.load_sequences(dataset_dir=tmp_path, augmentation_filter="only")
    assert [sample.video_id for sample in original] == ["train_manual_aku.mp4"]
    assert {sample.video_id for sample in augmented} == {"train_aku_augmented_001", "train_aku_augmentation_002"}
    assert all(sample.is_augmented for sample in augmented)


def test_smoke_train_and_load_all_gru_variants(tmp_path):
    data_dir = tmp_path / "data"
    model_dir = tmp_path / "models"
    data_dir.mkdir()
    _write_tiny_dataset(data_dir)

    for variant in gm.VARIANT_NAMES:
        ok, msg = gm.train_variant(
            variant,
            dataset_dir=data_dir,
            model_dir=model_dir,
            epochs=1,
            batch_size=3,
            patience=1,
            device="cpu",
            l1=0.0,
            l2=0.0,
        )
        assert ok, msg
        model, labels, metadata, device = gm.load_checkpoint(variant, model_dir=model_dir, device="cpu")
        assert len(labels) == 2
        assert metadata["feature_schema"] == sc.FEATURE_SCHEMA
        pred, conf, top = gm.predict_sequence(
            model,
            np.zeros((6, sc.FEATURE_DIM), dtype=np.float32),
            labels,
            gm.variant_spec(variant).target_frames,
            device,
        )
        assert pred in labels.values()
        assert 0.0 <= conf <= 1.0
        assert top


def test_live_sequence_buffer_waits_until_full_and_resets():
    buffer = lgf.LiveSequenceBuffer(target_frames=3, append_interval=0.0, reset_after=0.5)
    vec = np.ones(sc.FEATURE_DIM, dtype=np.float32)

    assert buffer.update(vec, True, 0.0)
    assert not buffer.ready
    assert buffer.update(vec, True, 0.1)
    assert not buffer.ready
    assert buffer.update(vec, True, 0.2)
    assert buffer.ready
    assert buffer.window().shape == (3, sc.FEATURE_DIM)

    buffer.update(vec, False, 0.4)
    assert buffer.ready
    buffer.update(vec, False, 0.8)
    assert buffer.was_reset
    assert len(buffer) == 0
    assert not buffer.ready


def test_label_debouncer_requires_stable_hits():
    debouncer = lgf.LabelDebouncer(threshold=0.65, hits_required=2)
    assert debouncer.update("aku", 0.90) == ("-", 0.0)
    assert debouncer.update("aku", 0.88) == ("aku", 0.88)
    assert debouncer.update("kamu", 0.91) == ("aku", 0.88)
    assert debouncer.update("kamu", 0.92) == ("kamu", 0.92)
    assert debouncer.update("kamu", 0.10) == ("-", 0.0)


def test_live_profile_aliases_and_default_1080p():
    assert lgf.DEFAULT_LIVE_PROFILE == "lossless1080_10"
    assert lgf.normalize_profile(None) == "lossless1080_10"
    assert lgf.normalize_profile("best") == "lossless1080_10"
    assert lgf.normalize_profile("balanced") == "lossless1080_10"
    assert lgf.normalize_profile("1080") == "lossless1080_10"
    assert lgf.normalize_profile("accurate") == "accurate10"
    assert lgf.normalize_profile("tflite") == "fast10"
    assert lgf.normalize_profile("rt-lite") == "fast10"
    assert lgf.normalize_profile("jetson") == "fast10"
    worker = lgf.FastGRULiveWorker()
    assert worker.profile == "lossless1080_10"
    assert worker.device_name == "auto"
    assert worker.segment_mode == "auto"
    profile_status = lgf.live_profile_status(None)
    assert profile_status["capture_width"] == 1920
    assert profile_status["capture_height"] == 1080
    assert profile_status["camera_fps"] == 30
    assert profile_status["proc_width"] == 960
    assert profile_status["pose_proc_width"] == 960
    assert profile_status["display_width"] == 960
    assert profile_status["display_height"] == 540
    assert int(lgf.LIVE_PROFILES["lossless1080_10"]["model_complexity"]) == 1
    assert int(lgf.LIVE_PROFILES["lossless1080_10"]["refine_face_landmarks"]) == 1
    assert float(lgf.LIVE_PROFILES["lossless1080_10"]["det_conf"]) == 0.50
    assert float(lgf.LIVE_PROFILES["lossless1080_10"]["track_conf"]) == 0.50
    assert lgf.LIVE_PROFILES["accurate10"]["shoulder_backend"] == "mp-pose"
    assert int(lgf.LIVE_PROFILES["accurate10"]["proc_width"]) == sc.BEST_EXTRACT_SETTINGS["proc_width"]
    assert int(lgf.LIVE_PROFILES["accurate10"]["pose_every"]) == sc.BEST_EXTRACT_SETTINGS["pose_every"]
    assert float(lgf.LIVE_PROFILES["accurate10"]["append_interval"]) == 1.0 / sc.TARGET_FPS
    assert int(lgf.LIVE_PROFILES["fast10"]["hand_every"]) == 1


def test_live_preview_resize_scales_1080p_without_upscaling_small_frames():
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    preview = lgf.resize_live_preview(frame, 960)
    assert preview.shape[:2] == (540, 960)

    small = np.zeros((480, 640, 3), dtype=np.uint8)
    same = lgf.resize_live_preview(small, 960)
    assert same is small
    assert same.shape[:2] == (480, 640)


def test_live_worker_preview_runs_before_mediapipe_result_and_restarts(monkeypatch):
    cameras = []
    pools = []
    imshow_calls = []

    class FakeCamera:
        def __init__(self, *args, **kwargs):
            self.backend = "fake"
            self.frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
            self.running = False
            self.cap = type("RawCap", (), {"released": False, "release": lambda raw: setattr(raw, "released", True)})()
            cameras.append(self)

        def start(self, require_frame=False, timeout=0.0):
            self.running = True
            return self

        def read(self):
            return True, self.frame.copy()

        def release(self, join_timeout=0.0):
            self.running = False
            self.cap.release()
            return {
                "camera_released": True,
                "camera_thread_alive_after_release": False,
                "camera_release_ms": 1.0,
            }

    class FakePool:
        def __init__(self, *args, **kwargs):
            self.submitted = 0
            self.stopped = False
            pools.append(self)

        def start(self):
            return self

        def submit(self, frame):
            self.submitted += 1
            return self.submitted

        def latest(self):
            return lgf.ExtractedFrameResult()

        def stop(self):
            self.stopped = True

        def join(self, timeout=0.0):
            return False

    class FakePredictor:
        def __init__(self, *args, **kwargs):
            self.stopped = False

        def start(self):
            return self

        def latest(self):
            return lgf.PredictionResult()

        def stop(self):
            self.stopped = True

        def join(self, timeout=0.0):
            return False

    monkeypatch.setattr(lgf, "LatestFrameCamera", FakeCamera)
    monkeypatch.setattr(lgf, "AsyncMediaPipePool", FakePool)
    monkeypatch.setattr(lgf, "AsyncGRUPredictorPool", FakePredictor)
    monkeypatch.setattr(lgf.cv2, "namedWindow", lambda *args, **kwargs: None)
    monkeypatch.setattr(lgf.cv2, "resizeWindow", lambda *args, **kwargs: None)
    monkeypatch.setattr(lgf.cv2, "imshow", lambda name, frame: imshow_calls.append((name, frame.shape[:2])))
    monkeypatch.setattr(lgf.cv2, "waitKey", lambda delay=1: ord("q"))
    monkeypatch.setattr(lgf.cv2, "getWindowProperty", lambda *args, **kwargs: 1.0)
    monkeypatch.setattr(lgf.cv2, "destroyWindow", lambda *args, **kwargs: None)

    fake_spec = type("Spec", (), {"target_frames": 4})()

    for _ in range(2):
        status_queue = queue.Queue()
        worker = lgf.FastGRULiveWorker(schema="smart180", show_window=True, status_queue=status_queue)
        worker._load_runtime = lambda: (object(), {0: "aku"}, {}, torch.device("cpu"), fake_spec)
        worker.run()
        events = []
        while not status_queue.empty():
            events.append(status_queue.get())
        assert any(event.get("event") == "started" and event.get("mp_backend") == "studio_holistic" for event in events)
        assert any(event.get("event") == "stopped" and event.get("camera_released") for event in events)

    assert len(cameras) == 2
    assert len(pools) == 2
    assert len(imshow_calls) == 2
    assert all(shape == (540, 960) for _name, shape in imshow_calls)
    assert all(camera.cap.released for camera in cameras)
    assert all(pool.submitted >= 1 and pool.stopped for pool in pools)


def test_live_worker_stop_signals_camera_and_pools_without_release():
    class RawCap:
        def __init__(self):
            self.released = False

        def release(self):
            self.released = True

    class FakeCameraRef:
        def __init__(self):
            self.running = True
            self.cap = RawCap()

    class FakePoolRef:
        def __init__(self):
            self.stopped = False

        def stop(self):
            self.stopped = True

    worker = lgf.FastGRULiveWorker()
    camera = FakeCameraRef()
    mp_pool = FakePoolRef()
    predictor = FakePoolRef()
    worker._camera = camera
    worker._mp_pool = mp_pool
    worker._predictor = predictor

    worker.stop()

    assert worker._stop_event.is_set()
    assert camera.running is False
    # Release belongs to the worker thread's finally block; stop() only signals.
    assert camera.cap.released is False
    assert mp_pool.stopped is True
    assert predictor.stopped is True

    # The watchdog force path still hard-releases when the worker never finished.
    worker.force_cleanup()
    assert camera.cap.released is True


def test_live_smart_schema_uses_studio_holistic_extractor(monkeypatch):
    import holistic_features

    calls = {}

    class FakeHolisticExtractor:
        def __init__(self, schema, **kwargs):
            calls["schema"] = schema
            calls["kwargs"] = kwargs
            self.backend_name = "studio_holistic"

    monkeypatch.setattr(holistic_features, "HolisticLiveExtractor", FakeHolisticExtractor)

    worker = lgf.FastGRULiveWorker(schema="smart180")
    extractor = worker._build_extractor(lgf.LIVE_PROFILES["lossless1080_10"])

    assert extractor.backend_name == "studio_holistic"
    assert calls["schema"] == "smart180"
    assert calls["kwargs"]["proc_width"] == 960
    assert calls["kwargs"]["model_complexity"] == 1
    assert calls["kwargs"]["det_conf"] == 0.50
    assert calls["kwargs"]["track_conf"] == 0.50
    assert calls["kwargs"]["smooth_landmarks"] is True
    assert calls["kwargs"]["refine_face_landmarks"] is True


def test_live_main_route_uses_plain_predictor_flag():
    main_worker = lgf.FastGRULiveWorker(route="main")
    expert_worker = lgf.FastGRULiveWorker(route="main_threshold")

    assert main_worker.route == "main"
    assert main_worker.uses_routed_predictor is False
    assert expert_worker.route == "main_threshold"
    assert expert_worker.uses_routed_predictor is True


def test_latest_frame_camera_release_is_idempotent():
    class FakeCap:
        def __init__(self):
            self.released = 0

        def release(self):
            self.released += 1

    class FakeThread:
        def __init__(self):
            self.joined = 0

        def is_alive(self):
            return True

        def join(self, timeout=None):
            self.joined += 1

    cap = FakeCap()
    thread = FakeThread()
    cam = object.__new__(lgf.LatestFrameCamera)
    cam.running = True
    cam.cap = cap
    cam.thread = thread
    cam.lock = threading.Lock()
    cam.ret = True
    cam.frame = np.zeros((2, 2, 3), dtype=np.uint8)

    cam.release()
    cam.release()

    assert cap.released == 1
    assert thread.joined == 1
    assert cam.cap is None
    assert cam.thread is None
    assert cam.frame is None
    assert cam.ret is False
    assert cam.last_release_info["camera_released"] is True
    assert cam.last_release_info["camera_thread_alive_after_release"] is True


def test_latest_frame_camera_start_requires_first_frame_and_releases_no_frame():
    class NoFrameCap:
        def __init__(self):
            self.released = 0

        def read(self):
            return False, None

        def release(self):
            self.released += 1

    cap = NoFrameCap()
    cam = object.__new__(lgf.LatestFrameCamera)
    cam.src = 0
    cam.cap = cap
    cam.thread = None
    cam.lock = threading.Lock()
    cam.ret = False
    cam.frame = None
    cam.running = False
    cam.last_release_info = {"camera_released": False, "camera_thread_alive_after_release": False, "camera_release_ms": 0.0}

    with pytest.raises(RuntimeError, match="produced no frames"):
        cam.start(require_frame=True, timeout=0.03)

    assert cap.released == 1
    assert cam.cap is None
    assert cam.thread is None
    assert cam.last_release_info["camera_released"] is True


def test_latest_frame_camera_can_start_release_and_start_again_with_fake_camera():
    class FrameCap:
        def __init__(self):
            self.released = 0
            self.frame = np.zeros((2, 2, 3), dtype=np.uint8)

        def read(self):
            return True, self.frame

        def release(self):
            self.released += 1

    for _ in range(2):
        cap = FrameCap()
        cam = object.__new__(lgf.LatestFrameCamera)
        cam.src = 0
        cam.cap = cap
        cam.thread = None
        cam.lock = threading.Lock()
        cam.ret = False
        cam.frame = None
        cam.running = False
        cam.last_release_info = {"camera_released": False, "camera_thread_alive_after_release": False, "camera_release_ms": 0.0}

        cam.start(require_frame=True, timeout=0.2)
        ok, frame = cam.read()
        info = cam.release()

        assert ok
        assert frame is not None
        assert cap.released == 1
        assert info["camera_released"] is True
        assert cam.cap is None


def test_live_gesture_segmenter_samples_10fps_and_finalizes():
    segmenter = lgf.LiveGestureSegmenter(
        append_interval=0.1,
        min_frames=3,
        max_frames=8,
        end_idle_samples=2,
        end_still_samples=2,
        motion_start=0.01,
        motion_end=0.01,
    )
    vec = np.zeros(sc.FEATURE_DIM, dtype=np.float32)
    vec[sc.SLICE_META][sc.IDX_META_LEFT_PRESENT] = 1.0

    assert segmenter.update(vec, True, 0.00).sampled
    assert not segmenter.update(vec, True, 0.05).sampled
    assert segmenter.update(vec, True, 0.10).finalized is None
    final = segmenter.update(vec, True, 0.20).finalized

    assert final is not None
    assert final.shape == (3, sc.FEATURE_DIM)
    assert segmenter.last_segment_len == 3


def test_segment_prediction_resamples_short_live_sequence():
    model = gm.build_model("adi", input_dim=sc.FEATURE_DIM, num_classes=2)
    model.eval()
    labels = {0: "aku", 1: "kamu"}
    seq = np.zeros((8, sc.FEATURE_DIM), dtype=np.float32)
    seq[:, sc.SLICE_META.start] = 1.0
    pred, conf, top = gm.predict_sequence(model, seq, labels, gm.VARIANTS["adi"].target_frames, torch.device("cpu"))
    assert pred in labels.values()
    assert 0.0 <= conf <= 1.0
    assert top


class _FakeCuda:
    def __init__(self, available: bool) -> None:
        self._available = available

    def is_available(self) -> bool:
        return self._available

    def get_device_name(self, index: int = 0) -> str:
        return "Orin"

    def get_device_capability(self, index: int = 0) -> tuple[int, int]:
        return (8, 7)


class _FakeVersion:
    def __init__(self, cuda: str | None) -> None:
        self.cuda = cuda


class _FakeTorch:
    def __init__(self, cuda_version: str | None, available: bool) -> None:
        self.__version__ = "fake"
        self.version = _FakeVersion(cuda_version)
        self.cuda = _FakeCuda(available)


def test_jetson_cuda_guard_rejects_wrong_torch_cuda():
    ok, msg = jr.validate_cuda126_for_jetson(_FakeTorch("13.0", False), require_cuda=False)
    assert not ok
    assert "CUDA 12.6" in msg
    assert "env_bisindo_cuda126" in msg


def test_jetson_cuda_guard_accepts_cuda126():
    ok, msg = jr.validate_cuda126_for_jetson(_FakeTorch("12.6", True), require_cuda=True)
    assert ok, msg


def test_jetson_live_device_policy():
    valid = _FakeTorch("12.6", True)
    assert jr.select_live_device("gru_adi", requested="auto", torch_module=valid)[0] == "cpu"
    assert jr.select_live_device("khukuh", requested="auto", torch_module=valid)[0] == "cuda"
    assert jr.select_live_device("hybrid", requested="auto", torch_module=valid)[0] == "cpu"
    assert jr.select_live_device("khukuh", requested="auto", torch_module=_FakeTorch("12.6", False))[0] == "cpu"
    with pytest.raises(RuntimeError):
        jr.select_live_device("adi", requested="auto", torch_module=_FakeTorch("13.0", False))


def test_live_cli_accepts_probe_auto_jetson_profile():
    default_args = lgf.build_arg_parser().parse_args([])
    assert default_args.profile == lgf.DEFAULT_LIVE_PROFILE
    args = lgf.build_arg_parser().parse_args(["--probe", "--probe-seconds", "0", "--profile", "accurate10", "--device", "auto", "--segment-mode", "auto"])
    assert args.probe
    assert args.profile == "accurate10"
    assert args.device == "auto"
    assert args.segment_mode == "auto"


class _SlowModel(torch.nn.Module):
    def forward(self, x):
        time.sleep(0.08)
        return torch.tensor([[0.1, 1.0]], dtype=torch.float32, device=x.device)


def test_async_predictor_submit_does_not_block_camera_thread():
    predictor = lgf.AsyncGRUPredictor(
        model=_SlowModel().eval(),
        labels={0: "aku", 1: "kamu"},
        target_frames=4,
        device=torch.device("cpu"),
    )
    predictor.start()
    try:
        seq = np.zeros((4, sc.FEATURE_DIM), dtype=np.float32)
        t0 = time.perf_counter()
        request_id = predictor.submit(seq)
        submit_ms = (time.perf_counter() - t0) * 1000.0
        assert submit_ms < 25.0

        deadline = time.perf_counter() + 2.0
        latest = predictor.latest()
        while latest.request_id != request_id and time.perf_counter() < deadline:
            time.sleep(0.01)
            latest = predictor.latest()

        assert latest.request_id == request_id
        assert latest.label == "kamu"
        assert latest.model_ms >= 70.0
    finally:
        predictor.stop()
        predictor.join(timeout=1.0)


def test_eval_all_skips_missing_checkpoint(tmp_path, capsys):
    data_dir = tmp_path / "data"
    model_dir = tmp_path / "models"
    data_dir.mkdir()
    model_dir.mkdir()
    _write_tiny_dataset(data_dir)

    parser = gm.build_arg_parser()
    args = parser.parse_args(
        [
            "eval",
            "--variant",
            "all",
            "--dataset-dir",
            str(data_dir),
            "--model-dir",
            str(model_dir),
            "--device",
            "cpu",
            "--suite",
            "all",
        ]
    )
    assert args.func(args) == 0
    out = capsys.readouterr().out
    assert "schema | variant | suite | split | samples" in out
    assert "smart180/khukuh/main: skip" in out
    assert "smart180/khukuh/chunk10: skip" in out
    assert "smart180/adi/main: skip" in out
    assert "smart180/hybrid/main: skip" in out


def test_classification_metrics_macro_micro_known_values():
    metrics = gm.classification_metrics(
        ["a", "a", "a", "b"],
        ["a", "b", "a", "a"],
    )

    assert metrics["accuracy"] == pytest.approx(0.5)
    assert metrics["precision_macro"] == pytest.approx(1.0 / 3.0)
    assert metrics["recall_macro"] == pytest.approx(1.0 / 3.0)
    assert metrics["f1_macro"] == pytest.approx(1.0 / 3.0)
    assert metrics["precision_micro"] == pytest.approx(0.5)
    assert metrics["recall_micro"] == pytest.approx(0.5)
    assert metrics["f1_micro"] == pytest.approx(0.5)


def test_chunk10_grouping_splits_23_labels_as_10_10_3():
    labels = [f"label_{idx:02d}" for idx in range(23)]
    bundle = ge.PrototypeBundle(
        labels=labels,
        mean=np.zeros(4, dtype=np.float32),
        std=np.ones(4, dtype=np.float32),
        prototypes={label: np.asarray([idx, 0.0, 0.0, 0.0], dtype=np.float32) for idx, label in enumerate(labels)},
    )
    assert [len(group) for group in ge.make_chunk10_groups(bundle)] == [10, 10, 3]


def test_threshold_groups_are_non_empty_and_min_two_when_possible():
    labels = [f"label_{idx:02d}" for idx in range(7)]
    bundle = ge.PrototypeBundle(
        labels=labels,
        mean=np.zeros(4, dtype=np.float32),
        std=np.ones(4, dtype=np.float32),
        prototypes={label: np.asarray([idx * 0.05, 0.0, 0.0, 0.0], dtype=np.float32) for idx, label in enumerate(labels)},
    )
    groups, threshold = ge.make_threshold_groups(bundle, threshold=0.10, min_group_size=2, max_group_size=3)
    assert threshold == 0.10
    assert all(group for group in groups)
    assert all(len(group) >= 2 for group in groups[:-1])


class _FakeExtractor:
    def __init__(self) -> None:
        self.closed = False

    def process(self, frame):
        time.sleep(0.08)
        return type("Result", (), {"value": int(frame[0, 0, 0])})()

    def close(self) -> None:
        self.closed = True


def test_async_mediapipe_pool_drops_stale_pending_frames():
    pool = lgf.AsyncMediaPipePool(lambda: _FakeExtractor(), workers=1).start()
    try:
        first = np.zeros((2, 2, 3), dtype=np.uint8)
        second = np.ones((2, 2, 3), dtype=np.uint8)
        third = np.ones((2, 2, 3), dtype=np.uint8) * 3
        pool.submit(first)
        time.sleep(0.01)
        pool.submit(second)
        last_id = pool.submit(third)
        deadline = time.perf_counter() + 2.0
        latest = pool.latest()
        while latest.request_id != last_id and time.perf_counter() < deadline:
            time.sleep(0.02)
            latest = pool.latest()
        assert latest.request_id == last_id
        assert latest.result.value == 3
    finally:
        pool.stop()
        pool.join(timeout=1.0)
