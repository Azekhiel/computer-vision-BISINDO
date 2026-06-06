import os
import queue
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import bisindo_cli as cli
import feature_schemas as fs
import main_ui
import photo_extract as pe
from smart_extract import contract as sc


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fake")
    return path


def _feature(value: float) -> str:
    return sc.format_feature_value(np.full(sc.FEATURE_DIM, value, dtype=np.float32))


def _fake_extract(path: Path, args=None, include_frames: bool = False):
    features = np.stack(
        [
            np.full(sc.FEATURE_DIM, 0.1, dtype=np.float32),
            np.full(sc.FEATURE_DIM, 0.2, dtype=np.float32),
            np.full(sc.FEATURE_DIM, 0.3, dtype=np.float32),
        ]
    )
    features[:, sc.SLICE_META] = 1.0
    return {
        "features": features,
        "frames": [
            {
                "target_source_frame": idx,
                "chosen_source_frame": idx,
                "time_sec": idx / sc.TARGET_FPS,
                "enhance_mode": "test",
                "left_present": 1.0,
                "right_present": 1.0,
                "left_detected": 1.0,
                "right_detected": 1.0,
                "left_held": 0.0,
                "right_held": 0.0,
                "left_score": 0.8,
                "right_score": 0.9,
            }
            for idx in range(3)
        ],
        "target_fps": sc.TARGET_FPS,
        "smart_mode": "best",
        "overlay_frames": [],
        "skeleton_frames": [],
    }


def test_cli_parser_accepts_main_subcommands():
    parser = cli.build_arg_parser()
    assert parser.parse_args(["live"]).profile == cli.DEFAULT_LIVE_PROFILE
    assert parser.parse_args(["live"]).route == "main"
    assert parser.parse_args(["diagnose-live"]).profile == cli.DEFAULT_LIVE_PROFILE
    assert parser.parse_args(["eval", "--suite", "all"]).suite == "all"
    live_args = parser.parse_args(["live", "--schema", "adi1662", "--window", "--device", "auto", "--mode", "fast10", "--segment-mode", "rolling"])
    assert live_args.command == "live"
    assert live_args.schema == "adi1662"
    assert live_args.profile == "fast10"
    assert live_args.segment_mode == "rolling"
    assert parser.parse_args(["diagnose-live", "--schema", "khukuh1629", "--mode", "accurate10"]).command == "diagnose-live"
    assert parser.parse_args(["train", "--schema", "all", "--variant", "all", "--epochs", "1", "--overwrite-existing"]).overwrite_existing
    assert parser.parse_args(["import", "--schema", "khukuh1629", "--path", "record/video", "--split", "train", "--overwrite-existing"]).overwrite_existing
    assert parser.parse_args(["record", "--schema", "all", "--label", "aku", "--overwrite-existing"]).overwrite_existing
    assert parser.parse_args(["gif", "make", "--schema", "adi1662", "--video", "a.mp4", "--label", "aku"]).gif_command == "make"
    assert parser.parse_args(["photo", "extract", "--path", "record/photo", "--schema", "smart180", "--workers", "2", "--overwrite-existing"]).overwrite_existing
    assert parser.parse_args(["photo", "commit", "--session", "tmp/photo_extract/demo", "--overwrite-existing"]).overwrite_existing
    assert parser.parse_args(["augment", "--schema", "smart180", "--target-per-class", "20", "--overwrite-existing"]).overwrite_existing
    assert parser.parse_args(["augment", "--schema", "smart180", "--vocab", "aku", "--vocab", "kamu", "--split", "train,val"]).vocab == ["aku", "kamu"]
    assert parser.parse_args(["augment-delete", "--schema", "smart180", "--all-vocab"]).all_vocab
    assert parser.parse_args(["extract-full", "--source", "dataset_full_mediapipe", "--schema", "full", "--clean", "backup"]).command == "extract-full"
    assert parser.parse_args(["train", "--schema", "full", "--variant", "all", "--epochs", "1"]).schema == "full"
    assert parser.parse_args(["train", "--schema", "full", "--variant", "adi,hybrid_dengan_augmentasi", "--train-data", "both"]).variant == "adi,hybrid_dengan_augmentasi"
    assert parser.parse_args(["train-suite", "--schema", "full", "--variant", "all", "--suite", "main,chunk10", "--overwrite-existing"]).overwrite_existing
    assert parser.parse_args(["record-live", "--label", "aku", "--overwrite-existing"]).overwrite_existing
    assert parser.parse_args(["live", "--route", "main_threshold", "--profile", "lossless1080_10", "--mp-workers", "2"]).route == "main_threshold"
    assert parser.parse_args(["gui"]).command == "gui"


def test_feature_schema_registry_paths(tmp_path):
    assert fs.get_schema("smart180").feature_dim == 180
    assert fs.get_schema("smart180").uses_face is False
    assert fs.get_schema("khukuh1629").feature_dim == 1629
    assert fs.get_schema("adi1662").feature_dim == 1662
    assert fs.normalize_schema_name("smart_face") == "smart180_face1584"
    assert fs.get_schema("smart180_face1584").uses_face is True
    assert fs.dataset_dir_for("khukuh1629", tmp_path) == tmp_path / "khukuh1629"
    assert fs.model_dir_for("adi1662", tmp_path) == tmp_path / "gru" / "adi1662"
    assert fs.expand_schema_names("all") == fs.FULL_SCHEMA_NAMES
    assert fs.expand_schema_names("original") == fs.BASE_SCHEMA_NAMES
    assert main_ui.resolve_live_route_name("Main GRU (tanpa expert)") == "main"
    assert main_ui.display_live_route_name("main") == "Main GRU (tanpa expert)"


class _FakeVar:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


class _FakeButton:
    def __init__(self):
        self.options = {}

    def configure(self, **kwargs):
        self.options.update(kwargs)


class _FakeCombo(_FakeButton):
    pass


class _FakeRoot:
    def __init__(self):
        self.cancelled = []
        self.scheduled = []

    def after(self, delay, callback):
        job = f"job-{len(self.scheduled) + 1}"
        self.scheduled.append((job, delay, callback))
        return job

    def after_cancel(self, job):
        self.cancelled.append(job)


class _ImmediateRoot(_FakeRoot):
    def after(self, delay, callback):
        callback()
        return f"job-{len(self.scheduled) + 1}"


class _FakeLiveWorker:
    def __init__(self, alive=False):
        self.alive = alive
        self.stopped = False

    def is_alive(self):
        return self.alive

    def stop(self):
        self.stopped = True


def _fake_live_ui():
    ui = object.__new__(main_ui.AppUI)
    ui.root = _FakeRoot()
    ui.btn_live = _FakeButton()
    ui.live_status_var = _FakeVar("Live: starting")
    ui.mode_var = _FakeVar("lossless1080_10")
    ui.live_route_var = _FakeVar("main")
    ui.live_threshold_var = _FakeVar("default")
    ui.live_model_data_var = _FakeVar("Original")
    ui.live_queue = queue.Queue()
    ui.live_poll_job = None
    ui.live_reset_job = None
    ui.live_stop_started_at = None
    ui.live_worker = _FakeLiveWorker(alive=False)
    ui.live_plus_worker = None
    ui.live_plus_reset_job = None
    ui.tts_events = []

    def ensure_tts(context, callback):
        ui.tts_events.append(("ensure", context))
        callback()
        return True

    def release_tts(context):
        ui.tts_events.append(("release", context))

    ui.ensure_tts_loaded_for_live = ensure_tts
    ui.release_tts_after_live = release_tts
    return ui


class _FakeSpeaker:
    def __init__(self):
        self.spoken = []

    def say(self, text, sink_name=None):
        self.spoken.append((text, sink_name))


class _ImmediateThread:
    def __init__(self, target, daemon=False):
        self.target = target
        self.daemon = daemon

    def start(self):
        self.target()


class _FakeTTSResult:
    def __init__(self, timing_total=1.23):
        self.timing_sec = {"total": timing_total}
        self.final_wav_path = Path("fake.wav")


class _FakeLoadedTTS:
    is_loaded = True

    def __init__(self, timing_total=1.23):
        self.timing_total = timing_total
        self.spoken = []

    def speak(self, text, **kwargs):
        self.spoken.append((text, kwargs))
        return _FakeTTSResult(self.timing_total)


def _fake_live_plus_ui():
    ui = object.__new__(main_ui.AppUI)
    ui.root = _FakeRoot()
    ui.btn_live_plus = _FakeButton()
    ui.live_plus_status_var = _FakeVar("LiveTest Plus: starting")
    ui.live_plus_buffer_var = _FakeVar("Buffer: -")
    ui.live_plus_output_var = _FakeVar("Output: -")
    ui.live_plus_use_llm_var = _FakeVar(False)
    ui.live_plus_allow_word_fix_var = _FakeVar(False)
    ui.live_plus_ollama_model_var = _FakeVar("qwen2.5:1.5b")
    ui.live_plus_audio_sink_var = _FakeVar("auto/default")
    ui.live_plus_audio_sink_display_to_name = {"auto/default": None}
    ui.live_plus_buffer = main_ui.lp.SentenceBuffer(max_words=2, idle_no_hand_sec=5.0)
    ui.live_plus_speaker = _FakeSpeaker()
    ui.live_plus_queue = queue.Queue()
    ui.live_plus_poll_job = None
    ui.live_plus_reset_job = None
    ui.live_plus_sentence_pending = 0
    ui.live_plus_stop_started_at = None
    ui.live_plus_worker = _FakeLiveWorker(alive=False)
    ui.live_worker = None
    ui.live_reset_job = None
    ui.mode_var = _FakeVar("lossless1080_10")
    ui.live_route_var = _FakeVar("main")
    ui.live_threshold_var = _FakeVar("default")
    ui.live_model_data_var = _FakeVar("Original")
    ui.tts_use_loaded_var = _FakeVar(False)
    ui.tts_player_var = _FakeVar("auto")
    ui.tts_runtime = None
    ui.tts_events = []

    def ensure_tts(context, callback):
        ui.tts_events.append(("ensure", context))
        callback()
        return True

    def release_tts(context):
        ui.tts_events.append(("release", context))

    ui.ensure_tts_loaded_for_live = ensure_tts
    ui.release_tts_after_live = release_tts
    return ui


def test_main_ui_reset_live_ui_cancels_poll_and_restores_button():
    ui = _fake_live_ui()
    ui.live_poll_job = "poll-1"

    ui._reset_live_ui("Live: done")

    assert ui.root.cancelled == ["poll-1"]
    assert ui.live_poll_job is None
    assert ui.live_worker is None
    assert ui.btn_live.options == {"text": "Start Live Test", "state": "normal"}
    assert ui.live_status_var.get() == "Live: done"


def test_main_ui_poll_live_stopped_resets_worker_and_shows_capture():
    ui = _fake_live_ui()
    ui.live_queue.put(
        {
            "event": "stopped",
            "message": "Live selesai",
            "profile": "lossless1080_10",
            "capture_width": 1920,
            "capture_height": 1080,
            "camera_fps": 30,
            "proc_width": 960,
            "pose_proc_width": 960,
            "display_width": 960,
            "display_height": 540,
            "camera_released": True,
            "camera_thread_alive_after_release": False,
            "camera_release_ms": 12.0,
        }
    )

    ui._poll_live()

    assert ui.live_worker is None
    assert ui.live_poll_job is None
    assert ui.btn_live.options == {"text": "Releasing camera...", "state": "disabled"}
    assert "1920x1080@30" in ui.live_status_var.get()
    assert "display 960x540" in ui.live_status_var.get()
    assert "camera released" in ui.live_status_var.get()

    ui.root.scheduled[-1][2]()
    assert ui.live_reset_job is None
    assert ui.btn_live.options == {"text": "Start Live Test", "state": "normal"}


def test_main_ui_poll_live_error_resets_worker(monkeypatch):
    warnings = []
    monkeypatch.setattr(main_ui.messagebox, "showwarning", lambda title, message: warnings.append((title, message)))
    ui = _fake_live_ui()
    ui.live_queue.put({"event": "error", "message": "kamera gagal"})

    ui._poll_live()

    assert ui.live_worker is None
    assert ui.live_poll_job is None
    assert ui.btn_live.options == {"text": "Start Live Test", "state": "normal"}
    assert ui.live_status_var.get() == "Live error: kamera gagal"
    assert warnings == [("Live Test", "kamera gagal")]


def test_main_ui_poll_live_dead_worker_without_events_resets():
    ui = _fake_live_ui()

    ui._poll_live()

    assert ui.live_worker is None
    assert ui.live_poll_job is None
    assert ui.btn_live.options == {"text": "Start Live Test", "state": "normal"}


def test_main_ui_poll_live_stop_watchdog_force_resets_stuck_worker():
    ui = _fake_live_ui()
    ui.live_worker = _FakeLiveWorker(alive=True)
    ui.live_stop_started_at = main_ui.time.perf_counter() - 11.0

    ui._poll_live()

    assert ui.live_worker is None
    assert ui.live_poll_job is None
    assert ui.live_reset_job is not None
    assert ui.btn_live.options == {"text": "Releasing camera...", "state": "disabled"}
    assert "force reset" in ui.live_status_var.get()


def test_main_ui_toggle_live_after_dead_worker_starts_fresh_main_route(monkeypatch):
    class NewWorker:
        def is_alive(self):
            return True

    calls = {}

    def fake_start(*args, **kwargs):
        calls["args"] = args
        calls["kwargs"] = kwargs
        return NewWorker()

    ui = _fake_live_ui()
    ui.live_worker = _FakeLiveWorker(alive=False)
    ui.selected_live_schema = lambda: "smart180"
    ui.selected_live_variant_value = lambda: "adi"
    ui.live_stream_workers_var = _FakeVar("1")
    ui.live_mp_workers_var = _FakeVar("1")
    ui.live_inference_workers_var = _FakeVar("1")
    ui.live_device_var = _FakeVar("cpu")
    ui.live_route_var = _FakeVar("Main GRU (tanpa expert)")
    monkeypatch.setattr(main_ui, "validate_live_checkpoint", lambda schema, variant, **kwargs: variant)
    monkeypatch.setattr(main_ui.live_gru_fast, "start_live_inference", fake_start)

    ui.toggle_live()

    assert ui.tts_events == [("release", "live"), ("ensure", "live")]
    assert ui.btn_live.options["text"] == "Stop Live Test"
    assert calls["kwargs"]["route"] == "main"
    assert calls["kwargs"]["schema"] == "smart180"
    assert "confidence_threshold" not in calls["kwargs"]
    assert calls["args"] == ("adi",)


def test_main_ui_toggle_live_passes_numeric_threshold(monkeypatch):
    class NewWorker:
        def is_alive(self):
            return True

    calls = {}

    def fake_start(*args, **kwargs):
        calls["args"] = args
        calls["kwargs"] = kwargs
        return NewWorker()

    ui = _fake_live_ui()
    ui.live_worker = _FakeLiveWorker(alive=False)
    ui.selected_live_schema = lambda: "smart180"
    ui.selected_live_variant_value = lambda: "adi"
    ui.live_stream_workers_var = _FakeVar("1")
    ui.live_mp_workers_var = _FakeVar("1")
    ui.live_inference_workers_var = _FakeVar("1")
    ui.live_device_var = _FakeVar("cpu")
    ui.live_route_var = _FakeVar("Main GRU (tanpa expert)")
    ui.live_threshold_var = _FakeVar("0.65")
    monkeypatch.setattr(main_ui, "validate_live_checkpoint", lambda schema, variant, **kwargs: variant)
    monkeypatch.setattr(main_ui.live_gru_fast, "start_live_inference", fake_start)

    ui.toggle_live()

    assert ui.tts_events == [("release", "live"), ("ensure", "live")]
    assert calls["kwargs"]["confidence_threshold"] == 0.65


def test_main_ui_toggle_live_blocks_when_livetest_plus_running(monkeypatch):
    warnings = []
    monkeypatch.setattr(main_ui.messagebox, "showwarning", lambda title, message: warnings.append((title, message)))
    ui = _fake_live_ui()
    ui.live_worker = None
    ui.live_plus_worker = _FakeLiveWorker(alive=True)

    ui.toggle_live()

    assert warnings == [("Live Test", "Stop LiveTest Plus dulu sebelum menjalankan Live Test biasa.")]
    assert ui.btn_live.options == {}


def test_main_ui_tts_load_failure_keeps_live_worker_stopped(monkeypatch):
    errors = []
    monkeypatch.setattr(main_ui.messagebox, "showerror", lambda title, message: errors.append((title, message)))

    class FailingRuntime:
        def __init__(self, profile_name, device="auto"):
            self.profile_name = profile_name
            self.device = device

        def load(self, warmup=True):
            raise RuntimeError("tts gagal")

        def unload(self):
            pass

    class ImmediateThread:
        def __init__(self, target, daemon=False):
            self.target = target

        def start(self):
            self.target()

        def is_alive(self):
            return False

    ui = object.__new__(main_ui.AppUI)
    ui.root = _ImmediateRoot()
    ui.btn_live = _FakeButton()
    ui.live_status_var = _FakeVar("Live: idle")
    ui.tts_status_var = _FakeVar("TTS: idle")
    ui.tts_profile_var = _FakeVar("cewek_dewasa_default")
    ui.tts_device_var = _FakeVar("auto")
    ui.tts_runtime = None
    ui.tts_load_thread = None
    ui.unload_tts_profile = lambda silent=False: setattr(ui, "tts_runtime", None)
    monkeypatch.setattr(main_ui.tts_rt, "LoadedTTSProfile", FailingRuntime)
    monkeypatch.setattr(main_ui.threading, "Thread", ImmediateThread)

    called = []
    ok = ui.ensure_tts_loaded_for_live("live", lambda: called.append(True))

    assert ok is True
    assert called == []
    assert ui.btn_live.options == {"text": "Start Live Test", "state": "normal"}
    assert errors == [("TTS Profile", "tts gagal")]


def test_main_ui_release_tts_after_live_unloads_runtime():
    class LoadedRuntime:
        is_loaded = True

        def __init__(self):
            self.unloaded = False

        def unload(self):
            self.unloaded = True

    ui = object.__new__(main_ui.AppUI)
    runtime = LoadedRuntime()
    ui.tts_runtime = runtime
    ui.tts_status_var = _FakeVar("")

    ui.release_tts_after_live("live")

    assert runtime.unloaded is True
    assert ui.tts_runtime is None
    assert ui.tts_status_var.get() == "TTS: unloaded after Live Test"


def test_main_ui_tts_profile_picker_filters_default_first(monkeypatch):
    ui = object.__new__(main_ui.AppUI)
    ui.tts_gender_var = _FakeVar("Cewek")
    ui.tts_demografi_var = _FakeVar("Dewasa")
    ui.tts_profile_var = _FakeVar("cowok_remaja_default")
    ui.tts_profile_combo = _FakeCombo()
    ui.tts_status_var = _FakeVar("")
    ui.tts_detail_var = _FakeVar("")
    ui.tts_device_var = _FakeVar("auto")
    ui.tts_runtime = None

    monkeypatch.setattr(
        main_ui.tts_rt,
        "profiles_for_picker",
        lambda gender, demografi: ["cewek_dewasa_default", "cewek_dewasa_soft_01"],
    )
    monkeypatch.setattr(
        main_ui.tts_rt,
        "profile_detail",
        lambda name: {
            "gender": "cewek",
            "demografi": "dewasa",
            "base_speaker": "Gadis",
            "pitch_semitones": 0.0,
            "speed": 1.0,
        },
    )

    ui.refresh_tts_profiles()

    assert ui.tts_profile_combo.options["values"] == ["cewek_dewasa_default", "cewek_dewasa_soft_01"]
    assert ui.tts_profile_var.get() == "cewek_dewasa_default"
    assert ui.tts_gender_var.get() == "Cewek"
    assert ui.tts_demografi_var.get() == "Dewasa"
    assert "cewek_dewasa_default" in ui.tts_detail_var.get()


def test_main_ui_tts_profile_selection_marks_loaded_runtime_stale(monkeypatch):
    class Runtime:
        is_loaded = True
        profile_name = "cewek_dewasa_default"
        device = "auto"

    ui = object.__new__(main_ui.AppUI)
    ui.tts_gender_var = _FakeVar("Cowok")
    ui.tts_demografi_var = _FakeVar("Remaja")
    ui.tts_profile_var = _FakeVar("cowok_remaja_default")
    ui.tts_status_var = _FakeVar("")
    ui.tts_detail_var = _FakeVar("")
    ui.tts_device_var = _FakeVar("cuda")
    ui.tts_runtime = Runtime()

    monkeypatch.setattr(
        main_ui.tts_rt,
        "profile_detail",
        lambda name: {
            "gender": "cowok",
            "demografi": "remaja",
            "base_speaker": "Wibowo",
            "pitch_semitones": 1.3,
            "speed": 1.04,
        },
    )

    ui._update_tts_profile_detail()

    assert ui.tts_gender_var.get() == "Cowok"
    assert ui.tts_demografi_var.get() == "Remaja"
    assert "pilihan baru cowok_remaja_default (cuda)" in ui.tts_status_var.get()


def test_main_ui_toggle_live_plus_starts_with_shared_live_settings(monkeypatch):
    class NewWorker:
        def is_alive(self):
            return True

    calls = {}

    def fake_start(*args, **kwargs):
        calls["args"] = args
        calls["kwargs"] = kwargs
        return NewWorker()

    ui = _fake_live_plus_ui()
    ui.selected_live_schema = lambda: "smart180"
    ui.selected_live_variant_value = lambda: "adi"
    ui.selected_live_route_value = lambda: "main_threshold"
    ui.live_stream_workers_var = _FakeVar("1")
    ui.live_mp_workers_var = _FakeVar("1")
    ui.live_inference_workers_var = _FakeVar("1")
    ui.live_device_var = _FakeVar("cpu")
    monkeypatch.setattr(main_ui, "validate_live_checkpoint", lambda schema, variant, **kwargs: variant)
    monkeypatch.setattr(main_ui.live_gru_fast, "start_live_inference", fake_start)

    ui.toggle_live_plus()

    assert ui.tts_events == [("release", "plus"), ("ensure", "plus")]
    assert ui.btn_live_plus.options["text"] == "Stop LiveTest Plus"
    assert calls["args"] == ("adi",)
    assert calls["kwargs"]["schema"] == "smart180"
    assert calls["kwargs"]["route"] == "main_threshold"
    assert calls["kwargs"]["status_queue"] is ui.live_plus_queue
    assert "confidence_threshold" not in calls["kwargs"]


def test_main_ui_toggle_live_plus_passes_numeric_threshold(monkeypatch):
    class NewWorker:
        def is_alive(self):
            return True

    calls = {}

    def fake_start(*args, **kwargs):
        calls["args"] = args
        calls["kwargs"] = kwargs
        return NewWorker()

    ui = _fake_live_plus_ui()
    ui.selected_live_schema = lambda: "smart180"
    ui.selected_live_variant_value = lambda: "adi"
    ui.selected_live_route_value = lambda: "main_threshold"
    ui.live_stream_workers_var = _FakeVar("1")
    ui.live_mp_workers_var = _FakeVar("1")
    ui.live_inference_workers_var = _FakeVar("1")
    ui.live_device_var = _FakeVar("cpu")
    ui.live_threshold_var = _FakeVar("0.8")
    monkeypatch.setattr(main_ui, "validate_live_checkpoint", lambda schema, variant, **kwargs: variant)
    monkeypatch.setattr(main_ui.live_gru_fast, "start_live_inference", fake_start)

    ui.toggle_live_plus()

    assert ui.tts_events == [("release", "plus"), ("ensure", "plus")]
    assert calls["kwargs"]["confidence_threshold"] == 0.8


@pytest.mark.parametrize("raw", ["abc", "-0.1", "1.1"])
def test_main_ui_live_threshold_rejects_invalid_values(raw):
    ui = _fake_live_ui()
    ui.live_threshold_var = _FakeVar(raw)

    with pytest.raises(ValueError, match="Threshold live"):
        ui.selected_live_confidence_threshold()


@pytest.mark.parametrize("raw", ["default", "", "  "])
def test_main_ui_live_threshold_default_omits_override(raw):
    ui = _fake_live_ui()
    ui.live_threshold_var = _FakeVar(raw)

    assert ui.selected_live_confidence_threshold() is None


def test_main_ui_toggle_live_plus_blocks_when_normal_live_running(monkeypatch):
    warnings = []
    monkeypatch.setattr(main_ui.messagebox, "showwarning", lambda title, message: warnings.append((title, message)))
    ui = _fake_live_plus_ui()
    ui.live_plus_worker = None
    ui.live_worker = _FakeLiveWorker(alive=True)

    ui.toggle_live_plus()

    assert warnings == [("LiveTest Plus", "Stop Live Test biasa dulu sebelum menjalankan LiveTest Plus.")]
    assert ui.btn_live_plus.options == {}


def test_main_ui_poll_live_plus_buffers_status_and_speaks_without_llm():
    ui = _fake_live_plus_ui()
    ui.live_plus_worker = _FakeLiveWorker(alive=False)
    ui.live_plus_queue.put({"event": "status", "prediction": "saya", "confidence": 0.91, "prediction_id": 1, "visible": True})
    ui.live_plus_queue.put({"event": "status", "prediction": "makan", "confidence": 0.92, "prediction_id": 2, "visible": True})

    ui._poll_live_plus()

    assert ui.live_plus_speaker.spoken == [("saya makan", None)]
    assert ui.live_plus_buffer.pending_words() == ()
    assert ui.live_plus_output_var.get() == "Output: saya makan"


def test_main_ui_poll_live_plus_reports_tts_latency_without_llm(monkeypatch):
    monkeypatch.setattr(main_ui.threading, "Thread", _ImmediateThread)
    ui = _fake_live_plus_ui()
    ui.tts_use_loaded_var = _FakeVar(True)
    ui.tts_runtime = _FakeLoadedTTS(timing_total=1.23)
    ui.live_plus_worker = _FakeLiveWorker(alive=False)
    ui.live_plus_queue.put({"event": "status", "prediction": "saya", "confidence": 0.91, "prediction_id": 1, "visible": True})
    ui.live_plus_queue.put({"event": "status", "prediction": "makan", "confidence": 0.92, "prediction_id": 2, "visible": True})

    ui._poll_live_plus()

    assert ui.tts_runtime.spoken[0][0] == "saya makan"
    assert ui.live_plus_output_var.get() == "Output: saya makan"
    assert "LiveTest Plus: TTS ready 1.23s | done " in ui.live_plus_status_var.get()


def test_main_ui_poll_live_plus_reports_llm_and_tts_latency(monkeypatch):
    class FakeOllamaClient:
        def __init__(self, model):
            self.model = model

        def compose(self, words, allow_word_correction=False):
            assert words == ("saya", "makan")
            assert allow_word_correction is True
            return "Saya makan."

    monkeypatch.setattr(main_ui.threading, "Thread", _ImmediateThread)
    monkeypatch.setattr(main_ui.lp, "OllamaSentenceClient", FakeOllamaClient)
    ui = _fake_live_plus_ui()
    ui.live_plus_use_llm_var = _FakeVar(True)
    ui.live_plus_allow_word_fix_var = _FakeVar(True)
    ui.tts_use_loaded_var = _FakeVar(True)
    ui.tts_runtime = _FakeLoadedTTS(timing_total=1.23)
    ui.live_plus_worker = _FakeLiveWorker(alive=False)
    ui.live_plus_queue.put({"event": "status", "prediction": "saya", "confidence": 0.91, "prediction_id": 1, "visible": True})
    ui.live_plus_queue.put({"event": "status", "prediction": "makan", "confidence": 0.92, "prediction_id": 2, "visible": True})

    ui._poll_live_plus()

    assert ui.tts_runtime.spoken[0][0] == "Saya makan."
    assert ui.live_plus_output_var.get() == "Output: Saya makan."
    status = ui.live_plus_status_var.get()
    assert status.startswith("LiveTest Plus: LLM ")
    assert " | TTS ready 1.23s | done " in status
    assert ui.live_plus_sentence_pending == 0


def _arg_value(args: list[str], flag: str) -> str:
    return args[args.index(flag) + 1]


def _fake_app_ui(dataset_dir: Path):
    ui = object.__new__(main_ui.AppUI)
    ui.dataset_dir_var = _FakeVar(str(dataset_dir))
    ui.overwrite_existing_var = _FakeVar(False)
    ui.gif_schema_var = _FakeVar("smart180")
    ui.gif_split_var = _FakeVar("train")
    ui.gif_limit_var = _FakeVar("2")
    ui.gif_width_var = _FakeVar("320")
    ui.gif_draw_face_var = _FakeVar("auto")
    ui.gif_force_var = _FakeVar(False)
    ui._selected_gif_vocab = lambda: "aku"
    ui.train_suite_schema_var = _FakeVar("smart180")
    ui.train_suite_var = _FakeVar("main,chunk10")
    ui.train_suite_suite_vars = {
        "main": _FakeVar(True),
        "chunk10": _FakeVar(True),
        "threshold": _FakeVar(False),
        "boosted": _FakeVar(False),
    }
    ui.eval_schema_var = _FakeVar("all")
    ui.eval_variant_var = _FakeVar("all")
    ui.eval_suite_var = _FakeVar("all")
    ui.eval_split_var = _FakeVar("test")
    ui.device_var = _FakeVar("cpu")
    ui.epochs_var = _FakeVar("3")
    ui.batch_var = _FakeVar("4")
    ui.photo_source_var = _FakeVar("record/photo")
    ui.photo_workers_var = _FakeVar("2")
    ui.photo_profile_var = _FakeVar("fast10")
    ui.photo_schema_vars = {"smart180": _FakeVar(True), "khukuh1629": _FakeVar(False)}
    ui.augment_schema_var = _FakeVar("smart180")
    ui.augment_split_var = _FakeVar("train")
    ui.augment_split_vars = {"train": _FakeVar(True), "val": _FakeVar(False), "test": _FakeVar(False)}
    ui.augment_vocab_var = _FakeVar("aku")
    ui.augment_all_vocab_var = _FakeVar(False)
    ui.augment_vocab_vars = {}
    ui.augment_delete_dry_var = _FakeVar(True)
    ui.augment_copies_var = _FakeVar("2")
    ui.augment_min_var = _FakeVar("5")
    ui.augment_target_var = _FakeVar("10")
    ui.augment_seed_var = _FakeVar("42")
    ui.sample_schema_var = _FakeVar("smart180")
    ui.sample_split_var = _FakeVar("train")
    ui.record_label_var = _FakeVar("Aku")
    ui.record_split_var = _FakeVar("train")
    ui.record_schema_var = _FakeVar("smart180")
    ui.record_count_var = _FakeVar("1")
    ui.record_duration_var = _FakeVar("2.5")
    ui.record_camera_var = _FakeVar("0")
    ui.record_profile_var = _FakeVar("fast10")
    ui.record_target_fps_var = _FakeVar("10")
    ui.record_preview_width_var = _FakeVar("640")
    ui.record_window_var = _FakeVar(True)
    ui.record_keep_short_var = _FakeVar(False)
    ui._cli_path = lambda: str(Path("src") / "bisindo_cli.py")
    return ui


def test_main_ui_dataset_dir_flows_into_command_builders(tmp_path):
    dataset_dir = tmp_path / "selected_dataset"
    ui = _fake_app_ui(dataset_dir)
    commands = [
        ui._gif_vocab_args(),
        ui._parse_gif_args(),
        ui._train_suite_args("adi"),
        ui._photo_extract_args(),
        ui._photo_commit_args("tmp/photo_extract/session", tmp_path / "accepted.json"),
        ui._augment_args(delete=False),
        ui._augment_args(delete=True),
        ui._sample_list_args("aku"),
        ui._sample_delete_args({"label": "aku", "split": "train", "video_id": "vid_1"}, dry_run=True),
        ui._record_live_args(),
        ui._eval_args(),
    ]
    for args in commands:
        assert _arg_value(args, "--dataset-dir") == str(dataset_dir)
    assert _arg_value(commands[-2], "--full-root") == str(dataset_dir / "full_features")
    assert _arg_value(commands[-1], "--suite") == "all"
    for args in commands:
        assert "--overwrite-existing" not in args
    augment_args = commands[5]
    assert _arg_value(augment_args, "--split") == "train"
    assert _arg_value(augment_args, "--vocab") == "aku"


def test_main_ui_overwrite_flag_only_when_checked(tmp_path):
    ui = _fake_app_ui(tmp_path / "selected_dataset")

    guarded = [
        ui._train_suite_args("adi"),
        ui._photo_extract_args(),
        ui._photo_commit_args("tmp/photo_extract/session", tmp_path / "accepted.json"),
        ui._augment_args(delete=False),
        ui._record_live_args(),
    ]
    assert all("--overwrite-existing" not in args for args in guarded)
    assert "--overwrite-existing" not in ui._augment_args(delete=True)

    ui.overwrite_existing_var.set(True)
    guarded = [
        ui._train_suite_args("adi"),
        ui._photo_extract_args(),
        ui._photo_commit_args("tmp/photo_extract/session", tmp_path / "accepted.json"),
        ui._augment_args(delete=False),
        ui._record_live_args(),
    ]
    assert all("--overwrite-existing" in args for args in guarded)
    assert "--overwrite-existing" not in ui._augment_args(delete=True)


def test_main_ui_live_smart_face_missing_checkpoint_is_blocked(tmp_path):
    assert main_ui.resolve_live_schema_name("smart_face") == "smart180_face1584"
    with pytest.raises(FileNotFoundError, match="smart180_face1584/gru_adi"):
        main_ui.validate_live_checkpoint("smart_face", "adi", model_dir=tmp_path)
    with pytest.raises(FileNotFoundError, match="schema smart180_face1584"):
        main_ui.validate_live_checkpoint("smart_face", "auto", model_dir=tmp_path)


def test_main_ui_live_model_data_maps_to_augmented_variant(tmp_path):
    paths = cli.gm.artifact_paths("adi_dengan_augmentasi", model_dir=tmp_path, schema="smart180")
    paths["weights"].parent.mkdir(parents=True, exist_ok=True)
    paths["weights"].write_bytes(b"checkpoint")
    paths["labels"].write_text('{"0": "aku"}', encoding="utf-8")

    resolved = main_ui.validate_live_checkpoint(
        "smart180",
        "adi",
        model_dir=tmp_path,
        model_data="Dengan augmentasi",
    )

    assert resolved == "adi_dengan_augmentasi"


def test_import_scanner_supported_layouts(tmp_path):
    root_split = tmp_path / "split_root"
    a = _touch(root_split / "train" / "aku" / "a.mp4")
    items = cli.scan_import_items([root_split], default_split="val")
    assert items == [cli.ImportItem(video_path=a, label="aku", split="train", video_id="train_manual_a.mp4")]

    root_vocab = tmp_path / "vocab_root"
    b = _touch(root_vocab / "kamu" / "b.mp4")
    items = cli.scan_import_items([root_vocab], default_split="val")
    assert items == [cli.ImportItem(video_path=b, label="kamu", split="val", video_id="val_manual_b.mp4")]

    direct = tmp_path / "direct_vocab"
    c = _touch(direct / "c.mp4")
    items = cli.scan_import_items([direct], default_split="test")
    assert items == [cli.ImportItem(video_path=c, label="direct_vocab", split="test", video_id="test_manual_c.mp4")]

    single = _touch(tmp_path / "single.mp4")
    items = cli.scan_import_items([single], default_split="train", label="Apa Kabar")
    assert items == [cli.ImportItem(video_path=single, label="apa_kabar", split="train", video_id="train_manual_single.mp4")]


def _fake_photo_processor(item: pe.PhotoItem, schemas, duplicate_frames: int, session_dir: Path, profile: str):
    entries = []
    for schema_name in schemas:
        spec = fs.get_schema(schema_name)
        value = float(len(item.label) + len(schema_name)) / 100.0
        features = np.full((duplicate_frames, spec.feature_dim), value, dtype=np.float32)
        if spec.name in {"smart180", "smart180_face1584"}:
            features[:, sc.SLICE_META] = 1.0
        frames = [
            {
                "out_index": idx,
                "time_sec": idx / sc.TARGET_FPS,
                "target_source_frame": 0,
                "chosen_source_frame": 0,
                "enhance_mode": "photo_live",
                "left_present": 1.0,
                "right_present": 1.0,
                "left_detected": 1.0,
                "right_detected": 1.0,
                "left_held": 0.0,
                "right_held": 0.0,
                "left_score": 0.8,
                "right_score": 0.9,
                "profile": profile,
            }
            for idx in range(duplicate_frames)
        ]
        entries.append(
            pe.stage_schema_result(
                item=item,
                schema=spec.name,
                features=features,
                frames=frames,
                session_dir=session_dir,
                profile=profile,
            )
        )
    return entries


def test_photo_scanner_supported_layouts_and_stable_ids(tmp_path):
    split_root = tmp_path / "photo_split"
    a = _touch(split_root / "train" / "aku" / "a.jpg")
    _touch(split_root / "train" / "aku" / "notes.txt")
    items = pe.scan_photo_items([split_root], default_split="val")
    assert items == [pe.PhotoItem(image_path=a, label="aku", split="train", video_id=pe.stable_photo_video_id("train", "aku", a))]
    items = pe.scan_photo_items([split_root / "train"], default_split="val")
    assert items == [pe.PhotoItem(image_path=a, label="aku", split="train", video_id=pe.stable_photo_video_id("train", "aku", a))]

    vocab_root = tmp_path / "photo_vocab"
    b = _touch(vocab_root / "kamu" / "b.png")
    items = pe.scan_photo_items([vocab_root], default_split="val")
    assert items == [pe.PhotoItem(image_path=b, label="kamu", split="val", video_id=pe.stable_photo_video_id("val", "kamu", b))]

    direct = tmp_path / "langsung"
    c = _touch(direct / "c.jpeg")
    items = pe.scan_photo_items([direct], default_split="test")
    assert items == [pe.PhotoItem(image_path=c, label="langsung", split="test", video_id=pe.stable_photo_video_id("test", "langsung", c))]

    single = _touch(tmp_path / "single.webp")
    items = pe.scan_photo_items([single], default_split="train", label="Apa Kabar")
    assert items == [pe.PhotoItem(image_path=single, label="apa_kabar", split="train", video_id=pe.stable_photo_video_id("train", "apa_kabar", single))]
    assert pe.stable_photo_video_id("train", "apa_kabar", single) == pe.stable_photo_video_id("train", "Apa Kabar", single)


def test_photo_extract_commit_writes_ten_duplicate_rows_and_skips_existing(tmp_path):
    image = _touch(tmp_path / "photo" / "train" / "aku" / "a.jpg")
    manifest = pe.extract_photo_session(
        paths=[tmp_path / "photo"],
        schemas=["smart180"],
        dataset_dir=tmp_path / "dataset",
        session_root=tmp_path / "sessions",
        task_processor=_fake_photo_processor,
    )
    assert manifest["summary"]["success"] == 1
    entry = manifest["entries"][0]
    assert entry["frames"] == 10
    assert entry["source_image_path"] == str(image.resolve())

    result = pe.commit_photo_session(
        session_dir=manifest["session_dir"],
        dataset_dir=tmp_path / "dataset",
        backup_root=tmp_path / "backups",
        gif_dir=tmp_path / "gifs",
    )
    assert result.committed == 1
    assert result.rows == 10
    df = pd.read_parquet(tmp_path / "dataset" / "smart180" / "aku.parquet")
    assert len(df) == 10
    assert df["video_id"].nunique() == 1
    vectors = [sc.parse_feature_value(value) for value in df["features"]]
    assert all(np.array_equal(vectors[0], vec) for vec in vectors[1:])
    assert df["source_media_path"].eq(str(image.resolve())).all()

    duplicate = pe.commit_photo_session(
        session_dir=manifest["session_dir"],
        dataset_dir=tmp_path / "dataset",
        backup_root=tmp_path / "backups",
        gif_dir=tmp_path / "gifs",
    )
    assert duplicate.committed == 0
    assert duplicate.skipped == 1


def test_photo_commit_filter_accepts_only_selected_entries(tmp_path):
    _touch(tmp_path / "photo" / "aku" / "a.jpg")
    _touch(tmp_path / "photo" / "kamu" / "b.jpg")
    manifest = pe.extract_photo_session(
        paths=[tmp_path / "photo"],
        schemas=["smart180"],
        dataset_dir=tmp_path / "dataset",
        session_root=tmp_path / "sessions",
        task_processor=_fake_photo_processor,
    )
    aku_entry = next(entry for entry in manifest["entries"] if entry["label"] == "aku")

    result = pe.commit_photo_session(
        session_dir=manifest["session_dir"],
        accepted_entry_ids=[aku_entry["entry_id"]],
        dataset_dir=tmp_path / "dataset",
        backup_root=tmp_path / "backups",
        gif_dir=tmp_path / "gifs",
    )
    assert result.committed == 1
    assert (tmp_path / "dataset" / "smart180" / "aku.parquet").exists()
    assert not (tmp_path / "dataset" / "smart180" / "kamu.parquet").exists()


def test_photo_temp_gif_then_commit_copies_to_assets(tmp_path):
    _touch(tmp_path / "photo" / "aku" / "a.jpg")
    manifest = pe.extract_photo_session(
        paths=[tmp_path / "photo"],
        schemas=["smart180"],
        dataset_dir=tmp_path / "dataset",
        session_root=tmp_path / "sessions",
        task_processor=_fake_photo_processor,
    )
    entry_id = manifest["entries"][0]["entry_id"]
    parser = cli.build_arg_parser()
    gif_args = parser.parse_args(
        [
            "photo",
            "gif",
            "--session",
            manifest["session_dir"],
            "--entry-id",
            entry_id,
            "--width",
            "96",
            "--height",
            "96",
            "--force",
        ]
    )
    assert gif_args.func(gif_args) == 0
    updated = pe.read_manifest(manifest["session_dir"])
    updated_entry = updated["entries"][0]
    assert Path(updated_entry["gif_temp_path"]).exists()

    result = pe.commit_photo_session(
        session_dir=manifest["session_dir"],
        accepted_entry_ids=[entry_id],
        dataset_dir=tmp_path / "dataset",
        backup_root=tmp_path / "backups",
        gif_dir=tmp_path / "gifs",
    )
    assert result.committed == 1
    assert result.gif_paths and result.gif_paths[0].exists()


def test_append_import_items_skips_duplicates_and_backs_up(tmp_path, capsys):
    dataset_dir = tmp_path / "dataset"
    backup_root = tmp_path / "backups"
    first_video = _touch(tmp_path / "videos" / "a.mp4")
    second_video = _touch(tmp_path / "videos" / "b.mp4")

    first = cli.ImportItem(first_video, "aku", "train", "train_manual_a.mp4")
    result = cli.append_import_items([first], dataset_dir=dataset_dir, backup_root=backup_root, extractor=_fake_extract)
    assert result.imported == 1
    assert result.rows == 3
    df = pd.read_parquet(dataset_dir / "smart180" / "aku.parquet")
    assert len(df) == 3
    assert df["feature_version"].eq(sc.FEATURE_SCHEMA).all()
    assert df["feature_dim"].astype(int).eq(sc.FEATURE_DIM).all()

    duplicate = cli.append_import_items([first], dataset_dir=dataset_dir, backup_root=backup_root, extractor=_fake_extract)
    assert duplicate.skipped == 1
    assert len(pd.read_parquet(dataset_dir / "smart180" / "aku.parquet")) == 3

    second = cli.ImportItem(second_video, "aku", "train", "train_manual_b.mp4")
    appended = cli.append_import_items([second], dataset_dir=dataset_dir, backup_root=backup_root, extractor=_fake_extract)
    assert appended.imported == 1
    assert appended.backup_dir is not None
    assert (appended.backup_dir / "smart180" / "aku.parquet").exists()
    assert len(pd.read_parquet(dataset_dir / "smart180" / "aku.parquet")) == 6
    out = capsys.readouterr().out
    assert "[SKIP existing video_id]" in out
    assert "[BACKUP]" in out
    assert "[APPEND parquet +3 rows]" in out


def test_append_import_items_skips_existing_video_id_across_parquets(tmp_path):
    dataset_dir = tmp_path / "dataset"
    schema_dir = dataset_dir / "smart180"
    schema_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "video_id": "train_manual_sidecar.mp4",
                "label": "aku",
                "frame_num": 0,
                "split": "train",
                "feature_version": sc.FEATURE_SCHEMA,
                "feature_dim": sc.FEATURE_DIM,
                "features": _feature(0.1),
            }
        ]
    ).to_parquet(schema_dir / "sidecar.parquet", index=False)

    def should_not_extract(*args, **kwargs):
        raise AssertionError("duplicate video_id should skip before extraction")

    item = cli.ImportItem(_touch(tmp_path / "videos" / "sidecar.mp4"), "aku", "train", "train_manual_sidecar.mp4")
    result = cli.append_import_items([item], dataset_dir=dataset_dir, backup_root=tmp_path / "backups", extractor=should_not_extract)

    assert result.skipped == 1
    assert not (schema_dir / "aku.parquet").exists()


def test_append_import_items_writes_schema_specific_parquet(tmp_path):
    def fake_extract(path: Path, args=None, include_frames: bool = False):
        spec = fs.get_schema("khukuh1629")
        return {
            "features": np.zeros((2, spec.feature_dim), dtype=np.float32),
            "frames": [{"target_source_frame": 0, "chosen_source_frame": 0, "time_sec": 0.0}],
            "target_fps": spec.target_fps,
            "feature_mode": spec.feature_mode,
            "feature_version": spec.feature_schema,
            "overlay_frames": [],
            "skeleton_frames": [],
        }

    video = _touch(tmp_path / "videos" / "paper.mp4")
    item = cli.ImportItem(video, "aku", "train", "train_manual_paper.mp4")
    result = cli.append_import_items([item], dataset_dir=tmp_path / "dataset", schema="khukuh1629", extractor=fake_extract)

    assert result.imported == 1
    df = pd.read_parquet(tmp_path / "dataset" / "khukuh1629" / "aku.parquet")
    assert df["feature_version"].eq(fs.get_schema("khukuh1629").feature_schema).all()
    assert df["feature_dim"].astype(int).eq(1629).all()


def test_record_command_records_raw_then_imports_all_schemas(tmp_path, monkeypatch):
    recorded: list[Path] = []
    imports: list[str] = []

    def fake_record(out_path: Path, *args, **kwargs):
        _touch(out_path)
        recorded.append(out_path)

    def fake_append(items, dataset_dir, backup_root, schema, save_gif=False, quiet=False, overwrite_existing=False):
        imports.append(schema)
        assert overwrite_existing is False
        assert len(items) == 2
        assert all(item.label == "aku" for item in items)
        return cli.ImportResult(scanned=len(items), imported=len(items), rows=4, failed=0)

    monkeypatch.setattr(cli, "_record_one_video", fake_record)
    monkeypatch.setattr(cli, "append_import_items", fake_append)
    args = cli.build_arg_parser().parse_args(
        [
            "record",
            "--schema",
            "all",
            "--label",
            "Aku",
            "--count",
            "2",
            "--duration",
            "0.1",
            "--raw-root",
            str(tmp_path / "raw"),
            "--dataset-dir",
            str(tmp_path / "dataset"),
        ]
    )

    assert cli.cmd_record(args) == 0
    assert len(recorded) == 2
    assert imports == list(fs.FULL_SCHEMA_NAMES)


def test_augment_dataset_generates_smart_v8_and_skips_idle(tmp_path):
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    for label, base in [("aku", 0.1), ("kamu", 0.2), ("idle", 0.0)]:
        rows = []
        for frame in range(5):
            rows.append(
                {
                    "video_id": f"train_manual_{label}.mp4",
                    "label": label,
                    "frame_num": frame,
                    "split": "train",
                    "feature_version": sc.FEATURE_SCHEMA,
                    "feature_dim": sc.FEATURE_DIM,
                    "features": _feature(base + frame / 100.0),
                }
            )
        pd.DataFrame(rows).to_parquet(dataset_dir / f"{label}.parquet", index=False)

    result = cli.augment_dataset(
        dataset_dir=dataset_dir,
        backup_root=tmp_path / "backups",
        target_per_class=2,
        copies_per_sample=0,
        min_source_samples=1,
        seed=7,
    )
    assert result.generated == 2
    assert result.backup_dir is not None

    aku = pd.read_parquet(dataset_dir / "aku.parquet")
    assert aku["video_id"].nunique() == 2
    generated_rows = aku[aku["video_id"].astype(str).str.contains("_augmentation_")]
    assert not generated_rows.empty
    vectors = [sc.parse_feature_value(value) for value in generated_rows["features"]]
    assert all(vec.shape == (sc.FEATURE_DIM,) and np.isfinite(vec).all() for vec in vectors)

    idle = pd.read_parquet(dataset_dir / "idle.parquet")
    assert idle["video_id"].nunique() == 1


def test_gif_list_command_uses_given_gif_dir(tmp_path, capsys):
    _touch(tmp_path / "samples" / "smart180" / "aku" / "vid_overlay.gif")
    _touch(tmp_path / "samples" / "smart180" / "aku" / "vid_skeleton.gif")
    parser = cli.build_arg_parser()
    args = parser.parse_args(["gif", "list", "--vocab", "aku", "--gif-dir", str(tmp_path)])
    assert args.func(args) == 0
    out = capsys.readouterr().out
    assert "total: 2" in out


def test_dataset_quick_summary_does_not_parse_feature_payload(tmp_path):
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    pd.DataFrame(
        [
            {
                "video_id": "vid1",
                "label": "aku",
                "split": "train",
                "feature_version": sc.FEATURE_SCHEMA,
                "feature_dim": sc.FEATURE_DIM,
                "features": "not parsed here",
            },
            {
                "video_id": "vid2",
                "label": "idle",
                "split": "train",
                "feature_version": sc.FEATURE_SCHEMA,
                "feature_dim": sc.FEATURE_DIM,
                "features": "not parsed here",
            },
        ]
    ).to_parquet(dataset_dir / "mix.parquet", index=False)

    summary = cli.dataset_quick_summary(dataset_dir)

    assert summary["total_samples"] == 2
    assert summary["num_classifier_classes"] == 1


def test_live_terminal_prints_mock_status(capsys):
    class FakeWorker:
        last_error = None

        def __init__(self):
            self.alive = True

        def is_alive(self):
            return self.alive

        def stop(self):
            self.alive = False

        def join(self, timeout=None):
            self.alive = False

    def fake_factory(*args, **kwargs):
        assert kwargs["schema"] == "smart180"
        q = kwargs["status_queue"]
        q.put({"event": "started", "variant": "adi", "profile": "jetson10", "device": "cpu", "device_reason": "test"})
        q.put(
            {
                "event": "status",
                "prediction": "aku",
                "confidence": 0.91,
                "raw_prediction": "aku",
                "raw_confidence": 0.92,
                "fps_camera": 12.0,
                "fps_predict": 2.0,
                "extract_ms": 40.0,
                "model_ms": 20.0,
                "buffer": 60,
                "target_frames": 60,
                "segment_len": 21,
                "segment_ms": 2000.0,
                "sample_fps": 10.0,
                "shoulder_ok": True,
                "left_present": 1.0,
                "right_present": 1.0,
                "top": [("aku", 0.91), ("kamu", 0.05)],
                "visible": True,
            }
        )
        q.put({"event": "stopped", "message": "done"})
        return FakeWorker()

    args = cli.build_arg_parser().parse_args(["live", "--duration", "1"])
    assert cli.cmd_live(args, worker_factory=fake_factory) == 0
    out = capsys.readouterr().out
    assert "started: schema=smart180 variant=adi" in out
    assert "pred=aku" in out
    assert "sample_fps=10.0" in out
