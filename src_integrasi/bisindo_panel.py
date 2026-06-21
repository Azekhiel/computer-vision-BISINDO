"""Adapter between the terminal BISINDO assistant and a Tkinter GUI."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import queue
import sys
import threading
import time
from typing import Any, Callable


ROOT_DIR = Path(__file__).resolve().parents[1]
for _base in (ROOT_DIR, ROOT_DIR / "src", ROOT_DIR / "LLM"):
    path = str(_base)
    if path not in sys.path:
        sys.path.insert(0, path)

import bisindo_live_assistant as live_assistant

from src_integrasi import camera_position as camera_pos


GENDER_OPTIONS = ("Cewek", "Cowok")
AGE_OPTIONS = ("Dewasa", "Remaja", "Anak-Anak")
_GENDER_TO_VALUE = {"Cewek": "cewek", "Cowok": "cowok"}
_AGE_TO_VALUE = {"Dewasa": "dewasa", "Remaja": "remaja", "Anak-Anak": "anak_anak"}


def build_tts_profile_name(gender: str, age: str, variation: str = "default") -> str:
    gender_value = _GENDER_TO_VALUE.get(str(gender or "").strip(), str(gender or "").strip().lower())
    age_value = _AGE_TO_VALUE.get(str(age or "").strip(), str(age or "").strip().lower())
    variation_value = str(variation or "default").strip() or "default"
    return f"{gender_value}_{age_value}_{variation_value}"


def shortcut_action(keysym: str, char: str = "") -> str | None:
    key = str(keysym or "").lower()
    text = str(char or "").lower()
    if key in {"escape"}:
        return "quit"
    if key in {"space"} or char == " ":
        return "space"
    if key == "q" or text == "q":
        return "quit"
    if key == "p" or text == "p":
        return "p"
    if key == "v" or text == "v":
        return "v"
    return None


class QueuePrintingAssistant(live_assistant.BISINDOLiveAssistant):
    def __init__(self, *args: Any, gui_queue: queue.Queue | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.gui_queue = gui_queue or queue.Queue()

    def print_line(self, message: str) -> None:
        event_type = "bisindo_output" if str(message).startswith("Output akhir:") else "bisindo_log"
        self.gui_queue.put(
            {
                "type": event_type,
                "message": str(message),
                "timestamp": time.time(),
            }
        )


class GuiBISINDOController:
    def __init__(
        self,
        config: live_assistant.AssistantConfig | None = None,
        *,
        event_queue: queue.Queue | None = None,
        assistant_factory: Callable[..., QueuePrintingAssistant] = QueuePrintingAssistant,
        worker_factory: Callable[..., Any] | None = None,
        preview_controller: live_assistant.PreviewController | None = None,
    ) -> None:
        self.base_config = config or live_assistant.AssistantConfig()
        self.event_queue = event_queue or queue.Queue()
        self.assistant_factory = assistant_factory
        self.worker_factory = worker_factory
        self.preview_controller = preview_controller
        self.assistant: QueuePrintingAssistant | None = None
        self._profile_name = self.base_config.tts_profile_name
        self._starting = False
        self._prepared = False
        self._start_thread: threading.Thread | None = None
        self._reload_thread: threading.Thread | None = None
        self._lock = threading.Lock()
        # Konfigurasi posisi kamera acuan (bahu) untuk CAMPOS; disimpan ke JSON
        # yang sama dipakai runtime MQTT (src_integrasi/camera_position.json).
        self.cam_cfg = camera_pos.CameraPositionConfig.load()

    @property
    def starting(self) -> bool:
        return self._starting

    @property
    def prepared(self) -> bool:
        return self._prepared

    def _make_config(self) -> live_assistant.AssistantConfig:
        return replace(self.base_config, tts_profile_name=self._profile_name)

    def _ensure_assistant(self) -> QueuePrintingAssistant:
        if self.assistant is None:
            kwargs: dict[str, Any] = {"gui_queue": self.event_queue}
            if self.worker_factory is not None:
                kwargs["worker_factory"] = self.worker_factory
            if self.preview_controller is not None:
                kwargs["preview_controller"] = self.preview_controller
            self.assistant = self.assistant_factory(self._make_config(), **kwargs)
        return self.assistant

    def start(self) -> None:
        with self._lock:
            if self._starting:
                return
            assistant = self._ensure_assistant()
            if self._prepared:
                if assistant.paused:
                    assistant.handle_key("p")
                elif assistant.worker is None:
                    assistant.start_worker()
                return
            self._starting = True
        self._start_thread = threading.Thread(target=self._prepare_and_start, daemon=True)
        self._start_thread.start()

    def _prepare_and_start(self) -> None:
        try:
            assistant = self._ensure_assistant()
            assistant.prepare_runtime()
            assistant.start_worker()
            self._prepared = True
            self.event_queue.put({"type": "bisindo_state", "message": "BISINDO live siap."})
        except Exception as exc:
            self.event_queue.put({"type": "bisindo_error", "message": str(exc)})
        finally:
            self._starting = False

    def pause_resume(self) -> None:
        if not self._prepared:
            self.start()
            return
        self._ensure_assistant().handle_key("p")

    def toggle_preview(self) -> None:
        if not self._prepared:
            self.start()
            return
        self._ensure_assistant().handle_key("v")

    def append_space(self) -> None:
        self._ensure_assistant().handle_key("space")

    def request_stop(self) -> None:
        if self.assistant is not None:
            self.assistant.request_stop()

    def set_tts_profile(self, profile_name: str, *, reload_now: bool = True) -> None:
        self._profile_name = str(profile_name)
        self.base_config = replace(self.base_config, tts_profile_name=self._profile_name)
        if self.assistant is None:
            self.event_queue.put({"type": "bisindo_log", "message": f"Profile TTS disiapkan: {self._profile_name}"})
            return
        self.assistant.config = replace(self.assistant.config, tts_profile_name=self._profile_name)
        if not reload_now:
            self.event_queue.put({"type": "bisindo_log", "message": f"Profile TTS disiapkan: {self._profile_name}"})
            return
        if self._reload_thread is not None and self._reload_thread.is_alive():
            self.event_queue.put({"type": "bisindo_log", "message": "Reload TTS masih berjalan."})
            return
        self._reload_thread = threading.Thread(target=self._reload_tts_profile, daemon=True)
        self._reload_thread.start()

    def _reload_tts_profile(self) -> None:
        assistant = self._ensure_assistant()
        try:
            if assistant.tts_runtime is not None:
                assistant.tts_runtime.unload()
                assistant.tts_runtime = None
            assistant._load_tts()
        except Exception as exc:
            self.event_queue.put({"type": "bisindo_error", "message": f"Gagal reload TTS: {exc}"})

    # --- posisi kamera / CAMPOS (acuan dari bahu MediaPipe) ---
    def camera_snapshot(self) -> dict[str, Any]:
        """Status posisi kamera saat ini untuk ditampilkan di tab Posisi."""
        mid_x = mid_y = width = None
        if self.assistant is not None:
            mid_x, mid_y, width = self.assistant.last_shoulder
        status = self.cam_cfg.evaluate(mid_x, mid_y)
        return {
            "has_signal": status is not None,
            "mid_x": mid_x,
            "mid_y": mid_y,
            "width": width,
            "campos": status,
            "ref_mid_x": self.cam_cfg.ref_mid_x,
            "ref_mid_y": self.cam_cfg.ref_mid_y,
            "tol_x": self.cam_cfg.tol_x,
            "tol_y": self.cam_cfg.tol_y,
            "flip_horizontal": self.cam_cfg.flip_horizontal,
            "calibrated": self.cam_cfg.calibrated,
        }

    def calibrate_camera(self) -> tuple[bool, str]:
        """Simpan posisi bahu saat ini sebagai acuan posisi tubuh yang benar."""
        if self.assistant is None:
            return False, "Live belum jalan. Tekan Start dulu di tab BISINDO."
        mid_x, mid_y, width = self.assistant.last_shoulder
        if mid_x is None or mid_y is None:
            return False, "Bahu tidak terdeteksi. Pastikan badan terlihat kamera."
        ref_width = width if width is not None else self.cam_cfg.ref_width
        self.cam_cfg.calibrate(mid_x, mid_y, ref_width)
        self.cam_cfg.save()
        return True, f"Acuan tersimpan: bahu=({mid_x:.3f}, {mid_y:.3f}), lebar={ref_width:.3f}."

    def set_camera_tolerance(self, tol_x: float, tol_y: float) -> None:
        self.cam_cfg.tol_x = max(0.0, float(tol_x))
        self.cam_cfg.tol_y = max(0.0, float(tol_y))
        self.cam_cfg.save()

    def set_camera_flip(self, flip: bool) -> None:
        self.cam_cfg.flip_horizontal = bool(flip)
        self.cam_cfg.save()

    def set_specialist_enabled(self, enabled: bool, *, restart_live: bool = True) -> None:
        enabled = bool(enabled)
        self.base_config = replace(self.base_config, specialist_enabled=enabled, specialist_name="all")
        if self.assistant is None:
            state = "aktif" if enabled else "mati"
            self.event_queue.put({"type": "bisindo_log", "message": f"Auto specialist disiapkan: {state}"})
            return
        self.assistant.config = replace(self.assistant.config, specialist_enabled=enabled, specialist_name="all")
        state = "aktif" if enabled else "mati"
        if not restart_live or not self._prepared or self.assistant.paused or self.assistant.worker is None:
            self.event_queue.put({"type": "bisindo_log", "message": f"Auto specialist: {state}"})
            return
        self.assistant.stop_worker()
        live_assistant.live_session.join_worker(self.assistant.worker, timeout=1.0)
        self.assistant.worker = None
        self.assistant.start_worker(reset_watchdog=True)
        self.event_queue.put({"type": "bisindo_log", "message": f"Auto specialist: {state} (live restart)"})

    def poll(self) -> list[dict[str, Any]]:
        assistant = self.assistant
        if assistant is not None and self._prepared:
            assistant.process_events()
            preview_key = assistant.preview_controller.poll()
            if preview_key:
                assistant.handle_key(preview_key)
            if assistant.worker is not None and not assistant.worker.is_alive():
                live_assistant.live_session.join_worker(assistant.worker, timeout=0.1)
                if not assistant.stop_requested and not assistant.paused:
                    assistant.paused = True
                    assistant.print_line("Live worker berhenti. Tekan Start/Pause untuk start lagi.")
                assistant.worker = None
            assistant.check_worker_watchdog()
        return self._drain_events()

    def snapshot(self) -> dict[str, Any]:
        assistant = self.assistant
        if assistant is None:
            return {
                "starting": self._starting,
                "prepared": False,
                "paused": True,
                "preview": False,
                "buffer": "-",
                "profile": self._profile_name,
            }
        return {
            "starting": self._starting,
            "prepared": self._prepared,
            "paused": bool(assistant.paused),
            "preview": bool(assistant.preview_enabled),
            "buffer": assistant.buffer.pending_text() or "-",
            "profile": self._profile_name,
        }

    def shutdown(self) -> None:
        if self.assistant is not None:
            self.assistant.shutdown()

    def _drain_events(self) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        while True:
            try:
                events.append(self.event_queue.get_nowait())
            except queue.Empty:
                return events
