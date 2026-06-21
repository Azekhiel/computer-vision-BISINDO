"""Terminal live assistant for the locked Smart180 ADI BISINDO setup."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import queue
import select
import shutil
import subprocess
import sys
import termios
import threading
import time
import tty
from typing import Any, Callable


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
LLM_DIR = ROOT_DIR / "LLM"
for path in (SRC_DIR, LLM_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

BUFFER_SEND_DELAY_SEC = 3.0
BUFFER_MAX_WORDS = 8
LLM_MODEL = "bisindo-sailor2"
TTS_PROFILE_NAME = "cewek_dewasa_default"
LIVE_SCHEMA = "smart180"
LIVE_VARIANT = "adi_dengan_augmentasi"
LIVE_ROUTE = "main"
LIVE_MP_METHOD = "holistic"
PREVIEW_STARTS_VISIBLE = False

LIVE_PROFILE = "lossless1080_10"
LIVE_CONFIDENCE_THRESHOLD = 0.65
LIVE_STREAM_WORKERS = 1
LIVE_MP_WORKERS = 1
LIVE_INFERENCE_WORKERS = 1
LLM_TIMEOUT_SEC = 180.0
LLM_WARMUP_TIMEOUT_SEC = 45.0
LOOP_SLEEP_SEC = 0.03
PRINT_STATUS_INTERVAL_SEC = 0.60
WORKER_STARTUP_TIMEOUT_SEC = 30.0
WORKER_FIRST_STATUS_TIMEOUT_SEC = 20.0
WORKER_START_STATUS_TIMEOUT_SEC = WORKER_FIRST_STATUS_TIMEOUT_SEC
LIVE_STALE_STATUS_SEC = 4.0
PREVIEW_WINDOW_NAME = "BISINDO Live Assistant Preview"

import gru_manager as gm
import live_gru_fast
import live_session
import livetest_plus as lp
import tts_profile_runtime as tts_rt


@dataclass(frozen=True)
class AssistantConfig:
    buffer_send_delay_sec: float = BUFFER_SEND_DELAY_SEC
    buffer_max_words: int = BUFFER_MAX_WORDS
    llm_model: str = LLM_MODEL
    tts_profile_name: str = TTS_PROFILE_NAME
    live_schema: str = LIVE_SCHEMA
    live_variant: str = LIVE_VARIANT
    live_route: str = LIVE_ROUTE
    live_mp_method: str = LIVE_MP_METHOD
    preview_starts_visible: bool = PREVIEW_STARTS_VISIBLE
    live_profile: str = LIVE_PROFILE
    confidence_threshold: float = LIVE_CONFIDENCE_THRESHOLD
    stream_workers: int = LIVE_STREAM_WORKERS
    mp_workers: int = LIVE_MP_WORKERS
    inference_workers: int = LIVE_INFERENCE_WORKERS
    specialist_enabled: bool = True
    specialist_name: str = "all"
    llm_timeout_sec: float = LLM_TIMEOUT_SEC
    llm_warmup_timeout_sec: float = LLM_WARMUP_TIMEOUT_SEC
    tts_device: str = "auto"
    tts_player: str = "auto"
    camera_index: int = 0
    device: str = "auto"


class TerminalKeyReader:
    def __init__(self) -> None:
        self.enabled = sys.stdin.isatty()
        self._old_attrs: list[Any] | None = None

    def __enter__(self) -> "TerminalKeyReader":
        if self.enabled:
            self._old_attrs = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
        return self

    def __exit__(self, *_exc: object) -> None:
        if self.enabled and self._old_attrs is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old_attrs)

    def read_key(self) -> str | None:
        if not self.enabled:
            return None
        ready, _, _ = select.select([sys.stdin], [], [], 0)
        if not ready:
            return None
        char = sys.stdin.read(1)
        if char == "\x1b":
            return "quit"
        if char == " ":
            return "space"
        lowered = char.lower()
        if lowered == "q":
            return "quit"
        if lowered in {"p", "v", "c"}:
            return lowered
        return None


class PreviewController:
    def __init__(
        self,
        preview_queue: queue.Queue,
        *,
        window_name: str = PREVIEW_WINDOW_NAME,
        cv2_module: Any | None = None,
    ) -> None:
        self.preview_queue = preview_queue
        self.window_name = window_name
        self.cv2 = cv2_module or live_gru_fast.cv2
        self.enabled = False
        self.window_ready = False
        self.latest_frame: Any | None = None

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = bool(enabled)
        if not self.enabled:
            self.close()

    def close(self) -> None:
        if self.window_ready:
            try:
                self.cv2.destroyWindow(self.window_name)
            except self.cv2.error:
                pass
            for _ in range(3):
                try:
                    self.cv2.waitKey(1)
                except self.cv2.error:
                    break
        self.window_ready = False
        self.latest_frame = None
        self._drain_latest_frame()

    def poll(self) -> str | None:
        self._drain_latest_frame()
        if not self.enabled:
            return None
        if self.latest_frame is None:
            return None
        try:
            if not self.window_ready:
                self.cv2.namedWindow(self.window_name, self.cv2.WINDOW_NORMAL)
                height, width = self.latest_frame.shape[:2]
                self.cv2.resizeWindow(self.window_name, int(width), int(height))
                self.window_ready = True
            self.cv2.imshow(self.window_name, self.latest_frame)
            key_name = live_gru_fast.live_preview_key_name(self.cv2.waitKey(1) & 0xFF)
            if self.cv2.getWindowProperty(self.window_name, self.cv2.WND_PROP_VISIBLE) < 1:
                self.enabled = False
                self.close()
                return "v"
            return key_name
        except self.cv2.error:
            self.enabled = False
            self.close()
            return None

    def _drain_latest_frame(self) -> None:
        while True:
            try:
                item = self.preview_queue.get_nowait()
            except queue.Empty:
                return
            frame = item.get("frame") if isinstance(item, dict) else item
            if frame is not None:
                self.latest_frame = frame


class BISINDOLiveAssistant:
    def __init__(
        self,
        config: AssistantConfig | None = None,
        *,
        worker_factory: Callable[..., Any] | None = None,
        preview_controller: PreviewController | None = None,
    ) -> None:
        self.config = config or AssistantConfig()
        self.worker_factory = worker_factory or live_gru_fast.start_live_inference
        self.status_queue: queue.Queue = queue.Queue()
        self.preview_queue: queue.Queue = queue.Queue(maxsize=1)
        self.preview_controller = preview_controller or PreviewController(self.preview_queue)
        self.buffer = lp.SentenceBuffer(
            max_words=self.config.buffer_max_words,
            idle_no_hand_sec=self.config.buffer_send_delay_sec,
        )
        self.preview_enabled = bool(self.config.preview_starts_visible)
        self.preview_controller.set_enabled(self.preview_enabled)
        # Profil TTS aktif (bisa diganti saat runtime, mis. dari MODE MQTT).
        self.active_tts_profile = self.config.tts_profile_name
        # Posisi bahu terakhir (mid_x, mid_y, width) ternormalisasi 0..1, untuk
        # panduan posisi kamera (CAMPOS) / kalibrasi. None kalau bahu tak terdeteksi.
        self.last_shoulder: tuple[float | None, float | None, float | None] = (None, None, None)
        self.paused = False
        self.stop_requested = False
        self.worker: Any | None = None
        self.tts_runtime: tts_rt.LoadedTTSProfile | None = None
        self.espeak = lp.EspeakSpeaker()
        self.sentence_threads: list[threading.Thread] = []
        self._last_status_print = 0.0
        self._worker_started_at: float | None = None
        self._last_worker_activity_at: float | None = None
        self._worker_started_event_at: float | None = None
        self._last_live_status_at: float | None = None
        self._has_received_live_status = False
        self._status_count_since_start = 0
        self._watchdog_restarted_once = False

    def print_line(self, message: str) -> None:
        print(message, flush=True)

    def validate_checkpoint(self) -> dict[str, Any]:
        variant = gm.normalize_variant_name(self.config.live_variant)
        if not gm.checkpoint_exists(variant, schema=self.config.live_schema):
            raise RuntimeError(f"Checkpoint tidak ditemukan: {self.config.live_schema}/gru_{variant}")
        return gm.load_metadata(variant, schema=self.config.live_schema)

    def prepare_runtime(self) -> None:
        metadata = self.validate_checkpoint()
        self.print_line(
            "Model live OK: "
            f"{self.config.live_schema}/gru_{self.config.live_variant} "
            f"target_frames={metadata.get('target_frames', '-')} "
            f"val_acc={metadata.get('best_val_acc', '-')}"
        )
        self._load_tts()
        self._warm_llm()

    def _ollama_model_available(self) -> bool:
        if shutil.which("ollama") is None:
            return False
        proc = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=10, check=False)
        if proc.returncode != 0:
            return False
        wanted = self.config.llm_model.lower()
        wanted_latest = f"{wanted}:latest"
        for line in proc.stdout.splitlines()[1:]:
            name = line.split(None, 1)[0].strip().lower() if line.strip() else ""
            if name in {wanted, wanted_latest}:
                return True
        return False

    def _warm_llm(self) -> None:
        if not self._ollama_model_available():
            self.print_line(
                f"Warning: Ollama model {self.config.llm_model} belum terdeteksi. "
                "Build: ollama create bisindo-sailor2 -f LLM/Modelfile.sailor2"
            )
            return
        started = time.perf_counter()
        try:
            output = lp.warmup_sentence_llm(
                model=self.config.llm_model,
                timeout=self.config.llm_warmup_timeout_sec,
            )
            elapsed = time.perf_counter() - started
            self.print_line(f"LLM warm: {self.config.llm_model} ({elapsed:.2f}s) -> {output}")
        except Exception as exc:
            self.print_line(f"Warning: LLM warmup gagal, nanti fallback ke buffer kalau perlu: {exc}")

    def _load_tts(self) -> None:
        started = time.perf_counter()
        profile_name = self.active_tts_profile
        try:
            runtime = tts_rt.LoadedTTSProfile(profile_name, device=self.config.tts_device)
            runtime.load(warmup=True)
            self.tts_runtime = runtime
            self.print_line(
                f"TTS loaded: {profile_name} ({self.config.tts_device}) "
                f"in {time.perf_counter() - started:.2f}s"
            )
        except Exception as exc:
            self.tts_runtime = None
            self.print_line(f"Warning: TTS profile gagal load, fallback eSpeak: {exc}")

    def set_tts_profile(self, profile_name: str) -> None:
        """Ganti voice TTS saat runtime (mis. saat MODE STS berubah dari HP)."""
        profile_name = str(profile_name or "").strip()
        if not profile_name or profile_name == self.active_tts_profile:
            return
        previous = self.active_tts_profile
        self.active_tts_profile = profile_name
        old_runtime = self.tts_runtime
        self.tts_runtime = None
        if old_runtime is not None:
            try:
                old_runtime.unload()
            except Exception:
                pass
        self.print_line(f"TTS profile ganti: {previous} -> {profile_name}")
        self._load_tts()

    def start_worker(
        self,
        *,
        reset_watchdog: bool = True,
    ) -> None:
        if self.worker is not None and self.worker.is_alive():
            return
        now = time.perf_counter()
        self._worker_started_at = now
        self._last_worker_activity_at = now
        self._worker_started_event_at = None
        self._last_live_status_at = None
        self._has_received_live_status = False
        self._status_count_since_start = 0
        if reset_watchdog:
            self._watchdog_restarted_once = False
        self.worker = self.worker_factory(
            self.config.live_variant,
            status_queue=self.status_queue,
            profile=self.config.live_profile,
            device=self.config.device,
            camera_index=self.config.camera_index,
            confidence_threshold=self.config.confidence_threshold,
            show_window=False,
            preview_queue=self.preview_queue,
            use_jit=True,
            segment_mode="auto",
            schema=self.config.live_schema,
            route=self.config.live_route,
            stream_workers=self.config.stream_workers,
            mp_workers=self.config.mp_workers,
            inference_workers=self.config.inference_workers,
            mp_method=self.config.live_mp_method,
            specialist_enabled=self.config.specialist_enabled,
            specialist_name=self.config.specialist_name,
            stop_on_window_close=False,
        )

    def stop_worker(self) -> None:
        if self.worker is None:
            return
        live_session.stop_worker(self.worker, join_timeout=3.0, force=True)
        self.worker = None
        self._worker_started_at = None
        self._last_worker_activity_at = None
        self._worker_started_event_at = None
        self._last_live_status_at = None
        self._has_received_live_status = False
        self._status_count_since_start = 0

    def toggle_pause(self) -> None:
        if self.paused:
            self.buffer.reset_prediction_tracking()
            self.start_worker(reset_watchdog=True)
            self.paused = False
            self.print_line("RESUME: live jalan lagi.")
            return
        self.preview_controller.close()
        self.stop_worker()
        self.paused = True
        self.print_line("PAUSE: kamera/MediaPipe/model live dilepas. Tekan p untuk lanjut.")

    def toggle_preview(self) -> None:
        self.preview_enabled = not self.preview_enabled
        self.preview_controller.set_enabled(self.preview_enabled)
        state = "on" if self.preview_enabled else "off"
        self.print_line(f"Preview {state}.")

    def append_manual_space(self) -> None:
        result = self.buffer.append_manual_token(lp.MANUAL_SPACE_TOKEN, reason="manual_space")
        self.print_line(f"Buffer: {self.buffer.pending_text() or '-'}")
        if result is not None:
            self.submit_sentence(result)

    def request_stop(self) -> None:
        self.stop_requested = True

    def handle_key(self, key: str) -> None:
        if key == "p":
            self.toggle_pause()
        elif key == "v":
            self.toggle_preview()
        elif key == "space":
            self.append_manual_space()
        elif key == "quit":
            self.request_stop()

    def process_events(self) -> None:
        while True:
            try:
                item = self.status_queue.get_nowait()
            except queue.Empty:
                return
            event = item.get("event")
            if event == "started":
                self._handle_started(item)
            elif event == "startup":
                self._handle_startup(item)
            elif event == "status":
                self._handle_status(item)
            elif event == "key":
                self.handle_key(str(item.get("key") or ""))
            elif event == "preview":
                self._mark_worker_activity()
                pass
            elif event == "assistant_sentence":
                self._print_sentence_event(item)
            elif event == "camera_retry":
                self._mark_worker_activity()
                self.print_line(f"Camera retry: {item.get('message')}")
            elif event == "error":
                self.print_line(f"Live error: {item.get('message')}")
                self.request_stop()
            elif event == "stopped":
                if item.get("quit_requested"):
                    self.request_stop()
                self.print_line(f"Live stopped: {item.get('message')} | {self._stopped_debug(item)}")

    def _mark_worker_activity(self, now: float | None = None) -> float:
        current = time.perf_counter() if now is None else float(now)
        self._last_worker_activity_at = current
        return current

    def _handle_started(self, item: dict[str, Any]) -> None:
        now = self._mark_worker_activity()
        self._worker_started_event_at = now
        self.print_line(
            "Live started: "
            f"{item.get('schema')}:{item.get('feature_dim')}D "
            f"gru_{item.get('variant')} route={item.get('route')} "
            f"mp={item.get('mp_backend_detail')} preview={int(bool(item.get('preview_enabled')))}"
        )

    def _handle_startup(self, item: dict[str, Any]) -> None:
        self._mark_worker_activity()
        stage = str(item.get("stage") or "startup")
        message = str(item.get("message") or stage)
        self.print_line(f"Live startup: {message}")

    def _handle_status(self, item: dict[str, Any]) -> None:
        now = self._mark_worker_activity()
        self._last_live_status_at = now
        self._has_received_live_status = True
        self._status_count_since_start += 1
        self.last_shoulder = (
            item.get("shoulder_mid_x"),
            item.get("shoulder_mid_y"),
            item.get("shoulder_width"),
        )
        result = self.buffer.observe_status(
            label=str(item.get("prediction", "-")),
            confidence=float(item.get("confidence", 0.0)),
            prediction_id=item.get("prediction_id"),
            visible=bool(item.get("visible", False)),
            now=now,
        )
        if result is not None:
            self.submit_sentence(result)
        if now - self._last_status_print >= PRINT_STATUS_INTERVAL_SEC:
            self._last_status_print = now
            self.print_line(
                f"Live: {item.get('prediction', '-')} ({float(item.get('confidence', 0.0)):.2f}) "
                f"raw={item.get('raw_prediction', '-')} visible={int(bool(item.get('visible', False)))} "
                f"cam={float(item.get('fps_camera', 0.0)):.1f} pred={float(item.get('fps_predict', 0.0)):.1f} "
                f"buffer={self.buffer.pending_text() or '-'}"
            )

    def _stopped_debug(self, item: dict[str, Any]) -> str:
        return (
            f"camera_released={int(bool(item.get('camera_released', False)))} "
            f"camera_thread_alive={int(bool(item.get('camera_thread_alive_after_release', False)))} "
            f"mp_thread_alive={int(bool(item.get('mp_thread_alive_after_stop', False)))} "
            f"predictor_thread_alive={int(bool(item.get('predictor_thread_alive_after_stop', False)))}"
        )

    def check_worker_watchdog(self, now: float | None = None) -> None:
        if self.worker is None or self.paused or self.stop_requested:
            return
        if hasattr(self.worker, "is_alive") and not self.worker.is_alive():
            return
        if self._worker_started_at is None:
            return
        current = time.perf_counter() if now is None else float(now)
        if self._has_received_live_status and self._last_live_status_at is not None:
            elapsed = current - self._last_live_status_at
            if elapsed < LIVE_STALE_STATUS_SEC:
                return
            reason = f"status live stale {elapsed:.1f}s"
        elif self._worker_started_event_at is not None:
            elapsed = current - self._worker_started_event_at
            if elapsed < WORKER_FIRST_STATUS_TIMEOUT_SEC:
                return
            reason = f"belum ada status live {elapsed:.1f}s sejak live started"
        else:
            activity_at = self._last_worker_activity_at or self._worker_started_at
            idle_elapsed = current - activity_at
            if idle_elapsed < WORKER_STARTUP_TIMEOUT_SEC:
                return
            total_elapsed = current - self._worker_started_at
            reason = f"startup live tidak ada progress {idle_elapsed:.1f}s (sejak start {total_elapsed:.1f}s)"
        self._handle_watchdog_stale(reason)

    def _handle_watchdog_stale(self, reason: str) -> None:
        if not self._watchdog_restarted_once:
            self.preview_controller.close()
            self.print_line(f"Watchdog: {reason}; restart worker sekali dengan preview off.")
            self.stop_worker()
            self._watchdog_restarted_once = True
            self.start_worker(reset_watchdog=False)
            return
        self.print_line(
            f"Watchdog: {reason} setelah restart. Live dipause, buffer tetap aman. Tekan p untuk coba lanjut."
        )
        self.preview_enabled = False
        self.preview_controller.set_enabled(False)
        self.stop_worker()
        self.paused = True

    def submit_sentence(self, result: lp.FlushResult) -> None:
        words = result.words
        input_text = result.text
        self.print_line(f"Buffer kirim ({result.reason}): {input_text}")

        def task() -> None:
            llm_ok = True
            llm_error = ""
            llm_started = time.perf_counter()
            output_text = input_text
            try:
                output_text = lp.OllamaSentenceClient(
                    model=self.config.llm_model,
                    timeout=self.config.llm_timeout_sec,
                ).compose(words)
            except Exception as exc:
                llm_ok = False
                llm_error = str(exc)
            llm_sec = time.perf_counter() - llm_started
            tts_info = self._speak_text(output_text)
            self.status_queue.put(
                {
                    "event": "assistant_sentence",
                    "ok": llm_ok,
                    "input": input_text,
                    "output": output_text,
                    "llm_sec": llm_sec,
                    "llm_error": llm_error,
                    **tts_info,
                }
            )

        thread = threading.Thread(target=task, daemon=True)
        self.sentence_threads.append(thread)
        thread.start()

    def _speak_text(self, text: str) -> dict[str, Any]:
        clean_text = str(text or "").strip()
        if not clean_text:
            return {"tts_engine": "-", "tts_ready_sec": 0.0, "tts_done_sec": 0.0, "tts_error": ""}
        started = time.perf_counter()
        if self.tts_runtime is not None and self.tts_runtime.is_loaded:
            try:
                result = self.tts_runtime.speak(clean_text, play=True, player=self.config.tts_player)
                done_sec = time.perf_counter() - started
                return {
                    "tts_engine": "loaded TTS",
                    "tts_ready_sec": float(result.timing_sec.get("total", done_sec)),
                    "tts_done_sec": done_sec,
                    "tts_error": "",
                }
            except Exception as exc:
                self.espeak.say(clean_text)
                return {
                    "tts_engine": "eSpeak fallback",
                    "tts_ready_sec": time.perf_counter() - started,
                    "tts_done_sec": time.perf_counter() - started,
                    "tts_error": str(exc),
                }
        self.espeak.say(clean_text)
        elapsed = time.perf_counter() - started
        return {"tts_engine": "eSpeak", "tts_ready_sec": elapsed, "tts_done_sec": elapsed, "tts_error": ""}

    def _print_sentence_event(self, item: dict[str, Any]) -> None:
        suffix = f" | LLM error: {item.get('llm_error')}" if not item.get("ok") else ""
        tts_error = f" | TTS error: {item.get('tts_error')}" if item.get("tts_error") else ""
        self.print_line(
            f"Output akhir: {item.get('output')} | "
            f"LLM {float(item.get('llm_sec', 0.0)):.2f}s | "
            f"{item.get('tts_engine')} ready {float(item.get('tts_ready_sec', 0.0)):.2f}s "
            f"done {float(item.get('tts_done_sec', 0.0)):.2f}s"
            f"{suffix}{tts_error}"
        )

    def run(self) -> int:
        self.prepare_runtime()
        self.start_worker()
        self.print_line("Kontrol: p=pause/resume, v=preview, space={spasi}, q/Esc=keluar.")
        try:
            with TerminalKeyReader() as key_reader:
                while not self.stop_requested:
                    key = key_reader.read_key()
                    if key:
                        self.handle_key(key)
                    self.process_events()
                    preview_key = self.preview_controller.poll()
                    if preview_key:
                        self.handle_key(preview_key)
                    if self.worker is not None and not self.worker.is_alive():
                        live_session.join_worker(self.worker, timeout=0.1)
                        if not self.stop_requested and not self.paused:
                            self.paused = True
                            self.print_line("Live worker berhenti. Tekan p untuk start lagi.")
                        self.worker = None
                    self.check_worker_watchdog()
                    time.sleep(LOOP_SLEEP_SEC)
        except KeyboardInterrupt:
            self.print_line("\nStopping...")
        finally:
            self.shutdown()
        return 0

    def shutdown(self) -> None:
        self.preview_controller.close()
        self.stop_worker()
        for thread in list(self.sentence_threads):
            thread.join(timeout=0.2)
        if self.tts_runtime is not None:
            self.tts_runtime.unload()
            self.tts_runtime = None
        self.espeak.stop()
        self.espeak.join(timeout=1.0)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Live BISINDO Smart180 ADI assistant")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--tts-device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--tts-player", default="auto")
    parser.add_argument("--threshold", type=float, default=LIVE_CONFIDENCE_THRESHOLD)
    parser.add_argument("--specialist", default="all", help="Nama specialist aktif, atau all untuk semua specialist")
    parser.add_argument("--no-specialist", action="store_true", help="Matikan auto-specialist routing")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    config = AssistantConfig(
        camera_index=args.camera,
        device=args.device,
        tts_device=args.tts_device,
        tts_player=args.tts_player,
        confidence_threshold=args.threshold,
        specialist_enabled=not args.no_specialist,
        specialist_name=args.specialist,
    )
    return BISINDOLiveAssistant(config).run()


if __name__ == "__main__":
    raise SystemExit(main())
