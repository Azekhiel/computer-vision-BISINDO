"""Runtime BISINDO headless yang dikendalikan lewat MQTT (HP Android <-> Jetson).

Jalankan setelah broker Mosquitto aktif (lihat src_integrasi/mqtt/README.md):

    python -m src_integrasi.mqtt_runtime

Alur:
- WORD tiap gesture dikenali  -> publish topic WORD + update topic SENTENCE.
- HP kirim MODE               -> ganti mode aktif + voice TTS (STS_*).
- HP kirim SENTENCEOK=GAS      -> rapikan buffer dgn LLM, suarakan, lalu clear.
- HP kirim SENTENCEOK=NO       -> clear buffer tanpa disuarakan.
- HP kirim WORDDEL              -> hapus kata terakhir (backspace), kirim ulang SENTENCE terkoreksi.
- Posisi bahu MediaPipe        -> publish topic CAMPOS (OK/UP/DOWN/LEFT/RIGHT).
- Tombol 'c' di terminal       -> kalibrasi posisi kamera acuan ke JSON.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import threading
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
for _base in (ROOT_DIR, ROOT_DIR / "src", ROOT_DIR / "LLM"):
    _path = str(_base)
    if _path not in sys.path:
        sys.path.insert(0, _path)

import bisindo_live_assistant as live_assistant  # noqa: E402
import livetest_plus as lp  # noqa: E402

from src_integrasi import camera_position as campos  # noqa: E402
from src_integrasi import mode_manager  # noqa: E402
from src_integrasi.configuration import load_configuration  # noqa: E402
from src_integrasi.mqtt_bridge import MqttBridge  # noqa: E402


class MqttBisindoRuntime(live_assistant.BISINDOLiveAssistant):
    """BISINDOLiveAssistant headless yang nyambung ke MQTT broker lokal."""

    def __init__(
        self,
        config: live_assistant.AssistantConfig | None = None,
        *,
        mqtt_host: str = "localhost",
        mqtt_port: int = 1883,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, **kwargs)
        # GAS-gated: buffer tidak pernah flush otomatis; finalisasi hanya saat GAS.
        self.buffer = lp.SentenceBuffer(
            max_words=self.config.buffer_max_words,
            idle_no_hand_sec=self.config.buffer_send_delay_sec,
            auto_flush=False,
        )
        self.current_mode = mode_manager.DEFAULT_MODE
        self.active_tts_profile = mode_manager.DEFAULT_TTS_PROFILE
        self.cam_cfg = campos.CameraPositionConfig.load()

        self._buffer_lock = threading.Lock()
        self._published_word_count = 0
        self._last_campos: str | None = None
        self._tts_reload_thread: threading.Thread | None = None

        self.bridge = MqttBridge(
            host=mqtt_host,
            port=mqtt_port,
            on_mode=self._on_mode,
            on_sentenceok=self._on_sentenceok,
            on_worddel=self._on_worddel,
            logger=self.print_line,
        )

    # --- lifecycle ---
    def prepare_runtime(self) -> None:
        super().prepare_runtime()
        if self.bridge.connect():
            # Reset display awal di HP.
            self.bridge.publish_sentence("")
        else:
            self.print_line("Warning: lanjut tanpa MQTT (broker tidak terhubung).")
        self.print_line("Kontrol tambahan: tekan 'c' untuk kalibrasi posisi kamera (CAMPOS).")

    def shutdown(self) -> None:
        try:
            self.bridge.disconnect()
        finally:
            super().shutdown()

    # --- MQTT command handlers (dipanggil dari thread paho) ---
    def _on_mode(self, payload: str) -> None:
        self.current_mode = payload
        _is_sts, profile = mode_manager.resolve(payload)
        self.print_line(f"MODE: {payload} (voice={profile})")
        # Reload TTS di thread terpisah supaya tidak memblok loop paho.
        if self._tts_reload_thread is not None and self._tts_reload_thread.is_alive():
            return
        self._tts_reload_thread = threading.Thread(
            target=self.set_tts_profile, args=(profile,), daemon=True
        )
        self._tts_reload_thread.start()

    def _on_sentenceok(self, payload: str) -> None:
        if payload == "GAS":
            with self._buffer_lock:
                words = self.buffer.pending_words()
                self.buffer.reset()
                self._published_word_count = 0
            if words:
                self.submit_sentence(lp.FlushResult(words, reason="gas"))
            else:
                self.print_line("SENTENCEOK=GAS tapi buffer kosong.")
            self.bridge.publish_sentence("")
        elif payload == "NO":
            with self._buffer_lock:
                self.buffer.reset()
                self._published_word_count = 0
            self.print_line("SENTENCEOK=NO: buffer di-clear.")
            self.bridge.publish_sentence("")

    def _on_worddel(self, _payload: str) -> None:
        # Hapus kata terakhir (backspace) saat HP kirim WORDDEL, lalu kirim ulang SENTENCE terkoreksi.
        with self._buffer_lock:
            removed = self.buffer.pop_word()
            # Sinkronkan counter publish supaya _handle_status tidak salah hitung kata baru.
            self._published_word_count = len(self.buffer.words)
            sentence_text = self.buffer.pending_text()
        if removed is None:
            self.print_line("WORDDEL: buffer kosong, tidak ada yang dihapus.")
            return
        self.print_line(f"WORDDEL: hapus '{removed}' -> {sentence_text or '(kosong)'}")
        self.bridge.publish_sentence(sentence_text)

    # --- status hook: publish WORD/SENTENCE/CAMPOS ---
    def _handle_status(self, item: dict[str, Any]) -> None:
        super()._handle_status(item)  # bookkeeping watchdog + observe_status + print

        with self._buffer_lock:
            current = len(self.buffer.words)
            new_words = list(self.buffer.words[self._published_word_count : current])
            sentence_text = self.buffer.pending_text()
            self._published_word_count = current

        for word in new_words:
            self.bridge.publish_word(word)
        if new_words:
            self.bridge.publish_sentence(sentence_text)

        self._update_campos(item)

    def _update_campos(self, item: dict[str, Any]) -> None:
        mid_x = item.get("shoulder_mid_x")
        mid_y = item.get("shoulder_mid_y")
        status = self.cam_cfg.evaluate(mid_x, mid_y)
        if status is None:
            return
        if status != self._last_campos:
            self._last_campos = status
            self.bridge.publish_campos(status)

    # --- STT hook (dipakai bila sumber speech-to-text aktif) ---
    def publish_stt(self, text: str) -> None:
        self.bridge.publish_stt(text)

    # --- kalibrasi posisi kamera (tombol 'c') ---
    def handle_key(self, key: str) -> None:
        if key == "c":
            self._calibrate_camera()
            return
        super().handle_key(key)

    def _calibrate_camera(self) -> None:
        cx, cy, cw = self.last_shoulder
        if cx is None or cy is None:
            self.print_line("Kalibrasi gagal: bahu tidak terdeteksi. Pastikan badan terlihat kamera.")
            return
        width = cw if cw is not None else self.cam_cfg.ref_width
        self.cam_cfg.calibrate(cx, cy, width)
        self.cam_cfg.save()
        self._last_campos = None  # paksa publish ulang status berikutnya
        self.print_line(
            f"Kalibrasi tersimpan: mid=({cx:.3f},{cy:.3f}) width={width:.3f} "
            f"tol=({self.cam_cfg.tol_x},{self.cam_cfg.tol_y}) -> {campos.CONFIG_PATH.name}"
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="BISINDO headless runtime via MQTT")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--tts-device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--tts-player", default="auto")
    # threshold/specialist default None -> kalau tak diisi, configuration.json yang menentukan.
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--mqtt-host", default="localhost")
    parser.add_argument("--mqtt-port", type=int, default=1883)
    parser.add_argument("--specialist", default=None)
    parser.add_argument("--no-specialist", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    # schema/model/suite/augmentasi/specialist/threshold/llm dari configuration.json; CLI di bawah
    # hanya override kalau benar-benar diisi user (threshold/specialist default None).
    extra: dict[str, Any] = {
        "camera_index": args.camera,
        "device": args.device,
        "tts_device": args.tts_device,
        "tts_player": args.tts_player,
        "tts_profile_name": mode_manager.DEFAULT_TTS_PROFILE,
    }
    if args.threshold is not None:
        extra["confidence_threshold"] = args.threshold
    if args.no_specialist:
        extra["specialist_enabled"] = False
    elif args.specialist is not None:
        extra["specialist_enabled"] = True
        extra["specialist_name"] = args.specialist
    config = load_configuration(**extra)
    runtime = MqttBisindoRuntime(config, mqtt_host=args.mqtt_host, mqtt_port=args.mqtt_port)
    return runtime.run()


if __name__ == "__main__":
    raise SystemExit(main())
