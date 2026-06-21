"""Terminal live assistant variant that plays TTS audio through ALSA/I2S."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Callable

import bisindo_live_assistant as base
from bisindo_live_assistant import AssistantConfig, BISINDOLiveAssistant, PreviewController
import i2s_audio
import tts_profile_runtime as tts_rt


I2S_ALSA_DEVICE = i2s_audio.I2S_ALSA_DEVICE
I2S_APLAY_BINARY = i2s_audio.I2S_APLAY_BINARY
I2S_FALLBACK_TO_NORMAL_PLAYER = False


@dataclass(frozen=True)
class I2SAssistantConfig(AssistantConfig):
    i2s_device: str = I2S_ALSA_DEVICE
    i2s_aplay_binary: str = I2S_APLAY_BINARY
    i2s_fallback_to_normal_player: bool = I2S_FALLBACK_TO_NORMAL_PLAYER


class BISINDOLiveAssistantI2S(BISINDOLiveAssistant):
    def __init__(
        self,
        config: I2SAssistantConfig | None = None,
        *,
        worker_factory: Callable[..., Any] | None = None,
        preview_controller: PreviewController | None = None,
        audio_player: i2s_audio.I2SAudioPlayer | None = None,
    ) -> None:
        i2s_config = config or I2SAssistantConfig()
        super().__init__(
            i2s_config,
            worker_factory=worker_factory,
            preview_controller=preview_controller,
        )
        self.config: I2SAssistantConfig = i2s_config
        self.i2s_player = audio_player or i2s_audio.I2SAudioPlayer(
            device=i2s_config.i2s_device,
            aplay_binary=i2s_config.i2s_aplay_binary,
        )

    def _speak_text(self, text: str) -> dict[str, Any]:
        clean_text = str(text or "").strip()
        if not clean_text:
            return {
                "tts_engine": "-",
                "tts_ready_sec": 0.0,
                "i2s_play_sec": 0.0,
                "tts_done_sec": 0.0,
                "tts_error": "",
            }
        started = time.perf_counter()
        if self.tts_runtime is None or not self.tts_runtime.is_loaded:
            if self.config.i2s_fallback_to_normal_player:
                return super()._speak_text(clean_text)
            elapsed = time.perf_counter() - started
            return {
                "tts_engine": "I2S TTS unavailable",
                "tts_ready_sec": elapsed,
                "i2s_play_sec": 0.0,
                "tts_done_sec": elapsed,
                "tts_error": "TTS profile belum loaded; I2S fallback normal off",
            }

        generated = None
        generate_sec = 0.0
        try:
            gen_started = time.perf_counter()
            generated = self.tts_runtime.speak(clean_text, play=False)
            generate_sec = time.perf_counter() - gen_started
            played = self.i2s_player.play(generated.final_wav_path)
            done_sec = time.perf_counter() - started
            return {
                "tts_engine": f"loaded TTS I2S {played.device}",
                "tts_ready_sec": float(generated.timing_sec.get("total", generate_sec)),
                "i2s_play_sec": float(played.elapsed_sec),
                "tts_done_sec": done_sec,
                "tts_error": "",
            }
        except Exception as exc:
            if self.config.i2s_fallback_to_normal_player:
                try:
                    if generated is not None:
                        fallback_started = time.perf_counter()
                        played_with = tts_rt.play_audio(generated.final_wav_path, player=self.config.tts_player)
                        done_sec = time.perf_counter() - started
                        return {
                            "tts_engine": f"normal player fallback {played_with}",
                            "tts_ready_sec": float(generated.timing_sec.get("total", generate_sec)),
                            "i2s_play_sec": time.perf_counter() - fallback_started,
                            "tts_done_sec": done_sec,
                            "tts_error": str(exc),
                        }
                    fallback_info = super()._speak_text(clean_text)
                    fallback_info["tts_error"] = str(exc)
                    return fallback_info
                except Exception as fallback_exc:
                    elapsed = time.perf_counter() - started
                    return {
                        "tts_engine": "I2S + fallback failed",
                        "tts_ready_sec": generate_sec,
                        "i2s_play_sec": 0.0,
                        "tts_done_sec": elapsed,
                        "tts_error": f"{exc}; fallback: {fallback_exc}",
                    }
            elapsed = time.perf_counter() - started
            return {
                "tts_engine": f"I2S error {self.config.i2s_device}",
                "tts_ready_sec": generate_sec,
                "i2s_play_sec": 0.0,
                "tts_done_sec": elapsed,
                "tts_error": str(exc),
            }

    def _print_sentence_event(self, item: dict[str, Any]) -> None:
        suffix = f" | LLM error: {item.get('llm_error')}" if not item.get("ok") else ""
        tts_error = f" | TTS error: {item.get('tts_error')}" if item.get("tts_error") else ""
        self.print_line(
            f"Output akhir: {item.get('output')} | "
            f"LLM {float(item.get('llm_sec', 0.0)):.2f}s | "
            f"{item.get('tts_engine')} gen {float(item.get('tts_ready_sec', 0.0)):.2f}s "
            f"I2S {float(item.get('i2s_play_sec', 0.0)):.2f}s "
            f"done {float(item.get('tts_done_sec', 0.0)):.2f}s"
            f"{suffix}{tts_error}"
        )

    def shutdown(self) -> None:
        self.i2s_player.stop()
        super().shutdown()


def build_arg_parser():
    parser = base.build_arg_parser()
    parser.description = "Live BISINDO Smart180 ADI assistant with ALSA/I2S audio"
    parser.add_argument("--i2s-device", default=I2S_ALSA_DEVICE)
    parser.add_argument("--aplay", default=I2S_APLAY_BINARY)
    parser.add_argument("--i2s-fallback-normal", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    config = I2SAssistantConfig(
        camera_index=args.camera,
        device=args.device,
        tts_device=args.tts_device,
        tts_player=args.tts_player,
        confidence_threshold=args.threshold,
        i2s_device=args.i2s_device,
        i2s_aplay_binary=args.aplay,
        i2s_fallback_to_normal_player=bool(args.i2s_fallback_normal),
    )
    return BISINDOLiveAssistantI2S(config).run()


if __name__ == "__main__":
    raise SystemExit(main())
