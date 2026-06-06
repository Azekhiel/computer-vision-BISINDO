"""Optional bridge from the main UI to the standalone src_test_tts runtime."""

from __future__ import annotations

from dataclasses import dataclass
import gc
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_TEST_TTS_DIR = ROOT_DIR / "src_test_tts"
OUTPUT_DIR = SRC_TEST_TTS_DIR / "outputs" / "main_ui_tts"
GENDER_DISPLAY_OPTIONS = ("Cewek", "Cowok")
DEMOGRAFI_DISPLAY_OPTIONS = ("Dewasa", "Remaja", "Anak-Anak")


class TTSProfileError(RuntimeError):
    pass


@dataclass(frozen=True)
class TTSGenerateResult:
    text: str
    final_wav_path: Path
    raw_wav_path: Path
    metadata_path: Path
    timing_sec: dict[str, float]
    played_with: str | None = None


def _ensure_src_test_tts_path() -> None:
    if not SRC_TEST_TTS_DIR.exists():
        raise TTSProfileError(f"Folder src_test_tts tidak ditemukan: {SRC_TEST_TTS_DIR}")
    path = str(SRC_TEST_TTS_DIR)
    if path not in sys.path:
        sys.path.insert(0, path)


def available_profiles() -> list[str]:
    _ensure_src_test_tts_path()
    try:
        from profile_manager import VoiceProfileManager
    except Exception as exc:
        raise TTSProfileError(f"Gagal import profile manager TTS: {exc}") from exc
    try:
        return VoiceProfileManager().list_names()
    except Exception as exc:
        raise TTSProfileError(f"Gagal membaca profile TTS: {exc}") from exc


def profiles_for_picker(gender: str, demografi: str) -> list[str]:
    _ensure_src_test_tts_path()
    try:
        from profile_manager import VoiceProfileManager, profile_names_for
    except Exception as exc:
        raise TTSProfileError(f"Gagal import profile picker TTS: {exc}") from exc
    try:
        return profile_names_for(VoiceProfileManager().list_names(), gender, demografi)
    except Exception as exc:
        raise TTSProfileError(f"Gagal memfilter profile TTS: {exc}") from exc


def normalize_gender_choice(value: str) -> str:
    _ensure_src_test_tts_path()
    try:
        from profile_manager import normalize_gender_choice as normalize

        return normalize(value)
    except Exception as exc:
        raise TTSProfileError(f"Gagal membaca pilihan gender TTS: {exc}") from exc


def normalize_demografi_choice(value: str) -> str:
    _ensure_src_test_tts_path()
    try:
        from profile_manager import normalize_demografi_choice as normalize

        return normalize(value)
    except Exception as exc:
        raise TTSProfileError(f"Gagal membaca pilihan usia TTS: {exc}") from exc


def display_gender(value: str) -> str:
    _ensure_src_test_tts_path()
    try:
        from profile_manager import display_gender as display

        return display(value)
    except Exception as exc:
        raise TTSProfileError(f"Gagal membaca label gender TTS: {exc}") from exc


def display_demografi(value: str) -> str:
    _ensure_src_test_tts_path()
    try:
        from profile_manager import display_demografi as display

        return display(value)
    except Exception as exc:
        raise TTSProfileError(f"Gagal membaca label usia TTS: {exc}") from exc


def profile_detail(profile_name: str) -> dict[str, Any]:
    _ensure_src_test_tts_path()
    try:
        from profile_manager import VoiceProfileManager

        return VoiceProfileManager().get(profile_name)
    except Exception as exc:
        raise TTSProfileError(f"Gagal membaca detail profile TTS {profile_name!r}: {exc}") from exc


class LoadedTTSProfile:
    def __init__(self, profile_name: str, device: str = "auto", output_dir: Path = OUTPUT_DIR) -> None:
        self.profile_name = str(profile_name)
        self.device = str(device or "auto")
        self.output_dir = Path(output_dir)
        self.runtime: Any | None = None
        self.profile: dict[str, Any] | None = None

    @property
    def is_loaded(self) -> bool:
        return self.runtime is not None

    def load(self, warmup: bool = True) -> float:
        _ensure_src_test_tts_path()
        start = time.perf_counter()
        try:
            from profile_manager import VoiceProfileManager
            from tts_engine import InProcessTTSRuntime
        except Exception as exc:
            raise TTSProfileError(f"Gagal import runtime TTS: {exc}") from exc
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.profile = VoiceProfileManager().get(self.profile_name)
        self.runtime = InProcessTTSRuntime(device=self.device)
        self.runtime.load()
        if warmup:
            self.runtime.warmup(self.output_dir)
        return time.perf_counter() - start

    def speak(
        self,
        text: str,
        *,
        play: bool = True,
        player: str = "auto",
        sink_name: str | None = None,
    ) -> TTSGenerateResult:
        if self.runtime is None or self.profile is None:
            raise TTSProfileError("TTS profile belum di-load. Klik Preload + Warmup dulu.")
        _ensure_src_test_tts_path()
        try:
            from tts_engine import generate_from_profile
        except Exception as exc:
            raise TTSProfileError(f"Gagal import generator TTS: {exc}") from exc
        result = generate_from_profile(
            text,
            self.profile_name,
            self.profile,
            self.output_dir,
            analyze=False,
            runtime=self.runtime,
        )
        played_with = None
        if play:
            played_with = play_audio(result.final_wav_path, player=player, sink_name=sink_name)
        metadata = result.metadata or {}
        return TTSGenerateResult(
            text=text,
            final_wav_path=Path(result.final_wav_path),
            raw_wav_path=Path(result.raw_wav_path),
            metadata_path=Path(result.metadata_path),
            timing_sec=dict(metadata.get("timing_sec") or {}),
            played_with=played_with,
        )

    def unload(self) -> None:
        self.runtime = None
        self.profile = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def available_players() -> list[str]:
    return [name for name in ("paplay", "aplay", "ffplay") if shutil.which(name)]


def play_audio(path: Path, *, player: str = "auto", sink_name: str | None = None) -> str:
    path = Path(path)
    if not path.exists():
        raise TTSProfileError(f"Audio TTS tidak ditemukan: {path}")
    selected = _select_player(player)
    cmd = _player_command(selected, path)
    env = os.environ.copy()
    if sink_name and selected == "paplay":
        env["PULSE_SINK"] = sink_name
    proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, env=env, check=False)
    if proc.returncode != 0:
        raise TTSProfileError(f"Gagal memutar audio dengan {selected}: {proc.stderr.strip()}")
    return selected


def _select_player(player: str) -> str:
    if player != "auto":
        if shutil.which(player):
            return player
        raise TTSProfileError(f"Audio player tidak ditemukan: {player}")
    for candidate in ("paplay", "aplay", "ffplay"):
        if shutil.which(candidate):
            return candidate
    raise TTSProfileError("Tidak ada audio player: butuh paplay, aplay, atau ffplay.")


def _player_command(player: str, path: Path) -> list[str]:
    if player == "paplay":
        return ["paplay", str(path)]
    if player == "aplay":
        return ["aplay", str(path)]
    if player == "ffplay":
        return ["ffplay", "-nodisp", "-autoexit", "-loglevel", "error", str(path)]
    raise TTSProfileError(f"Audio player belum didukung: {player}")
