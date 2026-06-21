"""Small ALSA/I2S playback helper for generated TTS WAV files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
import threading
import time
from typing import Callable


I2S_ALSA_DEVICE = "plughw:CARD=APE,DEV=0"
I2S_APLAY_BINARY = "aplay"


class I2SAudioError(RuntimeError):
    pass


@dataclass(frozen=True)
class I2SPlaybackResult:
    wav_path: Path
    device: str
    player: str
    command: tuple[str, ...]
    elapsed_sec: float


class I2SAudioPlayer:
    def __init__(
        self,
        *,
        device: str = I2S_ALSA_DEVICE,
        aplay_binary: str = I2S_APLAY_BINARY,
        popen_factory: Callable[..., subprocess.Popen] = subprocess.Popen,
    ) -> None:
        self.device = str(device or I2S_ALSA_DEVICE)
        self.aplay_binary = str(aplay_binary or I2S_APLAY_BINARY)
        self._popen_factory = popen_factory
        self._lock = threading.Lock()
        self._process: subprocess.Popen | None = None

    def command_for(self, wav_path: Path) -> list[str]:
        wav = Path(wav_path)
        if not wav.exists():
            raise I2SAudioError(f"WAV untuk I2S tidak ditemukan: {wav}")
        player = self._resolve_aplay()
        return [player, "-D", self.device, str(wav)]

    def play(self, wav_path: Path) -> I2SPlaybackResult:
        wav = Path(wav_path)
        cmd = self.command_for(wav)
        started = time.perf_counter()
        proc = self._popen_factory(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        with self._lock:
            self._process = proc
        try:
            _, stderr = proc.communicate()
        finally:
            with self._lock:
                if self._process is proc:
                    self._process = None
        elapsed = time.perf_counter() - started
        if proc.returncode != 0:
            message = (stderr or "").strip()
            raise I2SAudioError(
                f"Gagal memutar audio I2S device={self.device!r} dengan {cmd[0]} "
                f"(exit {proc.returncode}): {message or '-'}; cek `aplay -l` dan routing I2S/mixer Jetson."
            )
        return I2SPlaybackResult(
            wav_path=wav,
            device=self.device,
            player=cmd[0],
            command=tuple(cmd),
            elapsed_sec=elapsed,
        )

    def stop(self) -> None:
        with self._lock:
            proc = self._process
        if proc is None or proc.poll() is not None:
            return
        try:
            proc.terminate()
            proc.wait(timeout=1.0)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    def _resolve_aplay(self) -> str:
        if "/" in self.aplay_binary:
            path = Path(self.aplay_binary)
            if path.exists():
                return str(path)
            raise I2SAudioError(f"aplay untuk I2S tidak ditemukan: {self.aplay_binary}")
        resolved = shutil.which(self.aplay_binary)
        if resolved:
            return resolved
        raise I2SAudioError(f"aplay untuk I2S tidak ditemukan: {self.aplay_binary}")
