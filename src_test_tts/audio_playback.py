import shutil
import subprocess
from pathlib import Path
from typing import Iterable, List, Optional


class AudioPlaybackError(RuntimeError):
    pass


def available_players() -> List[str]:
    players = []
    for name in ("paplay", "aplay", "ffplay"):
        if shutil.which(name):
            players.append(name)
    return players


def play_audio(path: Path, player: str = "auto", wait: bool = True) -> str:
    path = Path(path)
    if not path.exists():
        raise AudioPlaybackError(f"Audio tidak ditemukan: {path}")
    selected = _select_player(player)
    cmd = _command_for_player(selected, path)
    if wait:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise AudioPlaybackError(
                f"Gagal memutar audio dengan {selected}.\n"
                f"STDERR: {result.stderr.strip()}\n"
                f"STDOUT: {result.stdout.strip()}"
            )
    else:
        subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return selected


def _select_player(player: str) -> str:
    if player != "auto":
        if shutil.which(player):
            return player
        raise AudioPlaybackError(f"Player audio tidak ditemukan: {player}")
    for candidate in ("paplay", "aplay", "ffplay"):
        if shutil.which(candidate):
            return candidate
    raise AudioPlaybackError("Tidak ada player audio: butuh paplay, aplay, atau ffplay.")


def _command_for_player(player: str, path: Path) -> List[str]:
    if player == "paplay":
        return ["paplay", str(path)]
    if player == "aplay":
        return ["aplay", str(path)]
    if player == "ffplay":
        return ["ffplay", "-nodisp", "-autoexit", "-loglevel", "error", str(path)]
    raise AudioPlaybackError(f"Player audio belum didukung: {player}")

