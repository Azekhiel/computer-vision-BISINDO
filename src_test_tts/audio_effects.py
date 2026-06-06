import math
import shutil
import subprocess
import tempfile
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy import signal


class AudioEffectError(RuntimeError):
    pass


def apply_profile_effects(raw_wav_path: Path, output_wav_path: Path, profile: Dict[str, Any]) -> Tuple[Path, List[str]]:
    warnings_list: List[str] = []
    raw_wav_path = Path(raw_wav_path)
    output_wav_path = Path(output_wav_path)
    output_wav_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        audio, sample_rate = _read_audio(raw_wav_path)
    except Exception as exc:
        raise AudioEffectError(f"Gagal membaca WAV raw: {raw_wav_path} ({exc})") from exc
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    audio = audio.astype(np.float32)

    highpass = float(profile.get("highpass_hz") or 0)
    lowpass = float(profile.get("lowpass_hz") or 0)
    if highpass > 0:
        audio = _filter(audio, sample_rate, highpass, btype="highpass")
    if lowpass > 0 and lowpass < sample_rate / 2:
        audio = _filter(audio, sample_rate, lowpass, btype="lowpass")

    pitch = float(profile.get("pitch_semitones") or 0.0)
    speed = float(profile.get("speed") or 1.0)
    if abs(pitch) > 0.001:
        audio, sample_rate, warning = _pitch_shift(audio, sample_rate, pitch)
        if warning:
            warnings_list.append(warning)
    if abs(speed - 1.0) > 0.001:
        audio, warning = _time_stretch(audio, speed)
        if warning:
            warnings_list.append(warning)

    formant_shift = float(profile.get("formant_shift") or 0.0)
    if abs(formant_shift) > 0.001:
        warnings_list.append("formant_shift belum diterapkan: rubberband/praat backend belum tersedia.")

    gain_db = float(profile.get("volume_gain_db") or 0.0)
    if abs(gain_db) > 0.001:
        audio = audio * (10.0 ** (gain_db / 20.0))

    if bool(profile.get("compressor")):
        audio = _compress(audio)
    if bool(profile.get("normalize")):
        audio = _normalize(audio)

    out_rate = int(profile.get("output_sample_rate") or sample_rate)
    if out_rate != sample_rate:
        audio = signal.resample_poly(audio, out_rate, sample_rate)
        sample_rate = out_rate

    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak >= 0.98:
        warnings_list.append("clipping_detected: peak audio mendekati/melewati batas aman.")
        audio = np.clip(audio, -0.98, 0.98)

    _write_audio(output_wav_path, audio, sample_rate)
    return output_wav_path, warnings_list


def _read_audio(path: Path) -> Tuple[np.ndarray, int]:
    try:
        import soundfile as sf

        audio, sample_rate = sf.read(path, always_2d=False)
        return np.asarray(audio), int(sample_rate)
    except ImportError as exc:
        raise AudioEffectError(
            "Dependency soundfile belum tersedia. Aktifkan venv lalu install requirements."
        ) from exc


def _write_audio(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    try:
        import soundfile as sf

        sf.write(path, np.asarray(audio, dtype=np.float32), sample_rate, subtype="PCM_16")
    except ImportError as exc:
        raise AudioEffectError(
            "Dependency soundfile belum tersedia. Aktifkan venv lalu install requirements."
        ) from exc


def _filter(audio: np.ndarray, sample_rate: int, cutoff: float, btype: str) -> np.ndarray:
    sos = signal.butter(4, cutoff, btype=btype, fs=sample_rate, output="sos")
    return signal.sosfiltfilt(sos, audio).astype(np.float32)


def _pitch_shift(audio: np.ndarray, sample_rate: int, semitones: float) -> Tuple[np.ndarray, int, str]:
    if shutil.which("ffmpeg"):
        ratio = 2.0 ** (semitones / 12.0)
        with tempfile.TemporaryDirectory() as tmpdir:
            in_path = Path(tmpdir) / "in.wav"
            out_path = Path(tmpdir) / "out.wav"
            _write_audio(in_path, audio, sample_rate)
            # asetrate changes pitch, atempo restores duration approximately.
            tempo = 1.0 / ratio
            filters = [f"asetrate={sample_rate * ratio:.3f}", f"aresample={sample_rate}"]
            filters.extend(_atempo_filters(tempo))
            cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(in_path), "-af", ",".join(filters), str(out_path)]
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if result.returncode == 0 and out_path.exists():
                shifted, sr = _read_audio(out_path)
                return shifted.astype(np.float32), sr, ""
    try:
        import librosa

        shifted = librosa.effects.pitch_shift(y=audio.astype(np.float32), sr=sample_rate, n_steps=semitones)
        return shifted.astype(np.float32), sample_rate, ""
    except Exception:
        factor = 2.0 ** (semitones / 12.0)
        resampled = signal.resample_poly(audio, 1000, max(1, int(1000 * factor)))
        restored = signal.resample(resampled, len(audio))
        return restored.astype(np.float32), sample_rate, "pitch_shift memakai fallback sederhana; cek naturalitas audio manual."


def _time_stretch(audio: np.ndarray, speed: float) -> Tuple[np.ndarray, str]:
    if speed <= 0:
        raise AudioEffectError("speed harus lebih besar dari 0.")
    try:
        import librosa

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stretched = librosa.effects.time_stretch(audio.astype(np.float32), rate=speed)
        return stretched.astype(np.float32), ""
    except Exception:
        target_len = max(1, int(len(audio) / speed))
        stretched = signal.resample(audio, target_len)
        return stretched.astype(np.float32), "tempo memakai fallback resampling; cek naturalitas audio manual."


def _atempo_filters(tempo: float) -> List[str]:
    values: List[float] = []
    while tempo < 0.5:
        values.append(0.5)
        tempo /= 0.5
    while tempo > 2.0:
        values.append(2.0)
        tempo /= 2.0
    values.append(tempo)
    return [f"atempo={value:.6f}" for value in values]


def _normalize(audio: np.ndarray, target_peak: float = 0.90) -> np.ndarray:
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak <= 1e-6:
        return audio
    return (audio / peak * target_peak).astype(np.float32)


def _compress(audio: np.ndarray, threshold_db: float = -18.0, ratio: float = 3.0) -> np.ndarray:
    threshold = 10.0 ** (threshold_db / 20.0)
    sign = np.sign(audio)
    mag = np.abs(audio)
    over = mag > threshold
    mag[over] = threshold + (mag[over] - threshold) / ratio
    return (sign * mag).astype(np.float32)

