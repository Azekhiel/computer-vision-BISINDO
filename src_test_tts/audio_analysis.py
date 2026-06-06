import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
from scipy import signal


def analyze_wav(path: Path, text: str = "") -> Dict[str, Any]:
    warnings: List[str] = []
    path = Path(path)
    audio, sample_rate = _read_audio(path)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    audio = audio.astype(np.float32)
    num_samples = int(audio.shape[0])
    duration = float(num_samples / sample_rate) if sample_rate else 0.0
    rms = float(np.sqrt(np.mean(np.square(audio)))) if num_samples else 0.0
    peak = float(np.max(np.abs(audio))) if num_samples else 0.0
    peak_db = _amp_to_db(peak)
    centroid = _spectral_centroid(audio, sample_rate) if num_samples else None
    f0_mean = None
    f0_median = None
    try:
        f0_values = _estimate_f0(audio, sample_rate)
        if f0_values.size:
            f0_mean = float(np.mean(f0_values))
            f0_median = float(np.median(f0_values))
        else:
            warnings.append("Estimasi F0 tidak menemukan frame voiced.")
    except Exception as exc:  # noqa: BLE001 - analysis must not crash generation.
        warnings.append(f"Estimasi F0 gagal: {exc}")
    clipping = bool(peak >= 0.98)
    words = len([part for part in text.split() if part.strip()]) if text else None
    speech_rate_wpm = float(words / duration * 60.0) if words and duration > 0 else None
    return {
        "path": str(path),
        "estimated_f0_mean_hz": f0_mean,
        "estimated_f0_median_hz": f0_median,
        "duration_sec": duration,
        "rms_loudness": rms,
        "peak_db": peak_db,
        "spectral_centroid_mean": centroid,
        "sample_rate": int(sample_rate),
        "num_samples": num_samples,
        "clipping_detected": clipping,
        "speech_rate_wpm": speech_rate_wpm,
        "warnings": warnings,
    }


def analyze_input(input_path: Path, text: str = "") -> List[Dict[str, Any]]:
    input_path = Path(input_path)
    if input_path.is_dir():
        wavs = sorted(input_path.rglob("*.wav"))
    else:
        wavs = [input_path]
    return [analyze_wav(path, text=text) for path in wavs]


def save_analysis_reports(rows: Iterable[Dict[str, Any]], output_dir: Path) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    json_path = output_dir / "analysis_report.json"
    csv_path = output_dir / "analysis_report.csv"
    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    fieldnames = [
        "path",
        "estimated_f0_mean_hz",
        "estimated_f0_median_hz",
        "duration_sec",
        "rms_loudness",
        "peak_db",
        "spectral_centroid_mean",
        "sample_rate",
        "num_samples",
        "clipping_detected",
        "speech_rate_wpm",
        "warnings",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            record = {key: row.get(key) for key in fieldnames}
            record["warnings"] = "; ".join(row.get("warnings") or [])
            writer.writerow(record)
    return json_path, csv_path


def _read_audio(path: Path) -> tuple[np.ndarray, int]:
    try:
        import soundfile as sf
    except ImportError as exc:
        raise RuntimeError("Dependency soundfile belum tersedia. Install requirements di venv.") from exc
    audio, sample_rate = sf.read(path, always_2d=False)
    return np.asarray(audio), int(sample_rate)


def _amp_to_db(value: float) -> float:
    if value <= 1e-12:
        return -120.0
    return float(20.0 * math.log10(value))


def _spectral_centroid(audio: np.ndarray, sample_rate: int) -> float:
    freqs, times, spectrum = signal.spectrogram(audio, fs=sample_rate, nperseg=min(2048, max(256, len(audio))))
    magnitude = np.abs(spectrum)
    denom = np.sum(magnitude, axis=0)
    valid = denom > 1e-12
    if not np.any(valid):
        return 0.0
    centroid = np.sum(freqs[:, None] * magnitude, axis=0)[valid] / denom[valid]
    return float(np.mean(centroid))


def _estimate_f0(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    try:
        import librosa

        f0 = librosa.yin(audio.astype(np.float32), fmin=50, fmax=600, sr=sample_rate)
        f0 = np.asarray(f0)
        f0 = f0[np.isfinite(f0)]
        return f0[(f0 >= 50) & (f0 <= 600)]
    except Exception:
        return _estimate_f0_autocorr(audio, sample_rate)


def _estimate_f0_autocorr(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    frame_len = int(0.04 * sample_rate)
    hop = int(0.02 * sample_rate)
    if frame_len <= 0 or hop <= 0 or len(audio) < frame_len:
        return np.array([], dtype=np.float32)
    values: List[float] = []
    min_lag = max(1, int(sample_rate / 600))
    max_lag = max(min_lag + 1, int(sample_rate / 50))
    for start in range(0, len(audio) - frame_len, hop):
        frame = audio[start : start + frame_len]
        frame = frame - np.mean(frame)
        if np.sqrt(np.mean(frame * frame)) < 1e-4:
            continue
        corr = np.correlate(frame, frame, mode="full")[frame_len - 1 :]
        search = corr[min_lag:max_lag]
        if search.size == 0:
            continue
        lag = int(np.argmax(search) + min_lag)
        if corr[lag] > 0.25 * corr[0]:
            values.append(sample_rate / lag)
    return np.asarray(values, dtype=np.float32)

