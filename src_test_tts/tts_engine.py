import json
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from html import unescape
from pathlib import Path
from typing import Any, Dict, List, Optional

from datetime import datetime

from audio_analysis import analyze_wav
from audio_effects import apply_profile_effects

from model_downloader import ModelDownloadError, ensure_model_available
from paths import MODEL_DIR, display_path


SPEAKER_TO_ID = {
    "Wibowo": "wibowo",
    "Gadis": "gadis",
}


class TTSEngineError(RuntimeError):
    pass


@dataclass(frozen=True)
class SynthesisResult:
    raw_wav_path: Path
    text_for_tts: str
    speaker_id: str


@dataclass(frozen=True)
class GenerationResult:
    raw_wav_path: Path
    final_wav_path: Path
    metadata_path: Path
    metadata: Dict[str, Any]


def generate_from_profile(
    text: str,
    profile_name: str,
    profile: Dict[str, Any],
    output_dir: Path,
    timestamp: Optional[str] = None,
    analyze: bool = True,
    runtime: Optional["InProcessTTSRuntime"] = None,
) -> GenerationResult:
    total_start = time.perf_counter()
    timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_profile_name = profile_name.replace("/", "_")
    raw_path = output_dir / f"{timestamp}_{safe_profile_name}_raw.wav"
    final_path = output_dir / f"{timestamp}_{safe_profile_name}.wav"
    metadata_path = output_dir / f"{timestamp}_{safe_profile_name}.json"
    if raw_path.exists() or final_path.exists() or metadata_path.exists():
        raise TTSEngineError(f"Output sudah ada untuk timestamp/profile: {timestamp}_{safe_profile_name}")
    synth_start = time.perf_counter()
    if runtime is not None:
        synth = runtime.synthesize_raw(text, profile["base_speaker"], raw_path, speed=profile.get("speed"))
    else:
        synth = synthesize_raw(text, profile["base_speaker"], raw_path, speed=profile.get("speed"))
    synth_sec = time.perf_counter() - synth_start
    effects_start = time.perf_counter()
    final_path, effect_warnings = apply_profile_effects(raw_path, final_path, profile)
    effects_sec = time.perf_counter() - effects_start
    analysis: Optional[Dict[str, Any]] = None
    analysis_warnings: List[str] = []
    analysis_sec = 0.0
    if analyze:
        analysis_start = time.perf_counter()
        analysis = analyze_wav(final_path)
        analysis_sec = time.perf_counter() - analysis_start
        analysis_warnings = list(analysis.get("warnings") or [])
    warnings = [*effect_warnings, *analysis_warnings]
    if analysis and analysis.get("clipping_detected"):
        warnings.append("clipping terdeteksi pada output final.")
    metadata = {
        "text": text,
        "profile_name": profile_name,
        "config": profile,
        "base_speaker": profile["base_speaker"],
        "speaker_id": synth.speaker_id,
        "raw_wav_path": str(raw_path),
        "final_wav_path": str(final_path),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "text_for_tts": synth.text_for_tts,
        "audio_analysis": analysis,
        "warnings": warnings,
        "timing_sec": {
            "synthesis": round(synth_sec, 3),
            "effects": round(effects_sec, 3),
            "analysis": round(analysis_sec, 3),
            "total": round(time.perf_counter() - total_start, 3),
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return GenerationResult(raw_path, final_path, metadata_path, metadata)


class InProcessTTSRuntime:
    def __init__(self, device: str = "auto"):
        self.device = device
        self._synthesizer: Any = None
        self._loaded_device = ""

    @property
    def is_loaded(self) -> bool:
        return self._synthesizer is not None

    def load(self) -> None:
        if self._synthesizer is not None:
            return
        try:
            paths = ensure_model_available()
            config_path = _prepare_runtime_config(paths)
            from TTS.utils.synthesizer import Synthesizer
        except Exception as exc:
            raise TTSEngineError(f"Gagal load in-process TTS runtime: {exc}") from exc
        resolved_device = _resolve_device(self.device)
        use_cuda = resolved_device == "cuda"
        try:
            self._synthesizer = Synthesizer(
                str(paths["checkpoint_1260000-inference.pth"]),
                str(config_path),
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                None,
                use_cuda=use_cuda,
            )
            self._loaded_device = resolved_device
        except Exception as exc:
            raise TTSEngineError(f"Gagal memuat model TTS ke device {resolved_device}: {exc}") from exc

    def warmup(self, output_dir: Path) -> float:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        start = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix="tts_warmup_", dir=output_dir) as tmpdir:
            self.synthesize_raw("Halo.", "Gadis", Path(tmpdir) / "warmup.wav")
        return time.perf_counter() - start

    def synthesize_raw(self, text: str, base_speaker: str, output_path: Path, speed: Optional[float] = None) -> SynthesisResult:
        if base_speaker not in SPEAKER_TO_ID:
            raise TTSEngineError("base_speaker hanya mendukung Wibowo atau Gadis.")
        if not text or not text.strip():
            raise TTSEngineError("Teks TTS tidak boleh kosong.")
        self.load()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        text_for_tts = _normalize_and_g2p(text)
        try:
            wav = self._synthesizer.tts(
                text_for_tts,
                speaker_name=SPEAKER_TO_ID[base_speaker],
                split_sentences=False,
            )
            self._synthesizer.save_wav(wav, str(output_path))
        except Exception as exc:
            raise TTSEngineError(f"Gagal menjalankan in-process TTS runtime: {exc}") from exc
        if not output_path.exists() or output_path.stat().st_size < 1000:
            raise TTSEngineError(f"TTS tidak menghasilkan WAV valid: {display_path(output_path)}")
        return SynthesisResult(output_path, text_for_tts, SPEAKER_TO_ID[base_speaker])


def _resolve_device(device: str) -> str:
    if device not in {"auto", "cpu", "cuda"}:
        raise TTSEngineError("device TTS hanya boleh auto, cpu, atau cuda.")
    if device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def synthesize_raw(text: str, base_speaker: str, output_path: Path, speed: Optional[float] = None) -> SynthesisResult:
    if base_speaker not in SPEAKER_TO_ID:
        raise TTSEngineError("base_speaker hanya mendukung Wibowo atau Gadis.")
    if not text or not text.strip():
        raise TTSEngineError("Teks TTS tidak boleh kosong.")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        paths = ensure_model_available()
    except ModelDownloadError:
        raise
    except Exception as exc:
        raise TTSEngineError(f"Gagal menyiapkan model: {exc}") from exc

    text_for_tts = _normalize_and_g2p(text)
    config_path = _prepare_runtime_config(paths)
    bin_tts = _find_tts_binary()
    cmd = [
        str(bin_tts),
        "--text",
        text_for_tts,
        "--model_path",
        str(paths["checkpoint_1260000-inference.pth"]),
        "--config_path",
        str(config_path),
        "--speaker_idx",
        SPEAKER_TO_ID[base_speaker],
        "--out_path",
        str(output_path),
    ]
    if speed is not None:
        # Coqui's generic CLI accepts many model-specific options inconsistently.
        # Speed is primarily handled by post-processing; this value is kept by callers in metadata.
        pass
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise TTSEngineError(
            "Gagal menjalankan Coqui TTS.\n"
            f"Command: {' '.join(cmd)}\n"
            f"STDERR:\n{result.stderr.strip()}\n"
            f"STDOUT:\n{result.stdout.strip()}"
        )
    if not output_path.exists() or output_path.stat().st_size < 1000:
        raise TTSEngineError(f"TTS tidak menghasilkan WAV valid: {display_path(output_path)}")
    return SynthesisResult(output_path, text_for_tts, SPEAKER_TO_ID[base_speaker])


def _find_tts_binary() -> Path:
    candidates = [
        Path(sys.prefix) / "bin" / "tts",
        Path(sys.executable).parent / "tts",
    ]
    found = shutil.which("tts")
    if found:
        candidates.append(Path(found))
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    raise TTSEngineError(
        "Binary Coqui `tts` tidak ditemukan. Jalankan:\n"
        "source env_bisindo_cuda126/bin/activate\n"
        "pip install -r src_test_tts/requirements.txt"
    )


def _normalize_and_g2p(text: str) -> str:
    try:
        from g2p_id.scripts.tts import g2p, text_normalization
    except Exception as exc:
        raise TTSEngineError(
            "Dependency TTS-Indonesia-Gratis/g2p_id belum tersedia. Jalankan:\n"
            "source env_bisindo_cuda126/bin/activate\n"
            "pip install -r src_test_tts/requirements.txt"
        ) from exc
    normalized = text_normalization(unescape(text))
    return g2p(normalized)


def _prepare_runtime_config(paths: Dict[str, Path]) -> Path:
    base_config = paths["config.json"]
    speakers = paths["speakers.pth"]
    runtime_config = MODEL_DIR / "runtime_config.json"
    try:
        data = json.loads(base_config.read_text(encoding="utf-8"))
    except Exception as exc:
        raise TTSEngineError(f"config.json tidak bisa dibaca: {base_config}") from exc
    model_args = data.setdefault("model_args", {})
    model_args["speakers_file"] = str(speakers.resolve())
    runtime_config.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return runtime_config
