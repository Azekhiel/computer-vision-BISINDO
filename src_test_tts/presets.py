from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

from paths import PROFILE_PATH, ensure_base_dirs


DEFAULT_PRESETS: Dict[str, Dict[str, Any]] = {
    "cowok_dewasa_default": {
        "gender": "cowok",
        "demografi": "dewasa",
        "variasi_conf": "default",
        "base_speaker": "Wibowo",
        "pitch_semitones": 0.0,
        "speed": 1.00,
        "volume_gain_db": 0.0,
        "formant_shift": 0.0,
        "highpass_hz": 70,
        "lowpass_hz": 9000,
        "normalize": True,
        "compressor": False,
        "output_sample_rate": 22050,
        "notes": "Baseline asli Wibowo untuk cowok dewasa. Jangan dibuat terlalu berbeda dari suara asli.",
    },
    "cewek_dewasa_default": {
        "gender": "cewek",
        "demografi": "dewasa",
        "variasi_conf": "default",
        "base_speaker": "Gadis",
        "pitch_semitones": 0.0,
        "speed": 1.00,
        "volume_gain_db": 0.0,
        "formant_shift": 0.0,
        "highpass_hz": 90,
        "lowpass_hz": 9500,
        "normalize": True,
        "compressor": False,
        "output_sample_rate": 22050,
        "notes": "Baseline asli Gadis untuk cewek dewasa. Jangan dibuat terlalu berbeda dari suara asli.",
    },
    "cowok_remaja_default": {
        "gender": "cowok",
        "demografi": "remaja",
        "variasi_conf": "default",
        "base_speaker": "Wibowo",
        "pitch_semitones": 1.3,
        "speed": 1.04,
        "volume_gain_db": 0.0,
        "formant_shift": 0.05,
        "highpass_hz": 90,
        "lowpass_hz": 9500,
        "normalize": True,
        "compressor": False,
        "output_sample_rate": 22050,
        "notes": "Remaja cowok dari baseline Wibowo. Jangan sampai chipmunk.",
    },
    "cewek_remaja_default": {
        "gender": "cewek",
        "demografi": "remaja",
        "variasi_conf": "default",
        "base_speaker": "Gadis",
        "pitch_semitones": 0.8,
        "speed": 1.04,
        "volume_gain_db": 0.0,
        "formant_shift": 0.04,
        "highpass_hz": 110,
        "lowpass_hz": 10000,
        "normalize": True,
        "compressor": False,
        "output_sample_rate": 22050,
        "notes": "Remaja cewek dari baseline Gadis. Karena Gadis sudah cewek, pitch jangan dinaikkan terlalu ekstrem.",
    },
    "cowok_anak_anak_default": {
        "gender": "cowok",
        "demografi": "anak_anak",
        "variasi_conf": "default",
        "base_speaker": "Wibowo",
        "pitch_semitones": 3.0,
        "speed": 1.08,
        "volume_gain_db": 0.0,
        "formant_shift": 0.13,
        "highpass_hz": 130,
        "lowpass_hz": 10500,
        "normalize": True,
        "compressor": True,
        "output_sample_rate": 22050,
        "notes": "Anak-anak cowok dari baseline Wibowo. Ini paling rawan terdengar tidak natural, jadi harus ada preview dan tuning manual.",
    },
    "cewek_anak_anak_default": {
        "gender": "cewek",
        "demografi": "anak_anak",
        "variasi_conf": "default",
        "base_speaker": "Gadis",
        "pitch_semitones": 2.2,
        "speed": 1.08,
        "volume_gain_db": 0.0,
        "formant_shift": 0.10,
        "highpass_hz": 140,
        "lowpass_hz": 11000,
        "normalize": True,
        "compressor": True,
        "output_sample_rate": 22050,
        "notes": "Anak-anak cewek dari baseline Gadis. Jangan pitch terlalu tinggi supaya tidak jadi chipmunk.",
    },
}


def get_default_presets() -> Dict[str, Dict[str, Any]]:
    return deepcopy(DEFAULT_PRESETS)


def init_default_profiles(path: Path = PROFILE_PATH, overwrite: bool = False) -> bool:
    ensure_base_dirs()
    if path.exists() and not overwrite:
        return False
    path.write_text(_json_dumps(DEFAULT_PRESETS), encoding="utf-8")
    return True


def _json_dumps(data: Dict[str, Dict[str, Any]]) -> str:
    import json

    return json.dumps(data, indent=2, ensure_ascii=False) + "\n"

