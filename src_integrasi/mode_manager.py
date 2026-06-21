"""Mode aktif sistem (dari topic MODE) dan pemetaannya ke voice TTS.

MODE menentukan mode aktif alat:
- STT  : speech-to-text (default nyala). Sistem tetap kirim text.
- STS_*: sign-to-speech dengan voice tertentu.

Pemetaan varian STS (A=dewasa, T=remaja; M=cowok, F=cewek):
- STS_AM = cowok dewasa
- STS_AF = cewek dewasa
- STS_TM = cowok remaja
- STS_TF = cewek remaja

Profil TTS mengikuti penamaan {gender}_{demografi}_default yang sudah ada di
src_test_tts/configs/voice_profiles.json (lihat build_tts_profile_name).
"""

from __future__ import annotations

DEFAULT_MODE = "STT"

# Voice default saat mode STT (GAS tetap bisa menyuarakan dengan voice ini).
DEFAULT_TTS_PROFILE = "cewek_dewasa_default"

MODE_TO_TTS_PROFILE = {
    "STT": DEFAULT_TTS_PROFILE,
    "STS_AM": "cowok_dewasa_default",
    "STS_AF": "cewek_dewasa_default",
    "STS_TM": "cowok_remaja_default",
    "STS_TF": "cewek_remaja_default",
}

VALID_MODES = tuple(MODE_TO_TTS_PROFILE.keys())


def is_valid_mode(mode: str) -> bool:
    return str(mode or "").strip() in MODE_TO_TTS_PROFILE


def is_sts(mode: str) -> bool:
    return str(mode or "").strip().startswith("STS_")


def resolve(mode: str) -> tuple[bool, str]:
    """Kembalikan (is_sts, tts_profile) untuk mode tertentu.

    Mode tidak valid -> jatuh ke default (STT + voice default).
    """
    key = str(mode or "").strip()
    profile = MODE_TO_TTS_PROFILE.get(key, DEFAULT_TTS_PROFILE)
    return is_sts(key), profile
