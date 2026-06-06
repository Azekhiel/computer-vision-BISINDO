import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from paths import PROFILE_PATH, ensure_base_dirs
from presets import DEFAULT_PRESETS


VALID_GENDERS = {"cowok", "cewek"}
VALID_DEMOGRAFI = {"dewasa", "remaja", "anak_anak"}
VALID_SPEAKERS = {"Wibowo", "Gadis"}
VARIASI_RE = re.compile(r"^[a-z0-9]+(?:_[a-z0-9]+)*$")
GENDER_DISPLAY_TO_VALUE = {"Cewek": "cewek", "Cowok": "cowok"}
DEMOGRAFI_DISPLAY_TO_VALUE = {"Dewasa": "dewasa", "Remaja": "remaja", "Anak-Anak": "anak_anak"}
GENDER_VALUE_TO_DISPLAY = {value: label for label, value in GENDER_DISPLAY_TO_VALUE.items()}
DEMOGRAFI_VALUE_TO_DISPLAY = {value: label for label, value in DEMOGRAFI_DISPLAY_TO_VALUE.items()}
GENDER_DISPLAY_OPTIONS = tuple(GENDER_DISPLAY_TO_VALUE.keys())
DEMOGRAFI_DISPLAY_OPTIONS = tuple(DEMOGRAFI_DISPLAY_TO_VALUE.keys())


class ProfileError(ValueError):
    pass


def build_profile_name(gender: str, demografi: str, variasi_conf: str) -> str:
    return f"{gender}_{demografi}_{variasi_conf}"


def normalize_gender_choice(value: str) -> str:
    raw = str(value or "").strip()
    return GENDER_DISPLAY_TO_VALUE.get(raw, raw)


def normalize_demografi_choice(value: str) -> str:
    raw = str(value or "").strip()
    return DEMOGRAFI_DISPLAY_TO_VALUE.get(raw, raw)


def display_gender(value: str) -> str:
    raw = normalize_gender_choice(value)
    return GENDER_VALUE_TO_DISPLAY.get(raw, raw)


def display_demografi(value: str) -> str:
    raw = normalize_demografi_choice(value)
    return DEMOGRAFI_VALUE_TO_DISPLAY.get(raw, raw)


def parse_profile_name(name: str) -> Dict[str, str]:
    if " " in name or not name:
        raise ProfileError("Nama profile tidak boleh kosong atau memakai spasi.")
    for gender in VALID_GENDERS:
        prefix = f"{gender}_"
        if not name.startswith(prefix):
            continue
        rest = name[len(prefix):]
        for demografi in sorted(VALID_DEMOGRAFI, key=len, reverse=True):
            dem_prefix = f"{demografi}_"
            if rest.startswith(dem_prefix):
                variasi_conf = rest[len(dem_prefix):]
                if not variasi_conf:
                    raise ProfileError("variasi_conf tidak boleh kosong.")
                if not VARIASI_RE.match(variasi_conf):
                    raise ProfileError("variasi_conf wajib snake_case huruf kecil/angka.")
                return {
                    "gender": gender,
                    "demografi": demografi,
                    "variasi_conf": variasi_conf,
                }
    raise ProfileError(
        "Format nama profile wajib {gender}_{demografi}_{variasi_conf}; "
        "gender: cowok/cewek, demografi: dewasa/remaja/anak_anak."
    )


def profile_names_for(names: Iterable[str], gender: str, demografi: str) -> List[str]:
    target_gender = normalize_gender_choice(gender)
    target_demografi = normalize_demografi_choice(demografi)
    selected: List[str] = []
    for name in names:
        try:
            parsed = parse_profile_name(name)
        except ProfileError:
            continue
        if parsed["gender"] == target_gender and parsed["demografi"] == target_demografi:
            selected.append(name)
    return sorted(selected, key=_profile_picker_sort_key)


def preferred_profile_name(names: Iterable[str], gender: str, demografi: str) -> Optional[str]:
    selected = profile_names_for(names, gender, demografi)
    return selected[0] if selected else None


def default_profile_name(gender: str, demografi: str) -> str:
    return build_profile_name(normalize_gender_choice(gender), normalize_demografi_choice(demografi), "default")


def _profile_picker_sort_key(name: str) -> tuple[int, str]:
    try:
        parsed = parse_profile_name(name)
    except ProfileError:
        return (1, name)
    variation = parsed["variasi_conf"]
    return (0 if variation == "default" else 1, variation)


def validate_profile_name(name: str) -> None:
    parse_profile_name(name)


def validate_profile(profile_name: str, profile: Dict[str, Any]) -> None:
    parsed = parse_profile_name(profile_name)
    for key, value in parsed.items():
        if profile.get(key) != value:
            raise ProfileError(f"Field {key} harus sama dengan nama profile: {value!r}.")
    speaker = profile.get("base_speaker")
    if speaker not in VALID_SPEAKERS:
        raise ProfileError("base_speaker hanya boleh Wibowo atau Gadis.")
    if parsed["gender"] == "cowok" and speaker != "Wibowo":
        raise ProfileError("Profile cowok wajib memakai base_speaker Wibowo.")
    if parsed["gender"] == "cewek" and speaker != "Gadis":
        raise ProfileError("Profile cewek wajib memakai base_speaker Gadis.")
    numeric_fields = {
        "pitch_semitones",
        "speed",
        "volume_gain_db",
        "formant_shift",
        "highpass_hz",
        "lowpass_hz",
        "output_sample_rate",
    }
    for field in numeric_fields:
        if field not in profile:
            raise ProfileError(f"Field wajib hilang: {field}")
        if not isinstance(profile[field], (int, float)):
            raise ProfileError(f"Field {field} wajib angka.")
    if profile["speed"] <= 0:
        raise ProfileError("speed harus lebih besar dari 0.")
    if profile["output_sample_rate"] < 8000:
        raise ProfileError("output_sample_rate terlalu rendah.")
    for field in ("normalize", "compressor"):
        if not isinstance(profile.get(field), bool):
            raise ProfileError(f"Field {field} wajib boolean.")


class VoiceProfileManager:
    def __init__(self, path: Path = PROFILE_PATH):
        self.path = path
        ensure_base_dirs()

    def init_defaults(self, overwrite: bool = False) -> bool:
        if self.path.exists() and not overwrite:
            return False
        self._write(DEFAULT_PRESETS)
        return True

    def load(self) -> Dict[str, Dict[str, Any]]:
        if not self.path.exists():
            self.init_defaults(overwrite=False)
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ProfileError(f"Config JSON rusak: {self.path} ({exc})") from exc
        if not isinstance(data, dict):
            raise ProfileError("voice_profiles.json wajib berisi object JSON.")
        for name, profile in data.items():
            if not isinstance(profile, dict):
                raise ProfileError(f"Profile {name} wajib object JSON.")
            validate_profile(name, profile)
        return data

    def list_names(self) -> List[str]:
        return sorted(self.load().keys())

    def get(self, name: str) -> Dict[str, Any]:
        data = self.load()
        if name not in data:
            raise ProfileError(f"Profile tidak ditemukan: {name}")
        return deepcopy(data[name])

    def save(self, name: str, profile: Dict[str, Any], overwrite: bool = False) -> None:
        validate_profile(name, profile)
        data = self.load()
        if name in data and not overwrite:
            raise ProfileError(f"Profile sudah ada: {name}. Pakai --overwrite untuk menimpa.")
        data[name] = deepcopy(profile)
        self._write(data)

    def update(self, name: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        data = self.load()
        if name not in data:
            raise ProfileError(f"Profile tidak ditemukan: {name}")
        profile = deepcopy(data[name])
        profile.update({k: v for k, v in updates.items() if v is not None})
        validate_profile(name, profile)
        data[name] = profile
        self._write(data)
        return deepcopy(profile)

    def delete(self, name: str) -> None:
        data = self.load()
        if name not in data:
            raise ProfileError(f"Profile tidak ditemukan: {name}")
        del data[name]
        self._write(data)

    def merge_defaults(self) -> List[str]:
        data = self.load()
        added: List[str] = []
        for name, profile in DEFAULT_PRESETS.items():
            if name not in data:
                data[name] = deepcopy(profile)
                added.append(name)
        if added:
            self._write(data)
        return added

    def _write(self, data: Dict[str, Dict[str, Any]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
        self.path.write_text(text, encoding="utf-8")


def make_profile(
    gender: str,
    demografi: str,
    variasi_conf: str,
    base_speaker: str,
    pitch: float = 0.0,
    speed: float = 1.0,
    volume: float = 0.0,
    formant: float = 0.0,
    highpass_hz: int = 80,
    lowpass_hz: int = 9500,
    normalize: bool = True,
    compressor: bool = False,
    output_sample_rate: int = 22050,
    notes: str = "",
) -> Dict[str, Any]:
    name = build_profile_name(gender, demografi, variasi_conf)
    profile = {
        "gender": gender,
        "demografi": demografi,
        "variasi_conf": variasi_conf,
        "base_speaker": base_speaker,
        "pitch_semitones": float(pitch),
        "speed": float(speed),
        "volume_gain_db": float(volume),
        "formant_shift": float(formant),
        "highpass_hz": int(highpass_hz),
        "lowpass_hz": int(lowpass_hz),
        "normalize": bool(normalize),
        "compressor": bool(compressor),
        "output_sample_rate": int(output_sample_rate),
        "notes": notes,
    }
    validate_profile(name, profile)
    return profile


def filter_update_args(items: Iterable[tuple[str, Optional[Any]]]) -> Dict[str, Any]:
    return {key: value for key, value in items if value is not None}
