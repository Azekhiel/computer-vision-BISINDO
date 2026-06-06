from pathlib import Path


MODULE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_DIR.parent
CONFIG_DIR = MODULE_DIR / "configs"
MODEL_DIR = MODULE_DIR / "models" / "tts_indonesia_gratis"
OUTPUT_DIR = MODULE_DIR / "outputs"
SAMPLES_DIR = MODULE_DIR / "samples"
PROFILE_PATH = CONFIG_DIR / "voice_profiles.json"


def ensure_base_dirs() -> None:
    for path in (CONFIG_DIR, MODEL_DIR, OUTPUT_DIR, SAMPLES_DIR):
        path.mkdir(parents=True, exist_ok=True)


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path.resolve())

