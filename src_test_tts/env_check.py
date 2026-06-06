import os
import sys
from pathlib import Path


DEFAULT_EXPECTED_VENV = "env_bisindo_cuda126"
WARNING_TEXT = (
    "WARNING: Anda tidak sedang memakai venv env_bisindo_cuda126. "
    "Aktifkan dulu dengan: source env_bisindo_cuda126/bin/activate"
)


def _venv_name_from_runtime() -> str:
    virtual_env = os.environ.get("VIRTUAL_ENV")
    if virtual_env:
        return Path(virtual_env).name
    prefix = Path(sys.prefix).resolve()
    base_prefix = Path(getattr(sys, "base_prefix", sys.prefix)).resolve()
    if prefix != base_prefix:
        return prefix.name
    return ""


def is_expected_venv(expected_name: str = DEFAULT_EXPECTED_VENV) -> bool:
    return _venv_name_from_runtime() == expected_name


def warn_if_wrong_venv(expected_name: str = DEFAULT_EXPECTED_VENV) -> bool:
    ok = is_expected_venv(expected_name)
    if not ok:
        print(WARNING_TEXT, file=sys.stderr)
    return ok


def require_expected_venv(expected_name: str = DEFAULT_EXPECTED_VENV) -> None:
    if not is_expected_venv(expected_name):
        raise RuntimeError(WARNING_TEXT)

