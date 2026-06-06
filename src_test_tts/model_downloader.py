import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import requests
from tqdm import tqdm

from paths import MODEL_DIR, display_path, ensure_base_dirs


WIKIDEPIA_RELEASE_API = "https://api.github.com/repos/Wikidepia/indonesian-tts/releases/latest"
DRAT_RAW_BASE = "https://raw.githubusercontent.com/drat/TTS-Indonesia-Gratis/main"
MANUAL_MODEL_URL_ENV = "TTS_INDONESIA_GRATIS_MODEL_URL"


@dataclass(frozen=True)
class ModelFileSpec:
    name: str
    min_size: int
    urls: tuple[str, ...]
    required: bool = True


MODEL_FILES: tuple[ModelFileSpec, ...] = (
    ModelFileSpec(
        "checkpoint_1260000-inference.pth",
        100_000_000,
        (
            "https://github.com/Wikidepia/indonesian-tts/releases/download/v1.2/checkpoint_1260000-inference.pth",
        ),
    ),
    ModelFileSpec(
        "config.json",
        1_000,
        (
            "https://github.com/Wikidepia/indonesian-tts/releases/download/v1.2/config.json",
            f"{DRAT_RAW_BASE}/g2p_id/data/config.json",
            f"{DRAT_RAW_BASE}/config.json",
        ),
    ),
    ModelFileSpec(
        "speakers.pth",
        1_000,
        (
            "https://github.com/Wikidepia/indonesian-tts/releases/download/v1.2/speakers.pth",
            f"{DRAT_RAW_BASE}/g2p_id/data/speakers.pth",
            f"{DRAT_RAW_BASE}/speakers.pth",
        ),
    ),
    ModelFileSpec(
        "languages.json",
        30,
        (
            f"{DRAT_RAW_BASE}/g2p_id/data/languages.json",
            f"{DRAT_RAW_BASE}/languages.json",
        ),
        required=False,
    ),
)


class ModelDownloadError(RuntimeError):
    pass


def get_model_paths(model_dir: Path = MODEL_DIR) -> Dict[str, Path]:
    return {spec.name: model_dir / spec.name for spec in MODEL_FILES}


def verify_model_file(path: Path, min_size: int) -> bool:
    return path.exists() and path.is_file() and path.stat().st_size >= min_size


def ensure_model_available(model_dir: Path = MODEL_DIR, download: bool = True) -> Dict[str, Path]:
    ensure_base_dirs()
    model_dir.mkdir(parents=True, exist_ok=True)
    missing = [spec for spec in MODEL_FILES if spec.required and not verify_model_file(model_dir / spec.name, spec.min_size)]
    optional_missing = [spec for spec in MODEL_FILES if not spec.required and not verify_model_file(model_dir / spec.name, spec.min_size)]
    if missing or optional_missing:
        if not download:
            _raise_missing(model_dir, missing)
        download_model_if_missing(model_dir=model_dir, specs=[*missing, *optional_missing])
    missing_after = [spec for spec in MODEL_FILES if spec.required and not verify_model_file(model_dir / spec.name, spec.min_size)]
    if missing_after:
        _raise_missing(model_dir, missing_after)
    return get_model_paths(model_dir)


def download_model_if_missing(model_dir: Path = MODEL_DIR, specs: Optional[Iterable[ModelFileSpec]] = None) -> Dict[str, Path]:
    ensure_base_dirs()
    model_dir.mkdir(parents=True, exist_ok=True)
    selected_specs = list(specs if specs is not None else MODEL_FILES)
    release_urls = _release_asset_urls()
    failures: List[str] = []
    for spec in selected_specs:
        target = model_dir / spec.name
        if verify_model_file(target, spec.min_size):
            print(f"[OK] {display_path(target)} sudah valid ({target.stat().st_size} bytes)")
            continue
        if target.exists():
            print(f"[WARN] {display_path(target)} terlalu kecil/corrupt, hapus lalu download ulang.")
            target.unlink()
        urls = _candidate_urls(spec, release_urls)
        downloaded = False
        for url in urls:
            try:
                _download_url(url, target, spec.min_size)
                downloaded = True
                print(f"[OK] Simpan {display_path(target)}")
                break
            except Exception as exc:  # noqa: BLE001 - keep trying candidate URLs.
                failures.append(f"{spec.name} dari {url}: {exc}")
                if target.exists():
                    target.unlink()
        if not downloaded and spec.required:
            print_manual_download_instructions(model_dir)
            detail = "\n".join(failures[-4:])
            raise ModelDownloadError(f"Gagal download file wajib {spec.name}.\n{detail}")
        if not downloaded:
            print(f"[WARN] File opsional {spec.name} belum tersedia.")
    return get_model_paths(model_dir)


def print_manual_download_instructions(model_dir: Path = MODEL_DIR) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    print()
    print("Instruksi manual download model TTS Indonesia:")
    print(f"1. Buka https://github.com/Wikidepia/indonesian-tts/releases/tag/v1.2")
    print("2. Download file berikut:")
    for spec in MODEL_FILES:
        label = "wajib" if spec.required else "opsional"
        print(f"   - {spec.name} ({label})")
    print(f"3. Letakkan semua file di: {model_dir.resolve()}")
    print("4. Jalankan ulang: python src_test_tts/cli.py download-model")
    print()
    print("Alternatif untuk checkpoint utama:")
    print(f"export {MANUAL_MODEL_URL_ENV}=https://alamat-manual/checkpoint_1260000-inference.pth")
    print("python src_test_tts/cli.py download-model")
    print()


def _raise_missing(model_dir: Path, missing: Iterable[ModelFileSpec]) -> None:
    names = ", ".join(spec.name for spec in missing)
    print_manual_download_instructions(model_dir)
    raise ModelDownloadError(f"Model/checkpoint belum tersedia: {names}")


def _candidate_urls(spec: ModelFileSpec, release_urls: Dict[str, str]) -> List[str]:
    urls: List[str] = []
    if spec.name == "checkpoint_1260000-inference.pth":
        env_url = os.environ.get(MANUAL_MODEL_URL_ENV, "").strip()
        if env_url:
            urls.append(env_url)
    if spec.name in release_urls:
        urls.append(release_urls[spec.name])
    urls.extend(spec.urls)
    deduped: List[str] = []
    for url in urls:
        if url and url not in deduped:
            deduped.append(url)
    return deduped


def _release_asset_urls() -> Dict[str, str]:
    try:
        response = requests.get(WIKIDEPIA_RELEASE_API, timeout=15)
        response.raise_for_status()
        payload = response.json()
    except Exception:
        return {}
    assets = payload.get("assets") or []
    urls: Dict[str, str] = {}
    for asset in assets:
        name = asset.get("name")
        url = asset.get("browser_download_url")
        if isinstance(name, str) and isinstance(url, str):
            urls[name] = url
    return urls


def _download_url(url: str, target: Path, min_size: int) -> None:
    tmp = target.with_suffix(target.suffix + ".tmp")
    headers = {"User-Agent": "bisindo-src-test-tts/1.0"}
    with requests.get(url, stream=True, timeout=30, headers=headers) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length") or 0)
        with tmp.open("wb") as handle:
            with tqdm(total=total or None, unit="B", unit_scale=True, desc=target.name) as progress:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    handle.write(chunk)
                    progress.update(len(chunk))
    if not verify_model_file(tmp, min_size):
        size = tmp.stat().st_size if tmp.exists() else 0
        tmp.unlink(missing_ok=True)
        raise ModelDownloadError(f"File hasil download terlalu kecil ({size} bytes).")
    shutil.move(str(tmp), str(target))


def model_status(model_dir: Path = MODEL_DIR) -> List[Dict[str, object]]:
    paths = get_model_paths(model_dir)
    rows: List[Dict[str, object]] = []
    for spec in MODEL_FILES:
        path = paths[spec.name]
        exists = path.exists()
        size = path.stat().st_size if exists else 0
        rows.append(
            {
                "name": spec.name,
                "path": str(path),
                "required": spec.required,
                "exists": exists,
                "size": size,
                "valid": verify_model_file(path, spec.min_size),
                "min_size": spec.min_size,
            }
        )
    return rows


def write_model_status_json(path: Path, model_dir: Path = MODEL_DIR) -> None:
    path.write_text(json.dumps(model_status(model_dir), indent=2), encoding="utf-8")

