"""Jetson CUDA 12.6 diagnostics and live-device policy for GRU runtime."""

from __future__ import annotations

import importlib
import os
import platform
import sys
from pathlib import Path
from typing import Any


EXPECTED_JETSON_CUDA = "12.6"
CUDA126_ENV_NAME = "env_bisindo_cuda126"


def _import_torch(torch_module: Any | None = None) -> tuple[Any | None, str]:
    if torch_module is not None:
        return torch_module, ""
    try:
        import torch  # type: ignore

        return torch, ""
    except Exception as exc:
        return None, str(exc)


def _normalize_variant(variant: str) -> str:
    value = str(variant or "adi").strip().lower()
    if value.startswith("gru_"):
        value = value[4:]
    if value in {"auto", "best"}:
        return "adi"
    return value


def is_jetson_platform() -> bool:
    if os.environ.get("BISINDO_FORCE_JETSON") == "1":
        return True
    markers = (
        Path("/etc/nv_tegra_release"),
        Path("/sys/module/tegra_fuse"),
        Path("/usr/lib/aarch64-linux-gnu/tegra"),
    )
    if any(path.exists() for path in markers):
        return True
    machine = platform.machine().lower()
    release = platform.uname().release.lower()
    return machine in {"aarch64", "arm64"} and "tegra" in release


def torch_cuda_version(torch_module: Any | None = None) -> str:
    torch, _ = _import_torch(torch_module)
    if torch is None:
        return ""
    value = getattr(getattr(torch, "version", None), "cuda", "")
    if value is None:
        return ""
    return str(value)


def torch_cuda_available(torch_module: Any | None = None) -> bool:
    torch, _ = _import_torch(torch_module)
    if torch is None:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def cuda_version_matches(torch_module: Any | None = None, expected: str = EXPECTED_JETSON_CUDA) -> bool:
    version = torch_cuda_version(torch_module)
    return not version or version.startswith(str(expected))


def cuda_guard_message(torch_module: Any | None = None) -> str:
    version = torch_cuda_version(torch_module) or "CPU-only/unknown"
    executable = sys.executable
    return (
        f"PyTorch CUDA aktif adalah {version}, sementara JetPack 6.2.2 butuh CUDA {EXPECTED_JETSON_CUDA}. "
        f"Pakai {CUDA126_ENV_NAME}: ./scripts/setup_jetson_cuda126.sh lalu jalankan "
        f"./scripts/run_live_jetson_gru.sh. Python sekarang: {executable}"
    )


def validate_cuda126_for_jetson(
    torch_module: Any | None = None,
    require_cuda: bool = False,
) -> tuple[bool, str]:
    torch, import_error = _import_torch(torch_module)
    if torch is None:
        return False, f"PyTorch tidak bisa diimport: {import_error}"

    cuda_version = torch_cuda_version(torch)
    if cuda_version and not cuda_version.startswith(EXPECTED_JETSON_CUDA):
        return False, cuda_guard_message(torch)

    cuda_available = torch_cuda_available(torch)
    if require_cuda and not cuda_available:
        return False, (
            f"CUDA tidak tersedia dari PyTorch ini. Pastikan menjalankan {CUDA126_ENV_NAME} "
            f"dan torch CUDA {EXPECTED_JETSON_CUDA}."
        )

    if cuda_version:
        return True, f"torch CUDA {cuda_version}, cuda_available={cuda_available}"
    return True, "PyTorch CPU-only; CUDA tidak diminta."


def select_live_device(
    variant: str,
    requested: str = "auto",
    torch_module: Any | None = None,
) -> tuple[str, str]:
    """Pick the fastest safe live device for a GRU variant on Jetson.

    The policy is based on local Jetson benchmark observations:
    Adi is more accurate but slower on CUDA because its custom ReLU-GRU loops
    in Python; Khukuh benefits from CUDA; Hybrid defaults CPU until trained and
    benchmarked.
    """

    requested = str(requested or "auto").strip().lower()
    variant_name = _normalize_variant(variant)

    if requested == "cpu":
        ok, msg = validate_cuda126_for_jetson(torch_module=torch_module, require_cuda=False)
        if ok:
            return "cpu", "requested_cpu"
        return "cpu", f"requested_cpu; warning: {msg}"

    if requested == "cuda":
        ok, msg = validate_cuda126_for_jetson(torch_module=torch_module, require_cuda=True)
        if not ok:
            raise RuntimeError(msg)
        return "cuda", "requested_cuda_valid_cuda126"

    if requested != "auto":
        raise ValueError("device harus salah satu: auto, cpu, cuda")

    ok, msg = validate_cuda126_for_jetson(torch_module=torch_module, require_cuda=False)
    if not ok:
        raise RuntimeError(msg)

    if variant_name == "adi":
        return "cpu", "policy_adi_cpu_faster_than_cuda_on_jetson"

    if variant_name == "khukuh":
        if torch_cuda_available(torch_module):
            return "cuda", "policy_khukuh_cuda_faster_on_jetson"
        return "cpu", f"policy_khukuh_cpu_fallback; {msg}"

    if variant_name == "hybrid":
        return "cpu", "policy_hybrid_cpu_until_local_benchmark_prefers_cuda"

    return "cpu", f"policy_unknown_variant_cpu_fallback:{variant_name}"


def _tensorrt_status() -> dict[str, Any]:
    try:
        trt = importlib.import_module("tensorrt")
        return {
            "ok": True,
            "version": str(getattr(trt, "__version__", "")),
            "error": "",
        }
    except Exception as exc:
        return {"ok": False, "version": "", "error": str(exc)}


def _opencv_gstreamer_status(cv2_module: Any | None) -> str:
    if cv2_module is None:
        try:
            cv2_module = importlib.import_module("cv2")
        except Exception:
            return "unknown"
    try:
        build = cv2_module.getBuildInformation()
    except Exception:
        return "unknown"
    for line in str(build).splitlines():
        if "GStreamer:" in line:
            return line.split("GStreamer:", 1)[1].strip()
    return "unknown"


def diagnostics(torch_module: Any | None = None, cv2_module: Any | None = None) -> dict[str, Any]:
    torch, import_error = _import_torch(torch_module)
    trt = _tensorrt_status()
    diag: dict[str, Any] = {
        "python": sys.executable,
        "sys_prefix": sys.prefix,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "is_jetson": is_jetson_platform(),
        "cuda_home": os.environ.get("CUDA_HOME", ""),
        "torch_import_ok": torch is not None,
        "torch_import_error": import_error,
        "torch_version": "",
        "torch_cuda": "",
        "cuda_available": False,
        "device_name": "",
        "device_capability": "",
        "tensorrt_import_ok": bool(trt["ok"]),
        "tensorrt_version": trt["version"],
        "tensorrt_error": trt["error"],
        "opencv_version": "",
        "opencv_gstreamer": _opencv_gstreamer_status(cv2_module),
        "warning": "",
    }

    if cv2_module is None:
        try:
            cv2_module = importlib.import_module("cv2")
        except Exception:
            cv2_module = None
    if cv2_module is not None:
        diag["opencv_version"] = str(getattr(cv2_module, "__version__", ""))

    if torch is None:
        diag["warning"] = f"PyTorch import gagal: {import_error}"
        return diag

    try:
        diag["torch_version"] = str(getattr(torch, "__version__", ""))
        diag["torch_cuda"] = torch_cuda_version(torch)
        diag["cuda_available"] = torch_cuda_available(torch)
        if diag["cuda_available"]:
            diag["device_name"] = str(torch.cuda.get_device_name(0))
            try:
                diag["device_capability"] = ".".join(map(str, torch.cuda.get_device_capability(0)))
            except Exception:
                diag["device_capability"] = ""
        ok, msg = validate_cuda126_for_jetson(torch, require_cuda=False)
        if not ok:
            diag["warning"] = msg
        elif not diag["cuda_available"]:
            diag["warning"] = "CUDA tidak tersedia; live GRU akan memakai CPU."
    except Exception as exc:
        diag["warning"] = str(exc)
    return diag


def diagnostics_text(diag: dict[str, Any] | None = None) -> str:
    data = diagnostics() if diag is None else diag
    lines = [
        f"python: {data.get('python', '-')}",
        f"torch: {data.get('torch_version', '-')} cuda={data.get('torch_cuda', '-')}",
        f"cuda_available: {data.get('cuda_available', False)}",
        f"device: {data.get('device_name', '-') or '-'} cc={data.get('device_capability', '-') or '-'}",
        f"tensorrt: {data.get('tensorrt_import_ok', False)} {data.get('tensorrt_version', '')}",
        f"opencv: {data.get('opencv_version', '-') or '-'} gstreamer={data.get('opencv_gstreamer', '-')}",
    ]
    warning = data.get("warning")
    if warning:
        lines.append(f"warning: {warning}")
    return "\n".join(lines)
