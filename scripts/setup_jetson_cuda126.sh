#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${BISINDO_CUDA126_VENV:-${ROOT_DIR}/env_bisindo_cuda126}"
PYTHON_BIN="${BISINDO_CUDA126_PYTHON:-/usr/bin/python3.10}"
TORCH_INDEX="${BISINDO_TORCH_INDEX:-https://pypi.jetson-ai-lab.io/jp6/cu126}"

if [[ "${1:-}" == "--recreate" ]]; then
  rm -rf "${VENV_DIR}"
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python 3.10 not found at ${PYTHON_BIN}" >&2
  exit 1
fi

if [[ ! -d "${VENV_DIR}" ]]; then
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/targets/aarch64-linux/lib:/usr/lib/aarch64-linux-gnu:/usr/lib/aarch64-linux-gnu/nvidia:${LD_LIBRARY_PATH:-}"
export YOLO_AUTOINSTALL=false

SYSTEM_DIST_PACKAGES="${BISINDO_SYSTEM_DIST_PACKAGES:-/usr/lib/python3.10/dist-packages}"
if [[ -d "${SYSTEM_DIST_PACKAGES}" ]]; then
  VENV_SITE_PACKAGES="$(python - <<'PY'
import site
print(site.getsitepackages()[0])
PY
)"
  printf '%s\n' "${SYSTEM_DIST_PACKAGES}" > "${VENV_SITE_PACKAGES}/jetson-system-dist-packages.pth"
else
  echo "Warning: Jetson system dist-packages not found at ${SYSTEM_DIST_PACKAGES}; TensorRT apt bindings may be hidden." >&2
fi

python -m pip install --upgrade pip setuptools wheel
python -m pip install --no-cache-dir numpy==1.26.4

# Keep the CUDA stack explicitly on JetPack 6 / CUDA 12.6. Do not use the
# default PyPI torch wheels on Jetson; they can pull CUDA 13 packages.
python -m pip install --no-cache-dir \
  torch==2.8.0 torchvision==0.23.0 \
  --index-url "${TORCH_INDEX}"

python -m pip install --no-cache-dir -r "${ROOT_DIR}/requirements/requirements_jetson_cuda126.txt"

# Ultralytics depends on torch/torchvision, so install it without dependencies
# after the correct Jetson torch stack is already present.
python -m pip install --no-cache-dir ultralytics ultralytics-thop --no-deps

PYTHONPATH="${ROOT_DIR}/src" python - <<'PY'
import sys
import torch
import cv2
import mediapipe
import ultralytics
import tensorrt as trt

print("python", sys.version.split()[0])
print("torch", torch.__version__)
print("torch cuda", torch.version.cuda)
print("cuda available", torch.cuda.is_available())
print("device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
print("opencv", cv2.__version__)
print("mediapipe", mediapipe.__version__)
print("ultralytics", ultralytics.__version__)
print("tensorrt", trt.__version__)
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available from this PyTorch install.")
x = torch.randn(1024, 1024, device="cuda")
print("matmul mean", float((x @ x).mean().item()))
PY

echo
echo "BISINDO CUDA 12.6 environment is ready:"
echo "  source ${VENV_DIR}/bin/activate"
echo "  PYTHONPATH=${ROOT_DIR}/src python ${ROOT_DIR}/src/main_ui.py"
