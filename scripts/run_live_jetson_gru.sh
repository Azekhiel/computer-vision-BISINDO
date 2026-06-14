#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${BISINDO_CUDA126_VENV:-${ROOT_DIR}/env_bisindo_cuda126}"
PYTHON_BIN="${VENV_DIR}/bin/python"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "env Jetson CUDA 12.6 belum siap: ${PYTHON_BIN}" >&2
  echo "Jalankan dulu:" >&2
  echo "  ${ROOT_DIR}/scripts/setup_jetson_cuda126.sh" >&2
  exit 1
fi

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/targets/aarch64-linux/lib:/usr/lib/aarch64-linux-gnu:/usr/lib/aarch64-linux-gnu/nvidia:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"
export YOLO_AUTOINSTALL=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

if ! "${PYTHON_BIN}" - <<'PY'
import jetson_runtime as jr

diag = jr.diagnostics()
print(jr.diagnostics_text(diag))
ok, msg = jr.validate_cuda126_for_jetson(require_cuda=True)
if not ok:
    raise SystemExit(msg)
PY
then
  echo >&2
  echo "Runtime CUDA 12.6 belum valid. Setup/perbaiki dengan:" >&2
  echo "  ${ROOT_DIR}/scripts/setup_jetson_cuda126.sh" >&2
  exit 1
fi

exec "${PYTHON_BIN}" "${ROOT_DIR}/src/live_gru_fast.py" \
  --schema "${BISINDO_GRU_SCHEMA:-smart180}" \
  --variant "${BISINDO_GRU_VARIANT:-auto}" \
  --profile "${BISINDO_GRU_PROFILE:-accurate10}" \
  --device "${BISINDO_GRU_DEVICE:-auto}" \
  --segment-mode "${BISINDO_GRU_SEGMENT_MODE:-auto}" \
  "$@"
