#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/env_bisindo_cuda126"
REQ_FILE="${ROOT_DIR}/src_test_tts/requirements.txt"
CONSTRAINTS_FILE="${ROOT_DIR}/src_test_tts/constraints.txt"

if ! command -v python3.10 >/dev/null 2>&1; then
  echo "ERROR: python3.10 tidak ditemukan."
  echo "Install Python 3.10 dulu, lalu jalankan ulang: bash src_test_tts/setup_env.sh"
  exit 1
fi

if [ ! -d "${VENV_DIR}" ]; then
  echo "[INFO] Membuat venv: ${VENV_DIR}"
  python3.10 -m venv "${VENV_DIR}"
else
  echo "[INFO] Venv sudah ada, pakai ulang: ${VENV_DIR}"
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip setuptools wheel
python -m pip install -c "${CONSTRAINTS_FILE}" -r "${REQ_FILE}"

echo
echo "Setup selesai."
echo "Aktifkan venv dengan:"
echo "source env_bisindo_cuda126/bin/activate"
echo
echo "Langkah berikutnya:"
echo "python src_test_tts/cli.py init"
echo "python src_test_tts/cli.py download-model"
