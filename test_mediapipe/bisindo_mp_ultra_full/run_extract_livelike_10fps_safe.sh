#!/usr/bin/env bash
# Safe runner: tidak pakai `exit` mendadak dan tidak nutup terminal kalau dijalankan dari file manager.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VIDEO="$1"
MODE="${2:-btj_global_local}"

pause_if_interactive() {
  if [ -t 0 ]; then
    echo ""
    read -r -p "Tekan ENTER untuk tutup..." _
  fi
}

if [ -z "$VIDEO" ]; then
  echo "ERROR: path video belum dikasih."
  echo ""
  echo "Cara pakai:"
  echo "  bash run_extract_livelike_10fps_safe.sh /path/to/video.mp4 [feature_mode]"
  echo ""
  echo "Contoh:"
  echo "  bash run_extract_livelike_10fps_safe.sh ./videos/test.mp4 btj_global_local"
  pause_if_interactive
  return 1 2>/dev/null || true
fi

if [ ! -f "$VIDEO" ]; then
  echo "ERROR: file video tidak ditemukan: $VIDEO"
  echo "Tips: kalau path ada spasi, pakai tanda kutip."
  echo "Contoh: bash run_extract_livelike_10fps_safe.sh \"/home/ta14/Videos/video test.mp4\""
  pause_if_interactive
  return 1 2>/dev/null || true
fi

PY_SCRIPT="$SCRIPT_DIR/extract_video_livelike_v7.py"
if [ ! -f "$PY_SCRIPT" ]; then
  # fallback kalau safe runner disimpan di folder berbeda
  PY_SCRIPT="./extract_video_livelike_v7.py"
fi

if [ ! -f "$PY_SCRIPT" ]; then
  echo "ERROR: extract_video_livelike_v7.py tidak ketemu."
  echo "Jalankan script ini dari folder bisindo_mp_v7_livelike, atau taruh safe runner di folder itu."
  pause_if_interactive
  return 1 2>/dev/null || true
fi

echo "[RUN] video=$VIDEO"
echo "[RUN] mode=$MODE"
echo "[RUN] script=$PY_SCRIPT"

python3 "$PY_SCRIPT" "$VIDEO" \
  --feature-mode "$MODE" \
  --target-fps 10 \
  --width 640 --height 480 \
  --center-crop 0.86 \
  --proc-width 256 \
  --shoulder-backend mp-pose \
  --pose-every 5 \
  --pose-proc-width 192 \
  --shoulder-smooth-alpha 0.35 \
  --hand-model-complexity 0 \
  --pose-model-complexity 0 \
  --det-conf 0.50 \
  --track-conf 0.50 \
  --smooth-alpha 0.85 \
  --hold-frames 2 \
  --gif-width 480

STATUS=$?
echo ""
if [ $STATUS -eq 0 ]; then
  echo "[DONE] Extract selesai."
else
  echo "[ERROR] Extract gagal dengan exit code: $STATUS"
fi
pause_if_interactive
return $STATUS 2>/dev/null || true
