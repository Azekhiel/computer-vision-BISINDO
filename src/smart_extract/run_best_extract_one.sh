#!/usr/bin/env bash
set -e
if [ $# -lt 1 ]; then
  echo "Usage: bash run_best_extract_one.sh /path/video.mp4 [feature_mode]"
  exit 1
fi

VIDEO="$1"
MODE="${2:-btj_global_local}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python3 "$SCRIPT_DIR/extract_video_smart_v8.py" "$VIDEO" \
  --feature-mode "$MODE" \
  --target-fps 10 \
  --width 640 --height 480 \
  --center-crop 1.0 \
  --proc-width 384 \
  --shoulder-backend mp-pose \
  --pose-every 3 \
  --pose-proc-width 256 \
  --det-conf 0.40 \
  --track-conf 0.45 \
  --smooth-alpha 0.78 \
  --hold-frames 5 \
  --smart-mode best \
  --search-radius 2 \
  --enhance auto \
  --fallback-variants auto,clahe_sharp,gamma_bright,sharp,denoise_clahe_sharp,none \
  --gif-width 420
