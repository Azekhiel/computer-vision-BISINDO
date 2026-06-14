#!/usr/bin/env bash
set -e
if [ $# -lt 1 ]; then
  echo "Usage: bash run_smart_extract_one.sh /path/video.mp4 [feature_mode]"
  exit 1
fi

VIDEO="$1"
MODE="${2:-btj_global_local}"

python3 extract_video_smart_v8.py "$VIDEO" \
  --feature-mode "$MODE" \
  --target-fps 10 \
  --width 640 --height 480 \
  --center-crop 1.0 \
  --proc-width 320 \
  --shoulder-backend mp-pose \
  --pose-every 5 \
  --pose-proc-width 224 \
  --det-conf 0.45 \
  --track-conf 0.45 \
  --smooth-alpha 0.82 \
  --hold-frames 4 \
  --smart-mode smart \
  --enhance auto \
  --gif-width 420
