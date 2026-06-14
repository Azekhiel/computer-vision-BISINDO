#!/usr/bin/env bash
set -e
if [ $# -lt 1 ]; then
  echo "Usage: bash run_extract_better_10fps.sh /path/to/video.mp4 [feature_mode]"
  exit 1
fi
VIDEO="$1"
MODE="${2:-btj_global_local}"
python3 extract_video_better_v6.py "$VIDEO" \
  --feature-mode "$MODE" \
  --target-fps 10 \
  --width 640 --height 480 \
  --center-crop 0.86 \
  --proc-width 320 \
  --shoulder-backend mp-pose \
  --pose-every 3 \
  --pose-proc-width 256 \
  --shoulder-smooth-alpha 0.35 \
  --hand-model-complexity 0 \
  --pose-model-complexity 0 \
  --det-conf 0.55 \
  --track-conf 0.60 \
  --smooth-alpha 0.80 \
  --hold-frames 3 \
  --gif-width 480
