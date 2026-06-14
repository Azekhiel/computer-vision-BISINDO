#!/usr/bin/env bash
set -e
if [ $# -lt 1 ]; then
  echo "Usage: bash run_extract_livelike_10fps.sh /path/to/video.mp4 [feature_mode]"
  exit 1
fi
VIDEO="$1"
MODE="${2:-btj_global_local}"
python3 extract_video_livelike_v7.py "$VIDEO" \
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
