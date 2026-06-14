#!/usr/bin/env bash
set -e
if [ $# -lt 1 ]; then
  echo "Usage: bash run_extract_10fps.sh /path/to/video.mp4 [feature_mode]"
  exit 1
fi
VIDEO="$1"
MODE="${2:-btj_global_local}"
python3 extract_video_skeleton_10fps.py "$VIDEO" \
  --feature-mode "$MODE" \
  --target-fps 10 \
  --width 640 --height 480 \
  --proc-width 256 \
  --center-crop 0.86 \
  --shoulder-backend none \
  --gif-width 360
