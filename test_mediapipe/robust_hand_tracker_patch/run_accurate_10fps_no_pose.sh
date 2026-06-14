#!/usr/bin/env bash
set -e
python3 live_bisindo_mp_accurate_288.py \
  --cam 0 \
  --width 640 --height 480 --fps 30 \
  --threaded-cam \
  --fourcc MJPG \
  --proc-width 384 \
  --hand-model-complexity 1 \
  --min-det-conf 0.60 \
  --min-track-conf 0.60 \
  --hand-every 1 \
  --shoulder-backend none \
  --center-crop 0.92 \
  --z-mode blend \
  --hold-frames 8 \
  --smooth-alpha 0.62 \
  --preview-width 640 \
  --draw-every 1 \
  --perf-log-every 30
