#!/usr/bin/env bash
set -e
python3 live_bisindo_mp_ultra_full.py \
  --cam 0 \
  --width 640 --height 480 --fps 30 \
  --fourcc MJPG \
  --feature-mode 268 \
  --shoulder-backend mp-pose \
  --pose-every 20 \
  --pose-proc-width 144 \
  --proc-width 288 \
  --center-crop 0.88 \
  --hand-model-complexity 0 \
  --pose-model-complexity 0 \
  --det-conf 0.55 --track-conf 0.55 \
  --smooth-alpha 0.82 \
  --hold-frames 3 \
  --preview-width 426 \
  --no-overlay \
  --perf-log-every 60
