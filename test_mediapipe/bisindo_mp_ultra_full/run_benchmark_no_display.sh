#!/usr/bin/env bash
set -e
MODE="${1:-btj_global_local}"
python3 live_bisindo_mp_ultra_full.py \
  --cam 0 \
  --width 640 --height 480 --fps 30 \
  --fourcc MJPG \
  --feature-mode "$MODE" \
  --shoulder-backend none \
  --proc-width 256 \
  --center-crop 0.86 \
  --hand-model-complexity 0 \
  --det-conf 0.50 --track-conf 0.50 \
  --smooth-alpha 0.88 \
  --hold-frames 2 \
  --no-display \
  --perf-log-every 60
