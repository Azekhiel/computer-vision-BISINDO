#!/usr/bin/env bash
set -e
MODE=${1:-179}
python3 live_bisindo_mp_modes_v5.py \
  --cam 0 \
  --width 640 --height 480 --fps 30 \
  --threaded-cam \
  --fourcc MJPG \
  --feature-mode "$MODE" \
  --proc-width 320 \
  --center-crop 0.88 \
  --hand-model-complexity 0 \
  --min-det-conf 0.55 \
  --min-track-conf 0.55 \
  --hand-every 1 \
  --hold-frames 3 \
  --smooth-alpha 0.75 \
  --shoulder-backend mp-pose \
  --shoulder-every 12 \
  --shoulder-proc-width 192 \
  --pose-model-complexity 0 \
  --z-mode blend \
  --preview-width 426 \
  --no-overlay \
  --out-dir runs_mp_feature_modes_v5 \
  --perf-log-every 30
