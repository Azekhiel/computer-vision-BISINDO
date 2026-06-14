#!/usr/bin/env bash
set -e
python3 live_bisindo_mp_modes_v5.py \
  --cam 0 \
  --width 640 --height 480 --fps 30 \
  --threaded-cam \
  --fourcc MJPG \
  --feature-mode btj_local \
  --proc-width 384 \
  --center-crop 0.92 \
  --hand-model-complexity 1 \
  --min-det-conf 0.60 \
  --min-track-conf 0.60 \
  --hand-every 1 \
  --hold-frames 4 \
  --smooth-alpha 0.70 \
  --shoulder-backend mp-pose \
  --shoulder-every 8 \
  --shoulder-proc-width 224 \
  --pose-model-complexity 0 \
  --z-mode blend \
  --preview-width 640 \
  --out-dir runs_mp_feature_modes_v5 \
  --perf-log-every 30 
