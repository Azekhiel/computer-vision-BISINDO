#!/usr/bin/env bash
set -e
MODE="${1:-btj_global_local}"
python3 live_bisindo_mp_real_shoulder_v6.py \
  --cam 0 \
  --width 640 --height 480 --fps 30 \
  --feature-mode "$MODE" \
  --shoulder-backend mp-pose \
  --pose-every 5 \
  --pose-proc-width 192 \
  --shoulder-smooth-alpha 0.35 \
  --proc-width 256 \
  --hand-model-complexity 0 \
  --pose-model-complexity 0 \
  --det-conf 0.50 \
  --track-conf 0.50 \
  --smooth-alpha 0.85 \
  --hold-frames 2 \
  --center-crop 0.86 \
  --preview-width 640
