#!/usr/bin/env bash
# setup_jetson.sh
# ───────────────────────────────────────────────────────────
# One-shot dependency installer for Jetson Orin Nano (JetPack 6.x)
# Ubuntu 22.04  |  CUDA 12.6  |  Python 3.10
#
# Run:  bash setup_jetson.sh
# ───────────────────────────────────────────────────────────

set -euo pipefail

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  MediaPipe Tools – Jetson Orin Nano setup"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# ── System packages ──────────────────────────────────────────
sudo apt-get update -qq
sudo apt-get install -y --no-install-recommends \
    python3-pip \
    python3-opencv \
    libopencv-dev \
    libgstreamer1.0-dev \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-libav \
    libgl1-mesa-glx \
    v4l-utils

# ── Python packages ──────────────────────────────────────────
pip3 install --upgrade pip

# MediaPipe: use the standard wheel (works on aarch64 Jetson with JetPack 6)
pip3 install mediapipe

# NumPy / imageio
pip3 install "numpy>=1.24" "imageio[ffmpeg]" "Pillow>=10"

echo ""
echo "✓  All dependencies installed."
echo ""
echo "Verify camera:"
echo "  v4l2-ctl --list-devices"
echo "  v4l2-ctl -d /dev/video0 --list-formats-ext"
echo ""
echo "Run live test:"
echo "  python3 live_test.py --cam 0"
echo ""
echo "Extract features from video:"
echo "  python3 extract_video.py input.mp4"
echo ""