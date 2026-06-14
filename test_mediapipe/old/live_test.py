#!/usr/bin/env python3
"""
live_test.py - Optimized for Jetson Orin Nano
Live webcam feature extraction & overlay – 640 × 480.
"""

import argparse
import collections
import time
import threading
from pathlib import Path

import cv2
import imageio
import numpy as np

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from feature_extractor import make_holistic, extract_features, draw_landmarks, draw_hud

# ── Threaded Camera Setup untuk Jetson ────────────────────────────────────────
class ThreadedCamera:
    def __init__(self, src=0, use_csi=False, width=640, height=480, fps=30):
        if use_csi:
            # GStreamer pipeline khusus Jetson untuk Kamera CSI MIPI (nvarguscamerasrc)
            gst = (f"nvarguscamerasrc sensor-id={src} ! "
                   f"video/x-raw(memory:NVMM), width={width}, height={height}, format=NV12, framerate={fps}/1 ! "
                   f"nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink drop=1")
            self.cap = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
        else:
            # USB Camera standard v4l2
            gst = (f"v4l2src device=/dev/video{src} ! "
                   f"video/x-raw, width={width}, height={height}, framerate={fps}/1 ! "
                   f"videoconvert ! appsink drop=1")
            self.cap = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
            if not self.cap.isOpened():
                print("[WARN] GStreamer failed, fallback V4L2")
                self.cap = cv2.VideoCapture(src, cv2.CAP_V4L2)
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                self.cap.set(cv2.CAP_PROP_FPS, fps)
        
        self.ret, self.frame = self.cap.read()
        self.running = True
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                self.ret, self.frame = ret, frame

    def read(self):
        return self.ret, self.frame.copy() if self.ret else None

    def release(self):
        self.running = False
        self.thread.join()
        self.cap.release()

def frames_to_gif(frames, path, fps=10):
    if not frames: return
    out_frames = []
    for bgr in frames:
        h, w = bgr.shape[:2]
        if w > 320:
            scale = 320 / w
            bgr = cv2.resize(bgr, (320, int(h * scale)), interpolation=cv2.INTER_AREA)
        out_frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    imageio.mimsave(path, out_frames, duration=1.0/fps, loop=0)
    print(f"[GIF] Saved {len(out_frames)} frames → {path}")

def main():
    parser = argparse.ArgumentParser(description="MediaPipe live test – Jetson Orin Nano")
    parser.add_argument("--cam",     type=int,   default=0, help="Kamera ID (0, 1, dll)")
    parser.add_argument("--csi",     action="store_true", help="Gunakan ini jika memakai kamera CSI (MIPI)")
    parser.add_argument("--fps",     type=int,   default=30)
    parser.add_argument("--gif-sec", type=float, default=5.0)
    parser.add_argument("--gif-out", type=str,   default="live_test.gif")
    parser.add_argument("--no-display", action="store_true")
    args = parser.parse_args()

    WIDTH, HEIGHT = 640, 480
    ring = collections.deque(maxlen=int(args.gif_sec * args.fps))

    # Gunakan Multithreaded Capture untuk efisiensi CPU pada Jetson
    cam = ThreadedCamera(src=args.cam, use_csi=args.csi, width=WIDTH, height=HEIGHT, fps=args.fps)
    print(f"[CAM] Multithreaded Camera Opened. CSI Mode: {args.csi}")

    holistic = make_holistic(model_complexity=0, smooth=True, det_conf=0.45, track_conf=0.45)
    
    fps_counter, t_prev, display_fps = 0, time.perf_counter(), 0.0

    print("=" * 50)
    print("Controls: Q/ESC = quit | G = save GIF | S = snapshot | R = reset buffer")
    print("=" * 50)

    try:
        while True:
            ret, frame = cam.read()
            if not ret or frame is None:
                continue

            if (frame.shape[1], frame.shape[0]) != (WIDTH, HEIGHT):
                frame = cv2.resize(frame, (WIDTH, HEIGHT))

            # ── Inferensi ─────────────────────────────────────────
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = holistic.process(rgb)
            rgb.flags.writeable = True

            feat = extract_features(results)
            vis = draw_landmarks(frame, results)
            vis = draw_hud(vis, feat, display_fps)

            cv2.putText(vis, "[G] GIF  [S] Save  [R] Reset  [Q] Quit",
                        (6, HEIGHT - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (130, 130, 130), 1)

            ring.append(vis.copy())

            if not args.no_display:
                cv2.imshow("MediaPipe Live - Orin Nano", vis)

            fps_counter += 1
            t_now = time.perf_counter()
            if t_now - t_prev >= 0.5:
                display_fps = fps_counter / (t_now - t_prev)
                fps_counter, t_prev = 0, t_now

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q'), 27): break
            elif key in (ord('g'), ord('G')): frames_to_gif(list(ring), args.gif_out, fps=15)
            elif key in (ord('s'), ord('S')): 
                np.save("live_feat_snapshot.npy", feat)
                print("[SNAP] Saved → live_feat_snapshot.npy")
            elif key in (ord('r'), ord('R')): ring.clear()

    except KeyboardInterrupt:
        print("\n[INFO] Dihentikan paksa.")
    finally:
        cam.release()
        holistic.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()