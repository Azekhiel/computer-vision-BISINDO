#!/usr/bin/env python3
"""
extract_video.py
================
Offline feature extraction from a video file – 640 × 480.
Optimized for Jetson Orin Nano (JetPack 6.2, CUDA 12.6)
"""

import argparse
import time
from pathlib import Path
import cv2
import imageio
import numpy as np
import sys, os

sys.path.insert(0, os.path.dirname(__file__))
from feature_extractor import make_holistic, extract_features, draw_landmarks, draw_hud

def frames_to_gif(frames: list[np.ndarray], path: str, fps: float = 12, max_width: int = 320) -> None:
    if not frames: return
    out = []
    for bgr in frames:
        h, w = bgr.shape[:2]
        if w > max_width:
            bgr = cv2.resize(bgr, (max_width, int(h * max_width / w)), interpolation=cv2.INTER_AREA)
        out.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    imageio.mimsave(path, out, duration=1.0 / fps, loop=0)
    print(f"[GIF] {len(out)} frames → {path}")

def main():
    parser = argparse.ArgumentParser(description="MediaPipe Extraction (Orin Nano)")
    parser.add_argument("video", type=str, help="Input video file path")
    parser.add_argument("--out-dir",      type=str,   default=None)
    parser.add_argument("--gif-every",    type=int,   default=0)
    parser.add_argument("--gif-max-sec",  type=float, default=10.0)
    parser.add_argument("--gif-scale",    type=int,   default=320)
    parser.add_argument("--no-display",   action="store_true")
    args = parser.parse_args()

    WIDTH, HEIGHT = 640, 480
    video_path = Path(args.video)
    if not video_path.is_file(): raise FileNotFoundError(f"Video not found: {video_path}")

    out_dir = Path(args.out_dir) if args.out_dir else video_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = video_path.stem

    # Coba menggunakan pipeline GStreamer untuk Hardware Decode (NVDEC via Jetson)
    gst_in = (f"uridecodebin uri=file://{video_path.absolute()} ! "
              f"nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink")
    cap = cv2.VideoCapture(gst_in, cv2.CAP_GSTREAMER)
    
    if not cap.isOpened():
        print("[WARN] Hardware Decode GStreamer gagal. Fallback to CPU OpenCV.")
        cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened(): raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_fps      = cap.get(cv2.CAP_PROP_FPS) or 30.0

    gif_max_frames = int(args.gif_max_sec * 15)
    gif_every = args.gif_every if args.gif_every > 0 else max(1, int(src_fps / 15))
    gif_fps   = src_fps / gif_every

    overlay_path = out_dir / f"{stem}_overlay.mp4"
    # Karena Orin Nano tidak punya NVENC, kita tetap menggunakan CPU encoding (mp4v)
    writer = cv2.VideoWriter(str(overlay_path), cv2.VideoWriter_fourcc(*"mp4v"), src_fps, (WIDTH, HEIGHT))

    holistic = make_holistic(model_complexity=0, smooth=True, det_conf=0.45, track_conf=0.45)
    features, gif_frames = [], []
    frame_idx, fps_ema, alpha, t0 = 0, 0.0, 0.1, time.perf_counter()

    try:
        while True:
            ret, frame = cap.read()
            if not ret: break

            if (frame.shape[1], frame.shape[0]) != (WIDTH, HEIGHT):
                frame = cv2.resize(frame, (WIDTH, HEIGHT))

            t_inf = time.perf_counter()
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = holistic.process(rgb)
            rgb.flags.writeable = True
            dt = time.perf_counter() - t_inf

            feat = extract_features(results)
            features.append(feat)

            elapsed_fps = 1.0 / max(dt, 1e-4)
            fps_ema = alpha * elapsed_fps + (1 - alpha) * fps_ema

            vis = draw_landmarks(frame.copy(), results)
            vis = draw_hud(vis, feat, fps_ema)

            pct_done = 100 * frame_idx / max(total_frames - 1, 1)
            cv2.putText(vis, f"Frame {frame_idx}/{total_frames} {pct_done:.0f}%", (WIDTH - 200, HEIGHT - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (130, 130, 130), 1)

            writer.write(vis)
            if frame_idx % gif_every == 0 and len(gif_frames) < gif_max_frames:
                gif_frames.append(vis.copy())

            if not args.no_display:
                cv2.imshow("Extracting", vis)
                if cv2.waitKey(1) & 0xFF in (ord('q'), 27): break

            if frame_idx % 30 == 0:
                print(f"\r[{frame_idx}/{total_frames}] Memproses...  ", end="", flush=True)
            frame_idx += 1

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        cap.release()
        writer.release()
        holistic.close()
        cv2.destroyAllWindows()

    feat_arr = np.stack(features).astype(np.float32)
    np.save(str(out_dir / f"{stem}_features.npy"), feat_arr)
    frames_to_gif(gif_frames, str(out_dir / f"{stem}_video.gif"), fps=gif_fps)
    print(f"\n[DONE] Selesai memproses {frame_idx} frames.")

if __name__ == "__main__":
    main()