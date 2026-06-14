#!/usr/bin/env python3
"""
extract_video_better_v6.py
==========================
Offline BISINDO video extractor with better tracking quality.

Key fix vs older extractor:
- OLD: skipped frames first, then ran MediaPipe only on sampled 10 FPS frames.
- NEW: runs MediaPipe on EVERY decoded source frame, then only SAVES/EXPORTS at target FPS.

This keeps MediaPipe's temporal tracker warm and usually makes offline extraction
better than live, because we are not constrained by real-time FPS.

Outputs:
1) overlay GIF: video + real shoulders + hands
2) skeleton-only GIF
3) .npz features
4) .csv features
5) .json metadata
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence

import cv2
import numpy as np

try:
    import imageio.v2 as imageio
except Exception:
    imageio = None

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from live_bisindo_mp_real_shoulder_v6 import (  # type: ignore
    FEATURE_DIMS,
    UltraMediaPipeExtractor,
    center_crop,
    draw_shoulders,
    draw_simple_hand,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Better offline video extractor for BISINDO skeleton/features")
    p.add_argument("video", type=str)
    p.add_argument("--out-dir", type=str, default=None)
    p.add_argument("--feature-mode", choices=list(FEATURE_DIMS), default="btj_global_local")
    p.add_argument("--target-fps", type=float, default=10.0)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--center-crop", type=float, default=0.86)

    # Offline quality defaults: a bit heavier than live, because video extraction is allowed to be slower.
    p.add_argument("--proc-width", type=int, default=320, help="Hand internal processing width. 288/320/384 for offline.")
    p.add_argument("--shoulder-backend", choices=["none", "mp-pose"], default="mp-pose")
    p.add_argument("--pose-proc-width", type=int, default=256)
    p.add_argument("--pose-every", type=int, default=3, help="In source frames. 1=best shoulder, 3=balanced offline.")
    p.add_argument("--shoulder-smooth-alpha", type=float, default=0.35)
    p.add_argument("--hand-every", type=int, default=1)
    p.add_argument("--hand-model-complexity", type=int, choices=[0, 1], default=0)
    p.add_argument("--pose-model-complexity", type=int, choices=[0, 1], default=0)
    p.add_argument("--det-conf", type=float, default=0.55)
    p.add_argument("--track-conf", type=float, default=0.60)
    p.add_argument("--smooth-alpha", type=float, default=0.80)
    p.add_argument("--hold-frames", type=int, default=3)
    p.add_argument("--mirror-input", action="store_true")
    p.add_argument("--no-mirror-handedness", action="store_true")

    # Important switch. Default false means process all frames for best quality.
    p.add_argument("--sample-before-process", action="store_true", help="Old faster behavior: skip frames before MediaPipe. Lower quality, faster.")

    p.add_argument("--gif-width", type=int, default=480)
    p.add_argument("--skeleton-bg", choices=["black", "white"], default="black")
    p.add_argument("--max-gif-frames", type=int, default=0, help="0=no limit. Use e.g. 120 to cap 12s at 10 FPS.")
    p.add_argument("--no-gif", action="store_true", help="Only save features/metadata, no GIF.")
    return p.parse_args()


def save_gif(frames: Sequence[np.ndarray], path: Path, fps: float, max_width: int = 480) -> None:
    if imageio is None:
        raise RuntimeError("imageio is not installed")
    if not frames:
        print(f"[GIF] skip empty {path}")
        return
    out = []
    for bgr in frames:
        h, w = bgr.shape[:2]
        if max_width > 0 and w > max_width:
            sc = max_width / float(w)
            bgr = cv2.resize(bgr, (max_width, int(round(h * sc))), interpolation=cv2.INTER_AREA)
        out.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    imageio.mimsave(path, out, duration=1.0 / max(float(fps), 1e-6), loop=0)
    print(f"[GIF] saved {len(out)} frames -> {path}")


def should_keep(t_video: float, next_keep_t: float, eps: float = 1e-9) -> bool:
    return t_video + eps >= next_keep_t


def main() -> None:
    args = parse_args()
    video_path = Path(args.video)
    if not video_path.is_file():
        raise FileNotFoundError(video_path)

    out_dir = Path(args.out_dir) if args.out_dir else video_path.parent / f"{video_path.stem}_better_v6_{int(round(args.target_fps))}fps"
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if src_fps <= 1e-6:
        src_fps = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_sec = (total_frames / src_fps) if total_frames > 0 else None

    extractor = UltraMediaPipeExtractor(
        feature_mode=args.feature_mode,
        shoulder_backend=args.shoulder_backend,
        proc_width=args.proc_width,
        pose_proc_width=args.pose_proc_width,
        pose_every=args.pose_every,
        hand_every=args.hand_every,
        hand_model_complexity=args.hand_model_complexity,
        pose_model_complexity=args.pose_model_complexity,
        det_conf=args.det_conf,
        track_conf=args.track_conf,
        smooth_alpha=args.smooth_alpha,
        hold_frames=args.hold_frames,
        mirror_input=args.mirror_input,
        mirror_handedness=not args.no_mirror_handedness,
        shoulder_smooth_alpha=args.shoulder_smooth_alpha,
    )

    overlay_frames: List[np.ndarray] = []
    skeleton_frames: List[np.ndarray] = []
    feature_rows: List[np.ndarray] = []
    meta_rows: List[dict] = []

    target_fps = float(args.target_fps)
    target_dt = 1.0 / max(target_fps, 1e-6)
    next_keep_t = 0.0

    frame_i = 0
    kept = 0
    processed = 0
    t_start = time.perf_counter()
    last_result = None

    print("=" * 72)
    print("BISINDO Better Video Extract V6")
    print(f"input={video_path}")
    print(f"source_fps={src_fps:.3f} target_fps={target_fps:.3f} frames={total_frames or '?'}")
    print(f"mode={args.feature_mode} dim={FEATURE_DIMS[args.feature_mode]}")
    print(f"process_all_frames={not args.sample_before_process} shoulder={args.shoulder_backend} pose_every={args.pose_every}")
    print("=" * 72)

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            frame_i += 1
            t_video = (frame_i - 1) / src_fps

            if frame.shape[1] != args.width or frame.shape[0] != args.height:
                frame = cv2.resize(frame, (args.width, args.height), interpolation=cv2.INTER_AREA)
            work = center_crop(frame, args.center_crop)

            keep_now = should_keep(t_video, next_keep_t)

            # Best quality: always process all source frames to keep MediaPipe tracker stable.
            # Faster old mode: process only sampled frames.
            if (not args.sample_before_process) or keep_now:
                last_result = extractor.process(work)
                processed += 1

            if not keep_now:
                continue

            while next_keep_t <= t_video + 1e-9:
                next_keep_t += target_dt

            if last_result is None:
                continue
            result = last_result

            overlay = work.copy()
            draw_shoulders(overlay, result.shoulders)
            draw_simple_hand(overlay, result.left, (80, 220, 80))
            draw_simple_hand(overlay, result.right, (80, 160, 255))
            cv2.putText(
                overlay,
                f"{video_path.name} | {args.feature_mode}:{FEATURE_DIMS[args.feature_mode]} | out={target_fps:g}fps | shoulder={args.shoulder_backend}",
                (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1, cv2.LINE_AA,
            )
            cv2.putText(
                overlay,
                f"t={t_video:.2f}s src_frame={frame_i} hand={result.hand_ms:.1f}ms pose={result.pose_ms:.1f}ms",
                (8, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (230, 230, 230), 1, cv2.LINE_AA,
            )

            if args.skeleton_bg == "white":
                skeleton = np.full_like(work, 255)
                lcol, rcol, text_col = (0, 150, 0), (0, 100, 220), (30, 30, 30)
            else:
                skeleton = np.zeros_like(work)
                lcol, rcol, text_col = (80, 220, 80), (80, 160, 255), (190, 190, 190)
            draw_shoulders(skeleton, result.shoulders)
            draw_simple_hand(skeleton, result.left, lcol)
            draw_simple_hand(skeleton, result.right, rcol)
            cv2.putText(
                skeleton,
                f"skeleton only | {args.feature_mode}:{FEATURE_DIMS[args.feature_mode]} | {target_fps:g}fps",
                (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, text_col, 1, cv2.LINE_AA,
            )

            if args.max_gif_frames <= 0 or len(overlay_frames) < args.max_gif_frames:
                overlay_frames.append(overlay)
                skeleton_frames.append(skeleton)

            feature_rows.append(result.vector.astype(np.float32))
            meta_rows.append({
                "kept_index": kept,
                "source_frame_index": frame_i,
                "time_sec": float(t_video),
                "feature_mode": args.feature_mode,
                "feature_dim": int(result.vector.shape[0]),
                "left_present": float(result.present[0]),
                "right_present": float(result.present[1]),
                "left_detected": float(result.detected[0]),
                "right_detected": float(result.detected[1]),
                "left_held": float(result.held[0]),
                "right_held": float(result.held[1]),
                "left_score": float(result.scores[0]),
                "right_score": float(result.scores[1]),
                "hand_ms": float(result.hand_ms),
                "pose_ms": float(result.pose_ms),
                "infer_fps_est": float(result.fps_infer),
            })
            kept += 1

            if kept % 20 == 0:
                elapsed = time.perf_counter() - t_start
                print(f"[PROGRESS] kept={kept} processed={processed} src={frame_i}/{total_frames or '?'} t={t_video:.1f}s wall={elapsed:.1f}s")

    finally:
        cap.release()
        extractor.close()

    if not feature_rows:
        raise RuntimeError("No sampled frames were extracted.")

    arr = np.stack(feature_rows).astype(np.float32)
    ts = time.strftime("%Y%m%d_%H%M%S")
    base = f"{video_path.stem}_{args.feature_mode}_{int(round(target_fps))}fps_v6_{ts}"

    overlay_gif = out_dir / f"{base}_overlay.gif"
    skeleton_gif = out_dir / f"{base}_skeleton_only.gif"
    npz_path = out_dir / f"{base}.npz"
    csv_path = out_dir / f"{base}.csv"
    json_path = out_dir / f"{base}_meta.json"

    if not args.no_gif:
        save_gif(overlay_frames, overlay_gif, fps=target_fps, max_width=args.gif_width)
        save_gif(skeleton_frames, skeleton_gif, fps=target_fps, max_width=args.gif_width)

    np.savez_compressed(
        npz_path,
        features=arr,
        feature_mode=args.feature_mode,
        feature_dim=arr.shape[1],
        target_fps=target_fps,
        source_fps=src_fps,
    )
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time_sec", "source_frame_index"] + [f"f{i}" for i in range(arr.shape[1])])
        for row_meta, row_feat in zip(meta_rows, arr):
            writer.writerow([row_meta["time_sec"], row_meta["source_frame_index"], *row_feat.tolist()])
    with json_path.open("w") as f:
        json.dump({
            "input_video": str(video_path),
            "source_fps": src_fps,
            "source_frame_count": total_frames,
            "source_duration_sec": duration_sec,
            "output_target_fps": target_fps,
            "sampled_frames": kept,
            "processed_frames": processed,
            "process_all_frames": not args.sample_before_process,
            "feature_mode": args.feature_mode,
            "feature_dim": int(arr.shape[1]),
            "settings": vars(args),
            "frames": meta_rows,
        }, f, indent=2)

    print("=" * 72)
    print("DONE")
    print(f"Input video     : {video_path}")
    print(f"Source FPS      : {src_fps:.3f}")
    print(f"Target FPS      : {target_fps:.3f}")
    print(f"Processed frames: {processed}")
    print(f"Sampled frames  : {kept}")
    print(f"Feature mode    : {args.feature_mode} (dim={arr.shape[1]})")
    if not args.no_gif:
        print(f"Overlay GIF     : {overlay_gif}")
        print(f"Skeleton GIF    : {skeleton_gif}")
    print(f"NPZ features    : {npz_path}")
    print(f"CSV features    : {csv_path}")
    print(f"Metadata JSON   : {json_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
