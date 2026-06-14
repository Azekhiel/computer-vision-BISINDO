#!/usr/bin/env python3
"""
extract_video_robust_v4.py
==========================
Offline video extraction using arm-locked + anti-jitter hand tracker.

Example:
    python3 extract_video_robust_v4.py input.mp4 --feature-mode compat --out-dir out
    python3 extract_video_robust_v4.py input.mp4 --feature-mode stable --no-display
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import imageio
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from feature_extractor_robust_v4 import make_tracker, extract_features, draw_landmarks, draw_hud


def frames_to_gif(frames, path: str, fps: float = 12.0, max_width: int = 360) -> None:
    if not frames:
        return
    out = []
    for bgr in frames:
        h, w = bgr.shape[:2]
        if w > max_width:
            scale = max_width / w
            bgr = cv2.resize(bgr, (max_width, int(h * scale)), interpolation=cv2.INTER_AREA)
        out.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    imageio.mimsave(path, out, duration=1.0 / fps, loop=0)
    print(f"[GIF] {len(out)} frames -> {path}")


def open_video(path: Path):
    gst = (
        f"uridecodebin uri=file://{path.absolute()} ! "
        f"nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! "
        f"video/x-raw, format=BGR ! appsink max-buffers=1 drop=true sync=false"
    )
    cap = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        print("[WARN] GStreamer decode failed, fallback OpenCV CPU decode")
        cap = cv2.VideoCapture(str(path))
    return cap


def main() -> None:
    parser = argparse.ArgumentParser(description="Robust V3 video feature extraction")
    parser.add_argument("video", type=str)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--feature-mode", choices=["compat", "stable", "palm", "palm_angles"], default="compat")
    parser.add_argument("--model-complexity", type=int, default=0, choices=[0, 1])
    parser.add_argument("--det-conf", type=float, default=0.55)
    parser.add_argument("--track-conf", type=float, default=0.55)
    parser.add_argument("--pose-every", type=int, default=1, help="For offline extraction, 1 is most stable.")
    parser.add_argument("--hold-frames", type=int, default=4)
    parser.add_argument("--arm-hold-frames", type=int, default=8)
    parser.add_argument("--arm-gate", type=float, default=0.13)
    parser.add_argument("--hand-deadband", type=float, default=0.0065)
    parser.add_argument("--pose-deadband", type=float, default=0.0045)
    parser.add_argument("--snap-strength", type=float, default=0.35)
    parser.add_argument("--snap-max", type=float, default=0.030)
    parser.add_argument("--no-snap-hand-to-arm", action="store_true")
    parser.add_argument("--no-arm-lock", action="store_true")
    parser.add_argument("--allow-unanchored", action="store_true")
    parser.add_argument("--no-mirror-handedness", action="store_true")
    parser.add_argument("--draw-full-pose", action="store_true")
    parser.add_argument("--draw-raw", action="store_true")
    parser.add_argument("--no-draw-gates", action="store_true")
    parser.add_argument("--gif-every", type=int, default=0)
    parser.add_argument("--gif-max-sec", type=float, default=8.0)
    parser.add_argument("--gif-scale", type=int, default=360)
    parser.add_argument("--no-display", action="store_true")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.is_file():
        raise FileNotFoundError(video_path)

    out_dir = Path(args.out_dir) if args.out_dir else video_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = video_path.stem

    cap = open_video(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    tracker = make_tracker(
        model_complexity=args.model_complexity,
        det_conf=args.det_conf,
        track_conf=args.track_conf,
        pose_every=args.pose_every,
        hold_frames=args.hold_frames,
        arm_hold_frames=args.arm_hold_frames,
        mirror_handedness=not args.no_mirror_handedness,
        arm_locked=not args.no_arm_lock,
        arm_gate=args.arm_gate,
        allow_unanchored_when_pose_missing=args.allow_unanchored,
        hand_deadband=args.hand_deadband,
        pose_deadband=args.pose_deadband,
        snap_strength=0.0 if args.no_snap_hand_to_arm else args.snap_strength,
        snap_max=args.snap_max,
    )

    overlay_path = out_dir / f"{stem}_robust_v4_overlay.mp4"
    writer = cv2.VideoWriter(str(overlay_path), cv2.VideoWriter_fourcc(*"mp4v"), src_fps, (args.width, args.height))

    gif_every = args.gif_every if args.gif_every > 0 else max(1, int(src_fps / 12.0))
    gif_limit = int(args.gif_max_sec * 12)
    gif_frames = []
    features = []
    meta_rows = []

    fps_ema = 0.0
    alpha = 0.12
    frame_i = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame.shape[1] != args.width or frame.shape[0] != args.height:
                frame = cv2.resize(frame, (args.width, args.height), interpolation=cv2.INTER_AREA)

            t0 = time.perf_counter()
            result = tracker.process(frame)
            feat = extract_features(result, mode=args.feature_mode)
            dt = time.perf_counter() - t0
            fps_ema = (1 - alpha) * fps_ema + alpha * (1.0 / max(dt, 1e-6))

            features.append(feat)
            meta_rows.append([
                frame_i,
                result.infer_ms,
                result.pose_ok,
                result.left_hand.quality,
                result.right_hand.quality,
                float(result.left_hand.detected),
                float(result.right_hand.detected),
                result.self_handshake_score,
                result.left_hand.arm_link_score,
                result.right_hand.arm_link_score,
                result.left_hand.motion_ema,
                result.right_hand.motion_ema,
                result.left_hand.snap_offset,
                result.right_hand.snap_offset,
            ])

            vis = draw_landmarks(
                frame.copy(),
                result,
                draw_full_pose=args.draw_full_pose,
                draw_gates=not args.no_draw_gates,
                draw_raw_detections=args.draw_raw,
            )
            vis = draw_hud(vis, result, fps_ema, feature_mode=args.feature_mode)
            if total_frames:
                cv2.putText(vis, f"Frame {frame_i}/{total_frames} {100 * frame_i / max(total_frames - 1, 1):.0f}%",
                            (args.width - 220, args.height - 8), cv2.FONT_HERSHEY_SIMPLEX,
                            0.42, (150, 150, 150), 1, cv2.LINE_AA)
            writer.write(vis)

            if frame_i % gif_every == 0 and len(gif_frames) < gif_limit:
                gif_frames.append(vis.copy())

            if not args.no_display:
                cv2.imshow("Robust Extract V3", vis)
                if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
                    break

            if frame_i % 30 == 0:
                print(f"\r[{frame_i}/{total_frames or '?'}] extracting...", end="", flush=True)
            frame_i += 1
    except KeyboardInterrupt:
        print("\n[INFO] interrupted")
    finally:
        cap.release()
        writer.release()
        tracker.close()
        cv2.destroyAllWindows()

    if features:
        arr = np.stack(features).astype(np.float32)
    else:
        empty_dim = {"compat": 179, "stable": 337, "palm": 52, "palm_angles": 84}[args.feature_mode]
        arr = np.zeros((0, empty_dim), dtype=np.float32)

    feat_path = out_dir / f"{stem}_robust_v4_features_{args.feature_mode}.npy"
    meta_path = out_dir / f"{stem}_robust_v4_meta.json"
    quality_path = out_dir / f"{stem}_robust_v4_quality.csv"
    np.save(str(feat_path), arr)
    np.savetxt(
        str(quality_path),
        np.asarray(meta_rows, dtype=np.float32),
        delimiter=",",
        header="frame,infer_ms,pose_ok,left_q,right_q,left_detected,right_detected,self_handshake,left_arm_link,right_arm_link,left_motion,right_motion,left_snap,right_snap",
        comments="",
    )
    frames_to_gif(gif_frames, str(out_dir / f"{stem}_robust_v4.gif"), fps=src_fps / gif_every, max_width=args.gif_scale)

    meta = {
        "video": str(video_path),
        "frames": int(arr.shape[0]),
        "feature_mode": args.feature_mode,
        "feature_dim": int(arr.shape[1]) if arr.ndim == 2 else 0,
        "overlay": str(overlay_path),
        "features": str(feat_path),
        "quality_csv": str(quality_path),
        "arm_lock": bool(not args.no_arm_lock),
        "arm_gate": float(args.arm_gate),
        "allow_unanchored": bool(args.allow_unanchored),
        "hand_deadband": float(args.hand_deadband),
        "pose_deadband": float(args.pose_deadband),
        "snap_strength": float(0.0 if args.no_snap_hand_to_arm else args.snap_strength),
        "snap_max": float(args.snap_max),
        "notes": "V4: anti-jitter + palm/shoulder feature modes; compat=179; stable=337; palm=52; palm_angles=84",
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"\n[DONE] frames={arr.shape[0]} dim={arr.shape[1] if arr.ndim == 2 else 0}")
    print(f"[OUT] features -> {feat_path}")
    print(f"[OUT] overlay  -> {overlay_path}")
    print(f"[OUT] meta     -> {meta_path}")


if __name__ == "__main__":
    main()
