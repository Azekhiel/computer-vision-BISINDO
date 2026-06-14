#!/usr/bin/env python3
"""
extract_video_smart_v8.py
=========================

Smart offline extractor for recorded BISINDO videos.

Goal:
- Keep output at a fixed target FPS, default 10 FPS.
- Make old recorded videos easier for MediaPipe Hands:
  brightness/gamma correction, CLAHE contrast, denoise, sharpen.
- If a target frame is bad/blurred, optionally search nearby frames and pick
  the best frame+preprocess variant using a lightweight MediaPipe Hands probe.
- Output:
  1) overlay GIF/video with skeleton
  2) skeleton-only GIF/video
  3) NPZ/CSV features
  4) JSON metadata with selected source frame + enhancement mode

Modes:
- fast  : auto-enhance target frame only. Fastest.
- smart : target frame first; if weak, try variants on the same frame.
- best  : search nearby frames and variants, choose best. Best quality, slower.

This script imports feature/extractor code from live_bisindo_mp_real_shoulder_v6.py
so the output features match the live pipeline.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import mediapipe as mp

try:
    import imageio.v2 as imageio
except Exception:
    imageio = None

from live_bisindo_mp_real_shoulder_v6 import (
    FEATURE_DIMS,
    UltraMediaPipeExtractor,
    center_crop,
    draw_shoulders,
    draw_simple_hand,
)

mp_hands = mp.solutions.hands


VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


def resize_to(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    if frame.shape[1] == width and frame.shape[0] == height:
        return frame
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def resize_width(frame: np.ndarray, width: int) -> np.ndarray:
    if width <= 0 or frame.shape[1] == width:
        return frame
    h, w = frame.shape[:2]
    sc = width / float(w)
    return cv2.resize(frame, (width, int(round(h * sc))), interpolation=cv2.INTER_AREA)


def lap_var(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def frame_stats(frame: np.ndarray) -> Dict[str, float]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean = float(gray.mean())
    std = float(gray.std())
    sharp = lap_var(gray)
    return {"mean": mean, "std": std, "sharp": sharp}


def adjust_gamma(frame: np.ndarray, gamma: float) -> np.ndarray:
    gamma = max(float(gamma), 1e-3)
    inv = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv * 255 for i in range(256)]).astype(np.uint8)
    return cv2.LUT(frame, table)


def clahe_bgr(frame: np.ndarray, clip_limit: float = 2.0, tile_grid: int = 8) -> np.ndarray:
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(tile_grid, tile_grid))
    l2 = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l2, a, b)), cv2.COLOR_LAB2BGR)


def unsharp(frame: np.ndarray, amount: float = 0.65, sigma: float = 1.0) -> np.ndarray:
    blur = cv2.GaussianBlur(frame, (0, 0), sigmaX=float(sigma), sigmaY=float(sigma))
    return cv2.addWeighted(frame, 1.0 + float(amount), blur, -float(amount), 0)


def denoise(frame: np.ndarray) -> np.ndarray:
    # Mild denoise; not too strong because finger edges are important.
    return cv2.fastNlMeansDenoisingColored(frame, None, 3, 3, 7, 21)


def enhance_frame(frame: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return frame
    if mode == "gamma_bright":
        return adjust_gamma(frame, 0.75)
    if mode == "gamma_dark":
        return adjust_gamma(frame, 1.25)
    if mode == "clahe":
        return clahe_bgr(frame, 2.0, 8)
    if mode == "clahe_sharp":
        return unsharp(clahe_bgr(frame, 2.0, 8), 0.55, 1.0)
    if mode == "sharp":
        return unsharp(frame, 0.75, 1.0)
    if mode == "denoise_clahe_sharp":
        return unsharp(clahe_bgr(denoise(frame), 2.0, 8), 0.45, 1.0)
    if mode == "auto":
        st = frame_stats(frame)
        out = frame
        # dark or low contrast video
        if st["mean"] < 85:
            out = adjust_gamma(out, 0.72)
            out = clahe_bgr(out, 2.2, 8)
        elif st["mean"] > 185:
            out = adjust_gamma(out, 1.20)
            out = clahe_bgr(out, 1.6, 8)
        elif st["std"] < 42:
            out = clahe_bgr(out, 2.0, 8)

        # mild sharpening if blurred/soft.
        st2 = frame_stats(out)
        if st2["sharp"] < 90:
            out = unsharp(out, 0.70, 1.0)
        elif st2["sharp"] < 150:
            out = unsharp(out, 0.45, 1.0)
        return out
    raise ValueError(f"Unknown enhance mode: {mode}")


def draw_status(img: np.ndarray, text: str, y: int = 18) -> None:
    cv2.putText(img, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 3, cv2.LINE_AA)
    cv2.putText(img, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (245, 245, 245), 1, cv2.LINE_AA)


class HandProbe:
    """Lightweight probe to score whether a frame is likely to be readable by MediaPipe Hands."""

    def __init__(self, proc_width: int, det_conf: float, track_conf: float, model_complexity: int = 0):
        self.proc_width = int(proc_width)
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=int(model_complexity),
            min_detection_confidence=float(det_conf),
            min_tracking_confidence=float(track_conf),
        )

    def close(self) -> None:
        self.hands.close()

    def score(self, frame_bgr: np.ndarray) -> Dict[str, float]:
        proc = resize_width(frame_bgr, self.proc_width)
        rgb = cv2.cvtColor(proc, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        res = self.hands.process(rgb)

        st = frame_stats(proc)
        n = 0
        avg_conf = 0.0
        avg_area = 0.0

        if res.multi_hand_landmarks:
            n = len(res.multi_hand_landmarks)
            if res.multi_handedness:
                vals = []
                for h in res.multi_handedness:
                    if h.classification:
                        vals.append(float(h.classification[0].score or 0.0))
                avg_conf = float(np.mean(vals)) if vals else 0.8

            areas = []
            for lms in res.multi_hand_landmarks:
                xy = np.array([[p.x, p.y] for p in lms.landmark], dtype=np.float32)
                mn = xy.min(axis=0)
                mx = xy.max(axis=0)
                areas.append(float((mx[0] - mn[0]) * (mx[1] - mn[1])))
            avg_area = float(np.mean(areas)) if areas else 0.0

        # Score prioritizes actual detection, then confidence/hand size, then visual quality.
        # Sharpness contribution is capped so a sharp non-hand frame cannot beat a detected hand.
        sharp_bonus = min(st["sharp"] / 250.0, 1.0) * 0.10
        contrast_bonus = min(st["std"] / 70.0, 1.0) * 0.08
        score = (n * 2.0) + avg_conf + (avg_area * 6.0) + sharp_bonus + contrast_bonus

        return {
            "score": float(score),
            "hands": float(n),
            "conf": float(avg_conf),
            "area": float(avg_area),
            "mean": st["mean"],
            "std": st["std"],
            "sharp": st["sharp"],
        }


def save_gif(frames: Sequence[np.ndarray], path: Path, fps: float, max_width: int) -> None:
    if imageio is None:
        print("[WARN] imageio unavailable; skip GIF:", path)
        return
    if not frames:
        return
    out = []
    for bgr in frames:
        h, w = bgr.shape[:2]
        if max_width > 0 and w > max_width:
            sc = max_width / float(w)
            bgr = cv2.resize(bgr, (max_width, int(round(h * sc))), interpolation=cv2.INTER_AREA)
        out.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    imageio.mimsave(path, out, duration=1.0 / max(float(fps), 1e-6), loop=0)


def save_mp4(frames: Sequence[np.ndarray], path: Path, fps: float) -> None:
    if not frames:
        return
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (w, h))
    for f in frames:
        if f.shape[:2] != (h, w):
            f = cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA)
        writer.write(f)
    writer.release()


def read_frame_at(cap: cv2.VideoCapture, frame_idx_1based: int) -> Optional[np.ndarray]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_idx_1based - 1))
    ok, frame = cap.read()
    return frame if ok and frame is not None else None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Smart extractor for old recorded BISINDO videos")
    p.add_argument("video", nargs="?", type=str, help="Input video path. Omit if using --batch-dir.")
    p.add_argument("--batch-dir", type=str, default=None, help="Process all videos in this directory recursively.")
    p.add_argument("--out-dir", type=str, default=None)

    p.add_argument("--feature-mode", choices=list(FEATURE_DIMS), default="btj_global_local")
    p.add_argument("--target-fps", type=float, default=10.0)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--center-crop", type=float, default=1.0, help="Use 1.0 for old videos to avoid cutting hands.")
    p.add_argument("--proc-width", type=int, default=320)
    p.add_argument("--shoulder-backend", choices=["none", "mp-pose"], default="mp-pose")
    p.add_argument("--pose-proc-width", type=int, default=224)
    p.add_argument("--pose-every", type=int, default=5)
    p.add_argument("--hand-model-complexity", type=int, choices=[0, 1], default=0)
    p.add_argument("--pose-model-complexity", type=int, choices=[0, 1], default=0)
    p.add_argument("--det-conf", type=float, default=0.45)
    p.add_argument("--track-conf", type=float, default=0.45)
    p.add_argument("--smooth-alpha", type=float, default=0.82)
    p.add_argument("--shoulder-smooth-alpha", type=float, default=0.35)
    p.add_argument("--hold-frames", type=int, default=4)
    p.add_argument("--mirror-input", action="store_true")
    p.add_argument("--no-mirror-handedness", action="store_true")

    p.add_argument("--smart-mode", choices=["fast", "smart", "best"], default="smart")
    p.add_argument("--enhance", choices=[
        "auto", "none", "gamma_bright", "gamma_dark", "clahe", "clahe_sharp", "sharp", "denoise_clahe_sharp"
    ], default="auto")
    p.add_argument("--fallback-variants", type=str, default="auto,clahe_sharp,gamma_bright,sharp,none",
                   help="Comma-separated enhancement variants used in smart/best mode.")
    p.add_argument("--search-radius", type=int, default=2,
                   help="For best mode: search +/- N source frames around target.")
    p.add_argument("--weak-hand-threshold", type=float, default=1.5,
                   help="If probe score below this, try fallback variants in smart mode.")

    p.add_argument("--gif-width", type=int, default=420)
    p.add_argument("--save-gif", action="store_true", default=True)
    p.add_argument("--no-gif", action="store_true")
    p.add_argument("--save-mp4", action="store_true", help="Also save overlay/skeleton mp4.")
    p.add_argument("--skeleton-bg", choices=["black", "white"], default="black")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def list_videos(batch_dir: Path) -> List[Path]:
    return sorted([p for p in batch_dir.rglob("*") if p.suffix.lower() in VIDEO_EXTS])


def choose_frame_and_enhancement(
    cap: cv2.VideoCapture,
    base_frame_idx: int,
    target_frame: np.ndarray,
    args: argparse.Namespace,
    probe: HandProbe,
) -> Tuple[np.ndarray, int, str, Dict[str, float]]:
    """Return enhanced frame, chosen source frame index, enhancement mode, score info."""

    def prepare(raw: np.ndarray, mode: str) -> np.ndarray:
        raw = resize_to(raw, args.width, args.height)
        raw = center_crop(raw, args.center_crop)
        return enhance_frame(raw, mode)

    variants = [v.strip() for v in args.fallback_variants.split(",") if v.strip()]
    if args.enhance not in variants:
        variants = [args.enhance] + variants

    target_pre = prepare(target_frame, args.enhance)

    if args.smart_mode == "fast":
        return target_pre, base_frame_idx, args.enhance, probe.score(target_pre)

    # First probe preferred enhancement.
    best_frame = target_pre
    best_idx = base_frame_idx
    best_mode = args.enhance
    best_score = probe.score(target_pre)

    if args.smart_mode == "smart":
        if best_score["score"] >= args.weak_hand_threshold:
            return best_frame, best_idx, best_mode, best_score

        # Same frame, multiple enhancements.
        for mode in variants:
            cand = prepare(target_frame, mode)
            sc = probe.score(cand)
            if sc["score"] > best_score["score"]:
                best_frame, best_idx, best_mode, best_score = cand, base_frame_idx, mode, sc
        return best_frame, best_idx, best_mode, best_score

    # best mode: search neighbor frames and enhancement variants.
    offsets = [0]
    for k in range(1, args.search_radius + 1):
        offsets.extend([-k, k])

    raw_cache: Dict[int, np.ndarray] = {base_frame_idx: target_frame}
    for off in offsets:
        idx = max(1, base_frame_idx + off)
        raw = raw_cache.get(idx)
        if raw is None:
            raw = read_frame_at(cap, idx)
            if raw is None:
                continue
            raw_cache[idx] = raw
        for mode in variants:
            cand = prepare(raw, mode)
            sc = probe.score(cand)
            # tiny penalty for choosing far neighbor, so we don't jump unnecessarily.
            sc_adj = dict(sc)
            sc_adj["score"] = float(sc["score"] - 0.03 * abs(off))
            if sc_adj["score"] > best_score["score"]:
                best_frame, best_idx, best_mode, best_score = cand, idx, mode, sc
    return best_frame, best_idx, best_mode, best_score


def process_one_video(video_path: Path, args: argparse.Namespace, out_root: Optional[Path] = None) -> Optional[Path]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERR] Cannot open: {video_path}")
        return None

    src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if src_fps <= 1e-6:
        src_fps = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_sec = (total_frames / src_fps) if total_frames > 0 else 0.0

    if out_root is not None:
        out_dir = out_root / video_path.stem
    else:
        out_dir = Path(args.out_dir) if args.out_dir else video_path.parent / f"{video_path.stem}_smart_v8"
    out_dir.mkdir(parents=True, exist_ok=True)

    extractor = UltraMediaPipeExtractor(
        feature_mode=args.feature_mode,
        shoulder_backend=args.shoulder_backend,
        proc_width=args.proc_width,
        pose_proc_width=args.pose_proc_width,
        pose_every=args.pose_every,
        hand_every=1,
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
    probe = HandProbe(args.proc_width, args.det_conf, args.track_conf, args.hand_model_complexity)

    overlay_frames: List[np.ndarray] = []
    skeleton_frames: List[np.ndarray] = []
    features: List[np.ndarray] = []
    meta: List[dict] = []

    target_fps = float(args.target_fps)
    target_dt = 1.0 / max(target_fps, 1e-6)
    n_targets = int(math.floor(duration_sec * target_fps + 1e-6)) + 1 if duration_sec > 0 else total_frames

    if not args.quiet:
        print("=" * 72)
        print(f"[VIDEO] {video_path}")
        print(f"src_fps={src_fps:.3f} frames={total_frames} duration={duration_sec:.2f}s")
        print(f"target_fps={target_fps:.2f} targets≈{n_targets} smart={args.smart_mode} mode={args.feature_mode}")
        print("=" * 72)

    t0 = time.perf_counter()
    try:
        for out_i in range(n_targets):
            t_sec = out_i * target_dt
            base_idx = int(round(t_sec * src_fps)) + 1
            if total_frames and base_idx > total_frames:
                break
            raw = read_frame_at(cap, base_idx)
            if raw is None:
                break

            work, chosen_idx, enh_mode, score_info = choose_frame_and_enhancement(
                cap=cap,
                base_frame_idx=base_idx,
                target_frame=raw,
                args=args,
                probe=probe,
            )

            res = extractor.process(work)

            overlay = work.copy()
            draw_shoulders(overlay, res.shoulders)
            draw_simple_hand(overlay, res.left, (0, 255, 0))
            draw_simple_hand(overlay, res.right, (0, 180, 255))
            draw_status(
                overlay,
                f"{video_path.name} | {args.feature_mode}:{FEATURE_DIMS[args.feature_mode]} | {target_fps:g}fps | {args.smart_mode}/{enh_mode}",
                18,
            )
            draw_status(
                overlay,
                f"t={t_sec:.2f}s target={base_idx} chosen={chosen_idx} probe_hands={score_info['hands']:.0f} score={score_info['score']:.2f}",
                38,
            )

            if args.skeleton_bg == "white":
                skeleton = np.full_like(work, 255)
                left_col, right_col, text_col = (0, 130, 0), (0, 90, 210), (40, 40, 40)
            else:
                skeleton = np.zeros_like(work)
                left_col, right_col, text_col = (0, 255, 0), (0, 180, 255), (220, 220, 220)
            draw_shoulders(skeleton, res.shoulders)
            draw_simple_hand(skeleton, res.left, left_col)
            draw_simple_hand(skeleton, res.right, right_col)
            cv2.putText(skeleton, f"skeleton | {video_path.name}", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, text_col, 1, cv2.LINE_AA)

            overlay_frames.append(overlay)
            skeleton_frames.append(skeleton)
            features.append(res.vector.astype(np.float32))
            meta.append({
                "out_index": int(out_i),
                "time_sec": float(t_sec),
                "target_source_frame": int(base_idx),
                "chosen_source_frame": int(chosen_idx),
                "enhance_mode": enh_mode,
                "probe": score_info,
                "left_present": float(res.present[0]),
                "right_present": float(res.present[1]),
                "left_detected": float(res.detected[0]),
                "right_detected": float(res.detected[1]),
                "left_held": float(res.held[0]),
                "right_held": float(res.held[1]),
                "left_score": float(res.scores[0]),
                "right_score": float(res.scores[1]),
                "hand_ms": float(res.hand_ms),
                "pose_ms": float(res.pose_ms),
            })

            if not args.quiet and (out_i + 1) % 20 == 0:
                elapsed = time.perf_counter() - t0
                print(f"[PROGRESS] {out_i+1}/{n_targets} t={t_sec:.1f}s chosen={chosen_idx} enh={enh_mode} wall={elapsed:.1f}s")

    finally:
        cap.release()
        extractor.close()
        probe.close()

    if not features:
        print(f"[ERR] No features extracted: {video_path}")
        return None

    arr = np.stack(features).astype(np.float32)
    ts = time.strftime("%Y%m%d_%H%M%S")
    base = f"{video_path.stem}_{args.feature_mode}_{int(round(target_fps))}fps_{args.smart_mode}_{ts}"

    npz_path = out_dir / f"{base}.npz"
    csv_path = out_dir / f"{base}.csv"
    json_path = out_dir / f"{base}_meta.json"
    overlay_gif = out_dir / f"{base}_overlay.gif"
    skeleton_gif = out_dir / f"{base}_skeleton_only.gif"
    overlay_mp4 = out_dir / f"{base}_overlay.mp4"
    skeleton_mp4 = out_dir / f"{base}_skeleton_only.mp4"

    np.savez_compressed(npz_path, features=arr, feature_mode=args.feature_mode, feature_dim=arr.shape[1], target_fps=target_fps)

    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time_sec", "target_source_frame", "chosen_source_frame", "enhance_mode"] + [f"f{i}" for i in range(arr.shape[1])])
        for m, row in zip(meta, arr):
            w.writerow([m["time_sec"], m["target_source_frame"], m["chosen_source_frame"], m["enhance_mode"], *row.tolist()])

    with json_path.open("w") as f:
        json.dump({
            "input_video": str(video_path),
            "source_fps": src_fps,
            "source_frame_count": total_frames,
            "source_duration_sec": duration_sec,
            "target_fps": target_fps,
            "feature_mode": args.feature_mode,
            "feature_dim": int(arr.shape[1]),
            "smart_mode": args.smart_mode,
            "settings": vars(args),
            "frames": meta,
        }, f, indent=2)

    if args.save_gif and not args.no_gif:
        save_gif(overlay_frames, overlay_gif, target_fps, args.gif_width)
        save_gif(skeleton_frames, skeleton_gif, target_fps, args.gif_width)

    if args.save_mp4:
        save_mp4(overlay_frames, overlay_mp4, target_fps)
        save_mp4(skeleton_frames, skeleton_mp4, target_fps)

    if not args.quiet:
        print("=" * 72)
        print("[DONE]", video_path.name)
        print("features:", arr.shape, "->", npz_path)
        if args.save_gif and not args.no_gif:
            print("overlay gif :", overlay_gif)
            print("skeleton gif:", skeleton_gif)
        print("metadata    :", json_path)
        print("=" * 72)

    return npz_path


def main() -> None:
    args = parse_args()

    if args.batch_dir:
        batch_dir = Path(args.batch_dir)
        videos = list_videos(batch_dir)
        if not videos:
            raise FileNotFoundError(f"No videos found in {batch_dir}")
        out_root = Path(args.out_dir) if args.out_dir else batch_dir / f"_smart_extract_v8_{int(round(args.target_fps))}fps"
        out_root.mkdir(parents=True, exist_ok=True)
        print(f"[BATCH] found {len(videos)} videos -> {out_root}")
        ok = 0
        fail = 0
        for i, vp in enumerate(videos, 1):
            print(f"\n[BATCH] {i}/{len(videos)} {vp}")
            try:
                r = process_one_video(vp, args, out_root=out_root)
                ok += int(r is not None)
                fail += int(r is None)
            except Exception as e:
                fail += 1
                print(f"[FAIL] {vp}: {type(e).__name__}: {e}")
        print(f"[BATCH DONE] ok={ok} fail={fail} out={out_root}")
        return

    if not args.video:
        raise SystemExit("Provide a video path or use --batch-dir")
    process_one_video(Path(args.video), args)


if __name__ == "__main__":
    main()
