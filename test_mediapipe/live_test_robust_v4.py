#!/usr/bin/env python3
"""
live_test_robust_v4.py
======================
Live webcam test using arm-locked + anti-jitter + palm/shoulder feature modes.

Recommended:
    python3 live_test_robust_v4.py --cam 0 --fps 30 --feature-mode compat
    python3 live_test_robust_v4.py --cam 0 --fps 30 --model-complexity 1 --feature-mode compat

Keys:
    Q/ESC = quit, G = save gif, S = save current feature vector, R = reset tracker/buffer
"""

from __future__ import annotations

import argparse
import collections
import os
import sys
import threading
import time
from typing import Optional, Tuple

import cv2
import imageio
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from feature_extractor_robust_v4 import make_tracker, extract_features, draw_landmarks, draw_hud


class LatestFrameCamera:
    """Camera reader that always serves the newest frame.

    This prevents lag from accumulating when inference is slower than camera FPS.
    """

    def __init__(
        self,
        src: int = 0,
        use_csi: bool = False,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        mjpeg: bool = True,
    ):
        self.width, self.height, self.fps = width, height, fps
        self.cap = self._open(src, use_csi, width, height, fps, mjpeg)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera src={src} csi={use_csi}")

        self.lock = threading.Lock()
        self.frame: Optional[np.ndarray] = None
        self.ret = False
        self.running = True
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

        t0 = time.perf_counter()
        while self.frame is None and time.perf_counter() - t0 < 2.0:
            time.sleep(0.01)

    def _open(self, src: int, use_csi: bool, width: int, height: int, fps: int, mjpeg: bool):
        if use_csi:
            gst = (
                f"nvarguscamerasrc sensor-id={src} ! "
                f"video/x-raw(memory:NVMM), width={width}, height={height}, format=NV12, framerate={fps}/1 ! "
                f"nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! "
                f"video/x-raw, format=BGR ! appsink max-buffers=1 drop=true sync=false"
            )
            cap = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
        else:
            # Try MJPEG first because many USB cameras deliver 640x480@30 more reliably with MJPEG.
            # If it fails, fall back to raw video/x-raw and then plain V4L2.
            if mjpeg:
                gst = (
                    f"v4l2src device=/dev/video{src} ! "
                    f"image/jpeg, width={width}, height={height}, framerate={fps}/1 ! "
                    f"jpegdec ! videoconvert ! video/x-raw, format=BGR ! "
                    f"appsink max-buffers=1 drop=true sync=false"
                )
                cap = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
                if cap.isOpened():
                    return cap

            gst = (
                f"v4l2src device=/dev/video{src} ! "
                f"video/x-raw, width={width}, height={height}, framerate={fps}/1 ! "
                f"videoconvert ! video/x-raw, format=BGR ! appsink max-buffers=1 drop=true sync=false"
            )
            cap = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
            if not cap.isOpened():
                print("[WARN] GStreamer V4L2 failed, fallback to cv2.CAP_V4L2")
                cap = cv2.VideoCapture(src, cv2.CAP_V4L2)
                if mjpeg:
                    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                cap.set(cv2.CAP_PROP_FPS, fps)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def _reader(self) -> None:
        while self.running:
            ret, frame = self.cap.read()
            if ret and frame is not None:
                with self.lock:
                    self.ret = True
                    self.frame = frame
            else:
                time.sleep(0.004)

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        with self.lock:
            if not self.ret or self.frame is None:
                return False, None
            return True, self.frame.copy()

    def release(self) -> None:
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)
        self.cap.release()


def frames_to_gif(frames, path: str, fps: int = 15, max_width: int = 360) -> None:
    if not frames:
        print("[GIF] Buffer kosong.")
        return
    out = []
    for bgr in frames:
        h, w = bgr.shape[:2]
        if w > max_width:
            scale = max_width / w
            bgr = cv2.resize(bgr, (max_width, int(h * scale)), interpolation=cv2.INTER_AREA)
        out.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    imageio.mimsave(path, out, duration=1.0 / fps, loop=0)
    print(f"[GIF] Saved {len(out)} frames -> {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Robust live hand/pose test V3")
    parser.add_argument("--cam", type=int, default=0)
    parser.add_argument("--csi", action="store_true")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--no-mjpeg", action="store_true", help="Disable MJPEG camera mode for USB cameras.")
    parser.add_argument("--model-complexity", type=int, default=0, choices=[0, 1], help="1 gives better pose skeleton but lower FPS.")
    parser.add_argument("--det-conf", type=float, default=0.55)
    parser.add_argument("--track-conf", type=float, default=0.55)
    parser.add_argument("--pose-every", type=int, default=1, help="Run pose every N frames. 1 = best arm lock/skeleton.")
    parser.add_argument("--hold-frames", type=int, default=4, help="Keep last hand landmarks for this many missed frames.")
    parser.add_argument("--arm-hold-frames", type=int, default=8, help="Keep last arm anchors for this many missed pose frames.")
    parser.add_argument("--arm-gate", type=float, default=0.13, help="Max normalized distance hand-wrist may be from pose-wrist.")
    parser.add_argument("--hand-deadband", type=float, default=0.0065, help="Freeze hand landmark micro-jitter below this normalized distance.")
    parser.add_argument("--pose-deadband", type=float, default=0.0045, help="Freeze arm anchor micro-jitter below this normalized distance.")
    parser.add_argument("--snap-strength", type=float, default=0.35, help="0 disables wrist snap; 0.25-0.45 is usually safe.")
    parser.add_argument("--snap-max", type=float, default=0.030, help="Max normalized snap shift per frame.")
    parser.add_argument("--no-snap-hand-to-arm", action="store_true", help="Disable gentle hand wrist snap to arm wrist.")
    parser.add_argument("--no-arm-lock", action="store_true", help="Disable strict hand-to-arm assignment. Not recommended for BISINDO/self-handshake.")
    parser.add_argument("--allow-unanchored", action="store_true", help="Allow hand assignment when pose wrist is missing. More permissive, can glitch.")
    parser.add_argument("--feature-mode", choices=["compat", "stable", "palm", "palm_angles"], default="compat")
    parser.add_argument("--no-mirror-handedness", action="store_true", help="Use if L/R labels look inverted on your camera.")
    parser.add_argument("--draw-full-pose", action="store_true", help="Draw full MediaPipe pose including face/body. Off by default.")
    parser.add_argument("--draw-raw", action="store_true", help="Draw raw hand detections before assignment.")
    parser.add_argument("--no-draw-gates", action="store_true")
    parser.add_argument("--gif-sec", type=float, default=5.0)
    parser.add_argument("--gif-out", type=str, default="live_test_robust_v3.gif")
    parser.add_argument("--no-gif-buffer", action="store_true", help="Do not copy frames into GIF ring buffer.")
    parser.add_argument("--no-display", action="store_true")
    args = parser.parse_args()

    ring = None if args.no_gif_buffer else collections.deque(maxlen=max(1, int(args.gif_sec * args.fps)))
    cam = LatestFrameCamera(args.cam, args.csi, args.width, args.height, args.fps, mjpeg=not args.no_mjpeg)
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

    print("=" * 72)
    print("Robust live test V4 opened")
    print(f"camera={args.cam} csi={args.csi} size={args.width}x{args.height} fps={args.fps} mjpeg={not args.no_mjpeg}")
    print(f"feature_mode={args.feature_mode} model_complexity={args.model_complexity} pose_every={args.pose_every}")
    print(f"arm_lock={not args.no_arm_lock} arm_gate={args.arm_gate} allow_unanchored={args.allow_unanchored}")
    print(f"anti_jitter hand_deadband={args.hand_deadband} pose_deadband={args.pose_deadband}")
    print(f"snap_strength={0.0 if args.no_snap_hand_to_arm else args.snap_strength} snap_max={args.snap_max}")
    print("Keys: Q/ESC quit | G save GIF | S save feature .npy | R reset tracker/buffer")
    print("=" * 72)

    fps_counter = 0
    fps_t0 = time.perf_counter()
    display_fps = 0.0
    last_feat = None

    try:
        while True:
            ret, frame = cam.read()
            if not ret or frame is None:
                time.sleep(0.002)
                continue

            if frame.shape[1] != args.width or frame.shape[0] != args.height:
                frame = cv2.resize(frame, (args.width, args.height), interpolation=cv2.INTER_AREA)

            result = tracker.process(frame)
            last_feat = extract_features(result, mode=args.feature_mode)

            vis = draw_landmarks(
                frame.copy(),
                result,
                draw_full_pose=args.draw_full_pose,
                draw_gates=not args.no_draw_gates,
                draw_raw_detections=args.draw_raw,
            )
            vis = draw_hud(vis, result, display_fps, feature_mode=args.feature_mode)
            if ring is not None:
                ring.append(vis.copy())

            if not args.no_display:
                cv2.imshow("Robust Hand Live V4", vis)

            fps_counter += 1
            now = time.perf_counter()
            if now - fps_t0 >= 0.5:
                display_fps = fps_counter / (now - fps_t0)
                fps_counter = 0
                fps_t0 = now

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if key in (ord("g"), ord("G")):
                if ring is None:
                    print("[GIF] Disabled because --no-gif-buffer is active.")
                else:
                    frames_to_gif(list(ring), args.gif_out, fps=15)
            if key in (ord("s"), ord("S")) and last_feat is not None:
                out = f"live_feat_{args.feature_mode}_v4.npy"
                np.save(out, last_feat.astype(np.float32))
                print(f"[SNAP] Saved {last_feat.shape} -> {out}")
            if key in (ord("r"), ord("R")):
                if ring is not None:
                    ring.clear()
                tracker.reset()
                print("[TRACKER] reset")
    except KeyboardInterrupt:
        print("\n[INFO] stopped")
    finally:
        cam.release()
        tracker.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
