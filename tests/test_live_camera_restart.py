import os
import queue
import sys
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import live_gru_fast
from smart_extract.live_bisindo_mp_real_shoulder_v6 import LatestFrameCamera


class _FakeCap:
    """Stand-in for cv2.VideoCapture that records teardown ordering."""

    def __init__(self, order: list[str], owner_box: dict, read_delay: float = 0.001):
        self.order = order
        self.owner_box = owner_box
        self.read_delay = read_delay
        self.released = False

    def isOpened(self) -> bool:
        return not self.released

    def read(self):
        time.sleep(self.read_delay)
        if self.released:
            return False, None
        return True, np.zeros((4, 4, 3), dtype=np.uint8)

    def release(self) -> None:
        self.released = True
        cam = self.owner_box.get("cam")
        reader_alive = bool(cam is not None and cam.thread is not None and cam.thread.is_alive())
        self.order.append("cap_release_reader_alive" if reader_alive else "cap_release")


def _make_camera(order: list[str]) -> LatestFrameCamera:
    owner_box: dict = {}
    fake = _FakeCap(order, owner_box)
    original_open = LatestFrameCamera._open
    LatestFrameCamera._open = lambda self: fake
    try:
        cam = LatestFrameCamera(src=0, width=4, height=4, fps=30, use_gstreamer=False)
    finally:
        LatestFrameCamera._open = original_open
    owner_box["cam"] = cam
    return cam


def test_release_joins_reader_before_cap_release():
    order: list[str] = []
    cam = _make_camera(order)
    cam.start(require_frame=True, timeout=2.0)
    assert cam.read()[0]

    info = cam.release(join_timeout=2.0)

    assert order == ["cap_release"], f"reader thread must be joined before cap.release(); got {order}"
    assert info["camera_released"] is True
    assert info["camera_thread_alive_after_release"] is False
    assert cam.cap is None and cam.thread is None


def test_start_stop_start_cycle_serves_frames_again():
    for _ in range(2):
        order: list[str] = []
        cam = _make_camera(order)
        cam.start(require_frame=True, timeout=2.0)
        ok, frame = cam.read()
        assert ok and frame is not None
        cam.release(join_timeout=2.0)
        ok, frame = cam.read()
        assert not ok and frame is None


def test_release_is_idempotent():
    order: list[str] = []
    cam = _make_camera(order)
    cam.start(require_frame=True, timeout=2.0)
    first = cam.release(join_timeout=2.0)
    second = cam.release(join_timeout=2.0)
    assert order == ["cap_release"]
    assert second == first


def test_request_stop_only_signals():
    order: list[str] = []
    cam = _make_camera(order)
    cam.start(require_frame=True, timeout=2.0)
    cam.request_stop()
    assert cam.running is False
    assert cam.cap is not None and not cam.cap.released
    cam.release(join_timeout=2.0)


def _bare_worker() -> live_gru_fast.FastGRULiveWorker:
    return live_gru_fast.FastGRULiveWorker(
        variant="adi",
        status_queue=queue.Queue(),
        show_window=False,
    )


def test_worker_stop_is_signal_only():
    worker = _bare_worker()
    camera = SimpleNamespace(request_stop=Mock(), release=Mock(), running=True)
    worker._camera = camera

    worker.stop()

    assert worker._stop_event.is_set()
    camera.request_stop.assert_called_once_with()
    camera.release.assert_not_called()


def test_worker_stop_falls_back_to_running_flag():
    worker = _bare_worker()
    camera = SimpleNamespace(release=Mock(), running=True)
    worker._camera = camera

    worker.stop()

    assert camera.running is False
    camera.release.assert_not_called()


def test_force_cleanup_releases_when_worker_stuck():
    worker = _bare_worker()
    camera = SimpleNamespace(request_stop=Mock(), release=Mock(return_value={}), running=True)
    worker._camera = camera

    worker.force_cleanup()

    camera.release.assert_called_once()


def test_force_cleanup_skips_release_after_clean_finish():
    worker = _bare_worker()
    camera = SimpleNamespace(request_stop=Mock(), release=Mock(return_value={}), running=True)
    worker._camera = camera
    worker.finished_event.set()

    worker.force_cleanup()

    camera.release.assert_not_called()


def test_open_camera_with_retry_pushes_retry_events(monkeypatch):
    worker = _bare_worker()
    fail_count = {"n": 0}

    class _FlakyCamera:
        def __init__(self, **kwargs):
            fail_count["n"] += 1
            if fail_count["n"] <= 2:
                raise RuntimeError("device busy")

        def start(self, require_frame: bool = False, timeout: float = 2.0):
            return self

        def release(self, join_timeout: float = 3.0):
            return {}

    monkeypatch.setattr(live_gru_fast, "LatestFrameCamera", _FlakyCamera)

    profile = {"width": 4, "height": 4, "camera_fps": 30, "use_gstreamer": 0, "camera_start_timeout": 0.1}
    t0 = time.perf_counter()
    cap = worker._open_camera_with_retry(profile, attempts=3)
    elapsed = time.perf_counter() - t0

    assert cap is not None
    assert fail_count["n"] == 3
    events = []
    while not worker.status_queue.empty():
        events.append(worker.status_queue.get())
    retry_events = [e for e in events if e.get("event") == "camera_retry"]
    assert [e["attempt"] for e in retry_events] == [1, 2]
    assert elapsed >= 1.4  # backoff 0.5 + 1.0


def test_open_camera_with_retry_raises_after_exhaustion(monkeypatch):
    worker = _bare_worker()

    class _DeadCamera:
        def __init__(self, **kwargs):
            raise RuntimeError("Cannot open camera src=0")

    monkeypatch.setattr(live_gru_fast, "LatestFrameCamera", _DeadCamera)
    profile = {"width": 4, "height": 4, "camera_fps": 30, "use_gstreamer": 0, "camera_start_timeout": 0.1}
    # Skip waiting between attempts to keep the test fast.
    worker._stop_event.wait = lambda timeout=None: False

    try:
        worker._open_camera_with_retry(profile, attempts=2)
    except RuntimeError as exc:
        assert "failed after 2 attempts" in str(exc)
    else:
        raise AssertionError("expected RuntimeError after exhausting attempts")


def test_open_camera_with_retry_aborts_on_stop(monkeypatch):
    worker = _bare_worker()
    worker._stop_event.set()

    class _NeverCalled:
        def __init__(self, **kwargs):
            raise AssertionError("camera must not be opened after stop")

    monkeypatch.setattr(live_gru_fast, "LatestFrameCamera", _NeverCalled)
    profile = {"width": 4, "height": 4, "camera_fps": 30, "use_gstreamer": 0, "camera_start_timeout": 0.1}
    assert worker._open_camera_with_retry(profile, attempts=3) is None
