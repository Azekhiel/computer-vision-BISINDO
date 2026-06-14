"""Tests for the studio holistic_stabilized glue (mediapipe_extract._stabilize_hand_arrays)."""

from __future__ import annotations

import os
import sys

import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from bisindo_dataset_studio import constants as C  # noqa: E402
from bisindo_dataset_studio import mediapipe_extract as me  # noqa: E402


def _base_data(total: int = 8):
    masks = np.ones((total, 9), dtype=bool)
    pose = np.zeros((total, 33, 5), dtype=np.float32)
    pose[:, 11, :2] = [0.4, 0.5]
    pose[:, 12, :2] = [0.6, 0.5]  # shoulder width 0.2
    left = np.zeros((total, 21, 5), dtype=np.float32)
    right = np.zeros((total, 21, 5), dtype=np.float32)
    for t in range(total):
        left[t, :, :3] = 0.45 + 0.005 * t
    right[:, :, :3] = 0.6
    return {
        "availability_masks": masks,
        "pose_landmarks": pose,
        "left_hand_landmarks": left,
        "right_hand_landmarks": right,
        "timestamp_ms": np.arange(total, dtype=np.float32) * 100.0,
        "source_fps": np.asarray([10.0], dtype=np.float32),
    }


def test_method_registered_in_constants():
    assert C.MEDIAPIPE_METHOD_HOLISTIC_STABILIZED == "holistic_stabilized"
    assert C.MEDIAPIPE_METHOD_HOLISTIC_STABILIZED in C.MEDIAPIPE_METHODS
    assert C.MEDIAPIPE_FEATURE_ROOTS[C.MEDIAPIPE_METHOD_HOLISTIC_STABILIZED] == "features_holistic_stabilized"


def test_tasks_config_defaults_to_bundled_models_face_optional():
    cfg = me._tasks_config_from_config(None)
    # pose + hand ship in test_mediapipe/ -> resolved to existing files
    assert cfg["pose_task_model_path"].endswith("pose_landmarker_full.task")
    assert cfg["hand_task_model_path"].endswith("hand_landmarker.task")
    assert os.path.exists(cfg["pose_task_model_path"])
    assert os.path.exists(cfg["hand_task_model_path"])
    # face_landmarker.task is not shipped -> face stays optional (empty unless present)
    if not (me.TASKS_MODEL_DIR / "face_landmarker.task").exists():
        assert cfg["face_task_model_path"] == ""


def test_tasks_config_respects_explicit_paths():
    cfg = me._tasks_config_from_config({"face_task_model_path": "/custom/face.task"})
    assert cfg["face_task_model_path"] == "/custom/face.task"


def test_stabilizer_rejects_jump_and_holds_previous():
    data = _base_data()
    data["left_hand_landmarks"][5, :, :3] = 0.95  # teleport on frame 5
    stats = me._stabilize_hand_arrays(data, {})
    assert stats["n_rejected_jumps"] >= 1
    # held value stays near the pre-jump position, not the teleport
    assert float(data["left_hand_landmarks"][5, 0, 0]) < 0.6


def test_stabilizer_covers_short_gap_marks_present():
    data = _base_data()
    data["availability_masks"][3, 2] = False  # left hand missing on frame 3 (short gap)
    me._stabilize_hand_arrays(data, {})
    # anti-missing: every left-hand frame is present again (held/filled)
    assert data["availability_masks"][:, 2].all()


def test_stabilizer_drops_truly_long_gap():
    data = _base_data(total=16)
    # left hand absent for frames 2..14 (13-frame gap > hold_frames + max_gap = 9)
    data["availability_masks"][2:15, 2] = False
    me._stabilize_hand_arrays(data, {})
    left_present = data["availability_masks"][:, 2]
    assert bool(left_present[3])       # near edge: recovered by hold
    assert not bool(left_present[10])  # deep in the gap: unrecoverable


def test_stabilizer_preserves_other_hand_and_returns_counts():
    data = _base_data()
    right_before = data["right_hand_landmarks"][:, :, :3].copy()
    stats = me._stabilize_hand_arrays(data, {})
    assert set(stats) == {"n_rejected_jumps", "n_filled_frames"}
    # right hand was clean and continuous -> essentially unchanged (1euro near-identity on constant)
    np.testing.assert_allclose(data["right_hand_landmarks"][:, 0, :3], right_before[:, 0, :3], atol=1e-3)
