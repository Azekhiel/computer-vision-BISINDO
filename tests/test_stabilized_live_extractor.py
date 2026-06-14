"""Integration test for StabilizedHolisticLiveExtractor using a fake Holistic."""

from __future__ import annotations

import os
import sys
import types

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import holistic_features as hf  # noqa: E402
import feature_schemas as fs  # noqa: E402


def _lm(arr, vis=None):
    pts = []
    for i, (x, y, z) in enumerate(arr):
        o = types.SimpleNamespace(x=float(x), y=float(y), z=float(z))
        if vis is not None:
            o.visibility = float(vis[i])
        pts.append(o)
    return types.SimpleNamespace(landmark=pts)


class _FakeHolistic:
    """Returns scripted results so the stabilizer can be exercised offline."""

    def __init__(self, scripted):
        self._scripted = scripted
        self._i = 0

    def process(self, _rgb):
        result = self._scripted[min(self._i, len(self._scripted) - 1)]
        self._i += 1
        return result

    def close(self):
        pass


def _make_result(left_wrist, right_wrist=(0.65, 0.5), left_present=True):
    rng = np.random.default_rng(0)
    pose = rng.uniform(0.3, 0.7, size=(33, 3)).astype(np.float32)
    pose[11] = [0.4, 0.5, 0.0]
    pose[12] = [0.6, 0.5, 0.0]
    face = np.zeros((478, 3), dtype=np.float32)
    left = np.full((21, 3), [left_wrist[0], left_wrist[1], 0.0], dtype=np.float32)
    right = np.full((21, 3), [right_wrist[0], right_wrist[1], 0.0], dtype=np.float32)
    return types.SimpleNamespace(
        pose_landmarks=_lm(pose, vis=np.ones(33)),
        pose_world_landmarks=None,
        face_landmarks=_lm(face),
        left_hand_landmarks=_lm(left) if left_present else None,
        right_hand_landmarks=_lm(right),
    )


def _build_extractor(monkeypatch, scripted):
    monkeypatch.setattr(hf.mp_holistic, "Holistic", lambda **_kw: _FakeHolistic(scripted))
    return hf.StabilizedHolisticLiveExtractor("smart180", refine_face_landmarks=False)


def test_stabilized_live_extractor_outputs_correct_dim(monkeypatch):
    scripted = [_make_result((0.40, 0.5)), _make_result((0.41, 0.5))]
    ext = _build_extractor(monkeypatch, scripted)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    res = ext.process(frame)
    assert res.vector.shape == (fs.get_schema("smart180").feature_dim,)
    assert np.isfinite(res.vector).all()
    assert ext.backend_name == "studio_holistic_stabilized"


def test_stabilized_live_extractor_rejects_jump_and_holds(monkeypatch):
    # frame 0 + 1 steady, frame 2 teleports the left wrist far away
    scripted = [
        _make_result((0.40, 0.5)),
        _make_result((0.41, 0.5)),
        _make_result((0.95, 0.5)),
    ]
    ext = _build_extractor(monkeypatch, scripted)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    ext.process(frame)
    r1 = ext.process(frame)
    r2 = ext.process(frame)
    # the teleporting frame is held: left-hand global slice stays close to r1
    left_global = slice(6, 39)
    np.testing.assert_allclose(r2.vector[left_global], r1.vector[left_global], atol=0.2)
    # held flag is raised for the left hand on the rejected frame
    assert float(r2.held[0]) == 1.0


def test_stabilized_live_extractor_supports_face_ref_schema(monkeypatch):
    scripted = [_make_result((0.40, 0.5)), _make_result((0.41, 0.5))]
    monkeypatch.setattr(hf.mp_holistic, "Holistic", lambda **_kw: _FakeHolistic(scripted))
    ext = hf.StabilizedHolisticLiveExtractor("smart180_handface220", refine_face_landmarks=False)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    res = ext.process(frame)
    assert res.vector.shape == (220,)
    assert np.isfinite(res.vector).all()
