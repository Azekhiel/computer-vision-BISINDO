"""Tests for face-reference feature blocks (src/face_reference.py)."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import face_reference as fr  # noqa: E402


SHOULDERS = np.array([[0.40, 0.50, 0.0, 1.0], [0.60, 0.50, 0.0, 1.0]], dtype=np.float32)
# anchor = (0.5, 0.5, 0.0); scale = ||(0.4,0.5)-(0.6,0.5)|| = 0.2


def _make_face(rng: np.random.Generator) -> np.ndarray:
    face = np.zeros((468, 3), dtype=np.float32)
    for idx in fr.MOUTHDYN_POINTS:
        face[idx] = rng.uniform(0.3, 0.7, size=3).astype(np.float32)
    # give the mouth/eye some plausible separation so derived scalars are non-zero
    face[fr.MOUTH_TOP] = np.array([0.50, 0.44, 0.0], dtype=np.float32)
    face[fr.MOUTH_BOTTOM] = np.array([0.50, 0.48, 0.0], dtype=np.float32)
    face[fr.MOUTH_LEFT] = np.array([0.46, 0.46, 0.0], dtype=np.float32)
    face[fr.MOUTH_RIGHT] = np.array([0.54, 0.46, 0.0], dtype=np.float32)
    face[fr.NOSE_TIP] = np.array([0.50, 0.40, 0.0], dtype=np.float32)
    face[fr.EYE_L_OUTER] = np.array([0.44, 0.36, 0.0], dtype=np.float32)
    face[fr.EYE_L_INNER] = np.array([0.48, 0.36, 0.0], dtype=np.float32)
    face[fr.EYE_R_INNER] = np.array([0.52, 0.36, 0.0], dtype=np.float32)
    face[fr.EYE_R_OUTER] = np.array([0.56, 0.36, 0.0], dtype=np.float32)
    return face


def _make_hand(rng: np.random.Generator, center: float) -> np.ndarray:
    return (rng.uniform(-0.05, 0.05, size=(21, 3)).astype(np.float32) + center)


def _smart_vec(rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(-1.0, 1.0, size=180).astype(np.float32)


@pytest.mark.parametrize(
    "schema,dim",
    [
        ("smart180_mouthdyn214", 214),
        ("smart180_mouthstat206", 206),
        ("smart180_handface220", 220),
        ("smart180_handface_vel286", 286),
    ],
)
def test_output_dims(schema, dim):
    rng = np.random.default_rng(0)
    smart = _smart_vec(rng)
    face = _make_face(rng)
    left = _make_hand(rng, 0.35)
    right = _make_hand(rng, 0.65)
    vec = fr.build_face_schema_vector(schema, smart, face, SHOULDERS, left, right, face_present=True, state={})
    assert vec.shape == (dim,)
    assert np.isfinite(vec).all()
    # smart180 base is preserved verbatim
    np.testing.assert_allclose(vec[:180], smart, rtol=0, atol=0)


@pytest.mark.parametrize("schema", fr.FACE_REF_SCHEMA_NAMES)
def test_column_names_match_block_dim(schema):
    names = fr.face_block_column_names(schema)
    assert len(names) == fr.FACE_REF_BLOCK_DIMS[schema]
    assert len(set(names)) == len(names)  # unique


@pytest.mark.parametrize("schema", fr.FACE_REF_SCHEMA_NAMES)
def test_missing_face_zeros_block_and_flag(schema):
    rng = np.random.default_rng(1)
    smart = _smart_vec(rng)
    left = _make_hand(rng, 0.35)
    right = _make_hand(rng, 0.65)
    vec = fr.build_face_schema_vector(
        schema, smart, None, SHOULDERS, left, right, face_present=False, state={}
    )
    block = vec[180:]
    # face_present flag must be 0 wherever it lives
    names = fr.face_block_column_names(schema)
    fp_idx = names.index("face_present")
    assert block[fp_idx] == 0.0
    # all face-derived entries are zero when the face is absent
    if schema == "smart180_handface220":
        # only hand-presence flags (last two) may be non-zero
        np.testing.assert_allclose(block[:38], 0.0)
    elif schema == "smart180_handface_vel286":
        np.testing.assert_allclose(block[:38], 0.0)  # handface face part
    else:
        # mouth schemas are purely face-derived → fully zero
        np.testing.assert_allclose(block, 0.0)


def test_missing_hand_zeros_hand_vectors():
    rng = np.random.default_rng(2)
    smart = _smart_vec(rng)
    face = _make_face(rng)
    right = _make_hand(rng, 0.65)
    vec = fr.build_face_schema_vector(
        "smart180_handface220", smart, face, SHOULDERS, None, right, face_present=True, state={}
    )
    block = vec[180:]
    names = fr.face_block_column_names("smart180_handface220")
    # left-hand vectors (wrist_to_* and palm_to_nose) must be zero
    for i, nm in enumerate(names):
        if nm.startswith("left_") or nm in ("dist_left_wrist_nose", "dist_left_palm_mouth"):
            assert block[i] == 0.0, nm
    assert block[names.index("left_present")] == 0.0
    assert block[names.index("right_present")] == 1.0
    assert block[names.index("face_present")] == 1.0


@pytest.mark.parametrize("schema", fr.FACE_REF_SCHEMA_NAMES)
def test_translation_invariance(schema):
    rng = np.random.default_rng(3)
    smart = _smart_vec(rng)
    face = _make_face(rng)
    left = _make_hand(rng, 0.35)
    right = _make_hand(rng, 0.65)
    base = fr.build_face_schema_vector(schema, smart, face, SHOULDERS, left, right, state={})

    delta = np.array([0.13, -0.07, 0.05], dtype=np.float32)
    face_t = face + delta
    left_t = left + delta
    right_t = right + delta
    shoulders_t = SHOULDERS.copy()
    shoulders_t[:, :3] += delta
    moved = fr.build_face_schema_vector(schema, smart, face_t, shoulders_t, left_t, right_t, state={})
    np.testing.assert_allclose(moved[180:], base[180:], rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("schema", fr.FACE_REF_SCHEMA_NAMES)
def test_scale_invariance(schema):
    rng = np.random.default_rng(4)
    smart = _smart_vec(rng)
    face = _make_face(rng)
    left = _make_hand(rng, 0.35)
    right = _make_hand(rng, 0.65)
    base = fr.build_face_schema_vector(schema, smart, face, SHOULDERS, left, right, state={})

    k = 2.5
    moved = fr.build_face_schema_vector(
        schema, smart, face * k, SHOULDERS * k, left * k, right * k, state={}
    )
    # velocity slices depend on smart (unscaled) so compare the face/handface part only
    block_dim = fr.FACE_REF_BLOCK_DIMS[schema]
    cmp = block_dim if schema != "smart180_handface_vel286" else 40
    np.testing.assert_allclose(moved[180:180 + cmp], base[180:180 + cmp], rtol=1e-3, atol=1e-4)


def test_mouthdyn_velocity_first_frame_zero_then_nonzero():
    rng = np.random.default_rng(5)
    smart = _smart_vec(rng)
    face0 = _make_face(rng)
    state: dict = {}
    names = fr.face_block_column_names("smart180_mouthdyn214")
    do_idx = names.index("d_mouth_openness")
    dy_idx = names.index("d_mouth_y")

    v0 = fr.build_face_schema_vector("smart180_mouthdyn214", smart, face0, SHOULDERS, state=state, dt=0.1)
    assert v0[180 + do_idx] == 0.0
    assert v0[180 + dy_idx] == 0.0

    face1 = face0.copy()
    face1[fr.MOUTH_TOP] = face1[fr.MOUTH_TOP] + np.array([0.0, -0.02, 0.0], dtype=np.float32)
    v1 = fr.build_face_schema_vector("smart180_mouthdyn214", smart, face1, SHOULDERS, state=state, dt=0.1)
    assert abs(v1[180 + do_idx]) > 1e-4  # openness changed


def test_handface_vel_velocity_uses_dt():
    rng = np.random.default_rng(6)
    face = _make_face(rng)
    left = _make_hand(rng, 0.35)
    right = _make_hand(rng, 0.65)
    smart0 = _smart_vec(rng)
    smart1 = smart0.copy()
    smart1[6:39] = smart1[6:39] + 0.5  # move left global slice

    state: dict = {}
    v0 = fr.build_face_schema_vector("smart180_handface_vel286", smart0, face, SHOULDERS, left, right, state=state, dt=0.1)
    np.testing.assert_allclose(v0[180 + 40:180 + 106], 0.0)  # first frame: no velocity

    v1 = fr.build_face_schema_vector("smart180_handface_vel286", smart1, face, SHOULDERS, left, right, state=state, dt=0.1)
    vel_left = v1[180 + 40:180 + 73]
    np.testing.assert_allclose(vel_left, 0.5 / 0.1, rtol=1e-4)  # (0.5)/dt


def test_missing_face_resets_mouthdyn_velocity_baseline():
    rng = np.random.default_rng(7)
    smart = _smart_vec(rng)
    face = _make_face(rng)
    state: dict = {}
    names = fr.face_block_column_names("smart180_mouthdyn214")
    do_idx = names.index("d_mouth_openness")

    fr.build_face_schema_vector("smart180_mouthdyn214", smart, face, SHOULDERS, state=state, dt=0.1)
    # face disappears -> baseline cleared
    fr.build_face_schema_vector("smart180_mouthdyn214", smart, None, SHOULDERS, face_present=False, state=state, dt=0.1)
    assert "mouthdyn_prev_openness" not in state
    # face returns -> velocity must be 0 again (no stale prev)
    v = fr.build_face_schema_vector("smart180_mouthdyn214", smart, face, SHOULDERS, state=state, dt=0.1)
    assert v[180 + do_idx] == 0.0


@pytest.mark.parametrize("schema", fr.FACE_REF_SCHEMA_NAMES)
def test_offline_live_parity(schema):
    """The live extractor (build_paper_feature) and the offline formula must agree."""
    import types

    import holistic_features as hf
    from smart_extract.live_bisindo_mp_real_shoulder_v6 import build_feature

    rng = np.random.default_rng(11)
    face = np.zeros((478, 3), dtype=np.float32)
    for idx in fr.MOUTHDYN_POINTS:
        face[idx] = rng.uniform(0.3, 0.7, size=3).astype(np.float32)
    pose = rng.uniform(0.3, 0.7, size=(33, 3)).astype(np.float32)
    pose[11] = np.array([0.40, 0.50, 0.0], dtype=np.float32)
    pose[12] = np.array([0.60, 0.50, 0.0], dtype=np.float32)
    left = rng.uniform(0.30, 0.45, size=(21, 3)).astype(np.float32)
    right = rng.uniform(0.55, 0.70, size=(21, 3)).astype(np.float32)

    def _lm(arr, vis=None):
        pts = []
        for i, (x, y, z) in enumerate(arr):
            o = types.SimpleNamespace(x=float(x), y=float(y), z=float(z))
            if vis is not None:
                o.visibility = float(vis[i])
            pts.append(o)
        return types.SimpleNamespace(landmark=pts)

    results = types.SimpleNamespace(
        left_hand_landmarks=_lm(left),
        right_hand_landmarks=_lm(right),
        pose_landmarks=_lm(pose, vis=np.ones(33)),
        face_landmarks=_lm(face),
    )

    live_vec, _ = hf.build_paper_feature(schema, results, state={}, dt=0.1)

    # Offline formula path on the same arrays.
    face468 = face[:468]
    shoulders = pose[[11, 12], :3]
    shoulders = np.concatenate([shoulders, np.ones((2, 1), dtype=np.float32)], axis=1)
    present = np.array([1.0, 1.0], dtype=np.float32)
    smart = build_feature("btj_global_local", left, right, shoulders, present, present.copy(), np.zeros(2, np.float32), present.copy())
    offline_vec = fr.build_face_schema_vector(schema, smart, face468, shoulders, left, right, face_present=True, state={}, dt=0.1)

    np.testing.assert_allclose(live_vec, offline_vec, rtol=1e-5, atol=1e-6)


def test_rejects_wrong_smart_dim_and_non_face_schema():
    rng = np.random.default_rng(8)
    face = _make_face(rng)
    with pytest.raises(ValueError):
        fr.build_face_schema_vector("smart180_mouthdyn214", np.zeros(179, np.float32), face, SHOULDERS)
    with pytest.raises(ValueError):
        fr.build_face_schema_vector("smart180", _smart_vec(rng), face, SHOULDERS)
