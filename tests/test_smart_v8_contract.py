import os
import sys

import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from smart_extract import contract as sc
import feature_schemas as fs
import holistic_features
from smart_extract.extract_video_smart_v8 import make_best_extract_args
from smart_extract.live_bisindo_mp_real_shoulder_v6 import build_feature


def _hand(anchor=(0.4, 0.5), scale=0.035):
    pts = np.zeros((21, 3), dtype=np.float32)
    pts[0, :2] = anchor
    for idx in range(1, 21):
        pts[idx, :2] = pts[0, :2] + np.array([(idx % 5) * scale, -(idx // 5) * scale], dtype=np.float32)
    return pts


def test_best_extract_args_match_run_best_contract():
    args = make_best_extract_args()

    assert args.feature_mode == "btj_global_local"
    assert args.target_fps == 10.0
    assert args.width == 640
    assert args.height == 480
    assert args.center_crop == 1.0
    assert args.proc_width == 384
    assert args.shoulder_backend == "mp-pose"
    assert args.pose_every == 3
    assert args.pose_proc_width == 256
    assert args.det_conf == 0.40
    assert args.track_conf == 0.45
    assert args.smooth_alpha == 0.78
    assert args.hold_frames == 5
    assert args.smart_mode == "best"
    assert args.search_radius == 2
    assert args.enhance == "auto"
    assert args.fallback_variants == sc.BEST_FALLBACK_VARIANTS
    assert args.gif_width == 420


def test_btj_global_local_builds_180d_smart_schema():
    shoulders = np.array([[0.36, 0.43, 0.0, 1.0], [0.64, 0.43, 0.0, 1.0]], dtype=np.float32)
    vec = build_feature(
        sc.FEATURE_MODE,
        _hand((0.38, 0.55)),
        _hand((0.62, 0.55)),
        shoulders,
        np.array([1.0, 1.0], dtype=np.float32),
        np.array([1.0, 1.0], dtype=np.float32),
        np.array([0.0, 0.0], dtype=np.float32),
        np.array([0.9, 0.8], dtype=np.float32),
    )

    assert sc.FEATURE_SCHEMA == "bisindo_smart_v8_btj_global_local_180_best"
    assert vec.shape == (sc.FEATURE_DIM,)
    assert vec.shape[0] == 180
    np.testing.assert_allclose(vec[sc.SLICE_META][0:2], [1.0, 1.0])


def test_filter_current_feature_rows_requires_smart_schema_and_180d():
    import pandas as pd

    df = pd.DataFrame(
        [
            {"feature_version": sc.FEATURE_SCHEMA, "feature_dim": 180, "features": sc.format_feature_value(np.zeros(180))},
            {"feature_version": "bisindo_v6_jetson_safe_handtrack", "feature_dim": 179, "features": "0"},
        ]
    )

    out = sc.filter_current_feature_rows(df)

    assert len(out) == 1
    assert out.iloc[0]["feature_version"] == sc.FEATURE_SCHEMA


class _Landmark:
    def __init__(self, value: float) -> None:
        self.x = value
        self.y = value + 0.1
        self.z = value + 0.2
        self.visibility = value + 0.3


class _Landmarks:
    def __init__(self, count: int, start: float = 0.01) -> None:
        self.landmark = [_Landmark(start + idx / 1000.0) for idx in range(count)]


class _Results:
    right_hand_landmarks = _Landmarks(21, 0.10)
    left_hand_landmarks = _Landmarks(21, 0.20)
    pose_landmarks = _Landmarks(33, 0.30)
    face_landmarks = _Landmarks(468, 0.40)


def test_paper_holistic_feature_shapes_and_zero_padding():
    smart, smart_meta = holistic_features.build_paper_feature("smart180", _Results())
    khukuh, khukuh_meta = holistic_features.build_paper_feature("khukuh1629", _Results())
    adi, adi_meta = holistic_features.build_paper_feature("adi1662", _Results())
    smart_face, smart_face_meta = holistic_features.build_paper_feature("smart180_face1584", _Results())

    assert smart.shape == (fs.get_schema("smart180").feature_dim,)
    assert khukuh.shape == (fs.get_schema("khukuh1629").feature_dim,)
    assert adi.shape == (fs.get_schema("adi1662").feature_dim,)
    assert smart_face.shape == (fs.get_schema("smart180_face1584").feature_dim,)
    assert smart_meta["left_present"] == 1.0
    assert khukuh_meta["left_present"] == 1.0
    assert adi_meta["right_present"] == 1.0
    assert smart_face_meta["right_present"] == 1.0

    class MissingResults:
        right_hand_landmarks = None
        left_hand_landmarks = None
        pose_landmarks = None
        face_landmarks = None

    missing, meta = holistic_features.build_paper_feature("khukuh1629", MissingResults())
    assert missing.shape == (1629,)
    assert np.count_nonzero(missing) == 0
    assert meta["left_present"] == 0.0
