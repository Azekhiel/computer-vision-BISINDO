import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import feature_schemas as fs
import full_mediapipe_converter as fmc
from smart_extract import contract as sc


def _write_synthetic_full_npz(path: Path, *, missing_left_face_frame: bool = True) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = 3
    pose = np.zeros((frames, 33, 5), dtype=np.float32)
    pose[:, :, 0] = np.linspace(0.2, 0.8, 33, dtype=np.float32)
    pose[:, :, 1] = np.linspace(0.1, 0.9, 33, dtype=np.float32)
    pose[:, 11, :4] = np.array([0.35, 0.42, 0.0, 0.9], dtype=np.float32)
    pose[:, 12, :4] = np.array([0.65, 0.42, 0.0, 0.9], dtype=np.float32)

    left = np.ones((frames, 21, 5), dtype=np.float32) * 0.25
    right = np.ones((frames, 21, 5), dtype=np.float32) * 0.65
    face = np.ones((frames, 478, 5), dtype=np.float32) * 0.5
    masks = np.ones((frames, 9), dtype=bool)
    masks[:, 5:] = False
    if missing_left_face_frame:
        masks[1, 2] = False
        masks[1, 4] = False
    np.savez_compressed(
        path,
        frame_index=np.arange(frames, dtype=np.int32),
        timestamp_ms=np.arange(frames, dtype=np.float32) * 100.0,
        image_size=np.tile(np.array([[1920, 1080]], dtype=np.int32), (frames, 1)),
        source_duration_sec=np.asarray([0.3], dtype=np.float32),
        source_fps=np.asarray([10.0], dtype=np.float32),
        pose_landmarks=pose,
        pose_world_landmarks=np.zeros((frames, 33, 4), dtype=np.float32),
        left_hand_landmarks=left,
        right_hand_landmarks=right,
        face_landmarks=face,
        face_blendshapes=np.zeros((frames, 0), dtype=np.float32),
        facial_transformation_matrixes=np.zeros((frames, 0, 4, 4), dtype=np.float32),
        left_handedness=np.zeros((frames, 2), dtype=np.float32),
        right_handedness=np.zeros((frames, 2), dtype=np.float32),
        availability_masks=masks,
        source_video_path=np.asarray(["source.mkv"]),
    )
    return path


def test_full_schema_names_has_eight_entries():
    assert len(fs.FULL_SCHEMA_NAMES) == 8
    for name in fs.FACE_REF_SCHEMA_NAMES:
        assert name in fs.FULL_SCHEMA_NAMES


def test_convert_synthetic_npz_to_all_schema_dims_and_missing_masks(tmp_path):
    npz = _write_synthetic_full_npz(tmp_path / "train" / "aku" / "train_aku_0001.landmarks.npz")
    seqs, metas = fmc.convert_npz_to_schema_sequences(npz, fs.FULL_SCHEMA_NAMES)

    assert seqs["smart180"].shape == (3, 180)
    assert seqs["khukuh1629"].shape == (3, 1629)
    assert seqs["adi1662"].shape == (3, 1662)
    assert seqs["smart180_face1584"].shape == (3, 1584)
    for name in fs.FACE_REF_SCHEMA_NAMES:
        assert seqs[name].shape == (3, fs.get_schema(name).feature_dim)
        assert np.isfinite(seqs[name]).all()
    assert metas[0]["source_width"] == 1920
    assert metas[0]["source_height"] == 1080
    assert metas[1]["left_present"] == 0.0
    assert metas[1]["face_present"] == 0.0
    assert np.allclose(seqs["smart180_face1584"][1, 180:], 0.0)


def test_convert_face_ref_zeros_face_block_when_face_missing(tmp_path):
    import face_reference as fr

    npz = _write_synthetic_full_npz(tmp_path / "train" / "aku" / "train_aku_0001.landmarks.npz")
    seqs, _ = fmc.convert_npz_to_schema_sequences(npz, fs.FACE_REF_SCHEMA_NAMES)
    # frame 1 has face mask off -> face_present flag drops to 0 in every face-ref schema
    for name in fs.FACE_REF_SCHEMA_NAMES:
        block = seqs[name][1, 180:]
        names = fr.face_block_column_names(name)
        assert block[names.index("face_present")] == 0.0
        if name in ("smart180_mouthdyn214", "smart180_mouthstat206"):
            assert np.allclose(block, 0.0)  # mouth schemas are purely face-derived


def test_extract_full_dataset_writes_schema_parquets_and_manifest(tmp_path):
    source = tmp_path / "dataset_full_mediapipe" / "features_holistic" / "fps_10"
    _write_synthetic_full_npz(source / "train" / "aku" / "train_aku_0001.landmarks.npz")
    (source / "train" / "aku" / "train_aku_0001.report.json").write_text('{"width":1920,"height":1080,"fps":10.0}', encoding="utf-8")
    result = fmc.extract_full_mediapipe_dataset(
        source=tmp_path / "dataset_full_mediapipe",
        dataset_dir=tmp_path / "dataset_parquets",
        model_dir=tmp_path / "models",
        backup_root=tmp_path / "backups",
        schema="face",
    )

    assert result.converted == 1
    parquet = tmp_path / "dataset_parquets" / "smart180_face1584" / "aku.parquet"
    assert parquet.exists()
    df = pd.read_parquet(parquet)
    assert len(df) == 3
    assert df["feature_dim"].astype(int).eq(1584).all()
    assert sc.parse_feature_value(df["features"].iloc[0]).shape[0] == 1584
    assert result.manifest_path and result.manifest_path.exists()
    # `face` group now also emits the 4 face-reference parquets.
    hf_parquet = tmp_path / "dataset_parquets" / "smart180_handface220" / "aku.parquet"
    assert hf_parquet.exists()
    df_hf = pd.read_parquet(hf_parquet)
    assert df_hf["feature_dim"].astype(int).eq(220).all()
    assert sc.parse_feature_value(df_hf["features"].iloc[0]).shape[0] == 220
