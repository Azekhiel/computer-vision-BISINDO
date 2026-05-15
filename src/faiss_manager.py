import json
import os

import faiss
import numpy as np
import pandas as pd

import database_manager as dbm

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATABASE_DIR = os.path.join(ROOT_DIR, "dataset_parquets")
MODEL_DIR = os.path.join(ROOT_DIR, "models")

FAISS_TARGET_FRAMES = 30
FAISS_FEATURE_DIM = 176
FAISS_DCT_COEFFS = 8
FAISS_VEL_DCT_COEFFS = 6
FAISS_SCORE_THRESHOLD = 0.72
FAISS_DESCRIPTOR_VERSION = "dct_position_velocity_v1"
FAISS_INDEX_FILE = os.path.join(MODEL_DIR, "sign_language.index")
FAISS_LABELS_FILE = os.path.join(MODEL_DIR, "label_map.npy")
FAISS_METADATA_FILE = os.path.join(MODEL_DIR, "faiss_metadata.json")


def parse_features(feature_str: str) -> np.ndarray:
    return np.array(list(map(float, feature_str.split(","))), dtype=np.float32)


def interpolate_sequence(sequence, target_frames: int = FAISS_TARGET_FRAMES) -> np.ndarray:
    """Resample a sequence to a fixed length with linear interpolation."""
    seq = np.asarray(sequence, dtype=np.float32)
    if seq.ndim != 2:
        raise ValueError(f"Expected 2-D sequence, got shape {seq.shape}")

    seq_len = seq.shape[0]
    if seq_len == 0:
        return np.zeros((target_frames, seq.shape[1]), dtype=np.float32)
    if seq_len == target_frames:
        return seq.astype(np.float32, copy=True)
    if seq_len == 1:
        return np.repeat(seq, target_frames, axis=0).astype(np.float32)

    x_old = np.linspace(0.0, 1.0, seq_len, dtype=np.float32)
    x_new = np.linspace(0.0, 1.0, target_frames, dtype=np.float32)
    new_seq = np.empty((target_frames, seq.shape[1]), dtype=np.float32)
    for i in range(seq.shape[1]):
        new_seq[:, i] = np.interp(x_new, x_old, seq[:, i])
    return new_seq


def _dct_basis(length: int, coeffs: int) -> np.ndarray:
    """Orthonormal DCT-II basis, returned as (coeffs, length)."""
    coeffs = int(min(max(coeffs, 1), length))
    n = np.arange(length, dtype=np.float32)
    k = np.arange(coeffs, dtype=np.float32)[:, None]
    basis = np.cos(np.pi / float(length) * (n + 0.5) * k).astype(np.float32)
    basis[0, :] *= np.sqrt(1.0 / length)
    if coeffs > 1:
        basis[1:, :] *= np.sqrt(2.0 / length)
    return basis


def make_temporal_descriptor(
    sequence,
    target_frames: int = FAISS_TARGET_FRAMES,
    feature_dim: int = FAISS_FEATURE_DIM,
    dct_coeffs: int = FAISS_DCT_COEFFS,
    velocity_dct_coeffs: int = FAISS_VEL_DCT_COEFFS,
) -> np.ndarray:
    """
    Compact FAISS descriptor for edge devices.

    The old descriptor flattened 30x176 into 5280 dimensions. This descriptor
    keeps low-frequency temporal shape with DCT position coefficients and DCT
    velocity coefficients. Default dimension: (8 + 6) * 176 = 2464.
    """
    seq = np.asarray(sequence, dtype=np.float32)
    if seq.ndim != 2:
        raise ValueError(f"Expected 2-D sequence, got shape {seq.shape}")
    if seq.shape[1] < feature_dim:
        raise ValueError(f"Expected at least {feature_dim} features, got {seq.shape[1]}")

    std_seq = interpolate_sequence(seq[:, :feature_dim], target_frames)
    std_seq = np.nan_to_num(std_seq, nan=0.0, posinf=0.0, neginf=0.0)

    pos_basis = _dct_basis(target_frames, dct_coeffs)
    pos_coeffs = pos_basis @ std_seq

    velocity = np.diff(std_seq, axis=0)
    if velocity.shape[0] == 0:
        vel_coeffs = np.zeros((velocity_dct_coeffs, feature_dim), dtype=np.float32)
    else:
        vel_basis = _dct_basis(velocity.shape[0], velocity_dct_coeffs)
        vel_coeffs = vel_basis @ velocity

    descriptor = np.concatenate([pos_coeffs.reshape(-1), vel_coeffs.reshape(-1)]).astype(np.float32)
    return normalize_vector(descriptor)


def normalize_vector(vector: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    vec = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vec))
    if norm <= eps:
        return vec
    return (vec / norm).astype(np.float32)


def load_faiss_metadata() -> dict:
    if not os.path.exists(FAISS_METADATA_FILE):
        return {
            "descriptor_version": "legacy_flattened",
            "target_frames": FAISS_TARGET_FRAMES,
            "feature_dim": FAISS_FEATURE_DIM,
            "score_threshold": FAISS_SCORE_THRESHOLD,
        }
    with open(FAISS_METADATA_FILE, "r") as f:
        return json.load(f)


def search_sequence(index, labels, sequence, k: int = 1, threshold: float | None = None):
    threshold = FAISS_SCORE_THRESHOLD if threshold is None else float(threshold)
    descriptor = make_temporal_descriptor(sequence).reshape(1, -1).astype("float32")
    if hasattr(index, "d") and int(index.d) != descriptor.shape[1]:
        raise ValueError(
            f"FAISS index dimension {index.d} does not match descriptor dimension {descriptor.shape[1]}. "
            "Rebuild FAISS after the preprocessing rewrite."
        )
    scores, indices = index.search(descriptor, k=k)
    best_score = float(scores[0][0])
    best_idx = int(indices[0][0])
    if best_idx < 0 or best_score < threshold:
        return "unknown", best_score, scores, indices
    return str(labels[best_idx]), best_score, scores, indices


def build_faiss_index():
    if not os.path.exists(DATABASE_DIR):
        return False, "Folder database tidak ditemukan."

    parquet_files = [f for f in os.listdir(DATABASE_DIR) if f.endswith(".parquet")]
    if not parquet_files:
        return False, "Database kosong."

    vectors = []
    labels = []

    for file in parquet_files:
        vocab = file.replace(".parquet", "")
        if vocab == "idle":
            continue

        df = pd.read_parquet(os.path.join(DATABASE_DIR, file))
        df_train = df[df["split"] == "train"]
        if df_train.empty:
            continue

        for _, group in df_train.groupby("video_id"):
            group = group.sort_values("frame_num")
            seq = np.array([parse_features(f) for f in group["features"]], dtype=np.float32)
            vectors.append(make_temporal_descriptor(seq[:, :FAISS_FEATURE_DIM]))
            labels.append(vocab)

    if not vectors:
        return False, "Tidak ada data training valid."

    X = np.vstack(vectors).astype("float32")
    faiss.normalize_L2(X)

    index = faiss.IndexFlatIP(X.shape[1])
    index.add(X)

    os.makedirs(MODEL_DIR, exist_ok=True)
    faiss.write_index(index, FAISS_INDEX_FILE)
    np.save(FAISS_LABELS_FILE, np.array(labels))

    metadata = {
        "descriptor_version": FAISS_DESCRIPTOR_VERSION,
        "target_frames": FAISS_TARGET_FRAMES,
        "feature_dim": FAISS_FEATURE_DIM,
        "dct_coeffs": FAISS_DCT_COEFFS,
        "velocity_dct_coeffs": FAISS_VEL_DCT_COEFFS,
        "descriptor_dim": int(X.shape[1]),
        "metric": "inner_product_on_l2_normalized_descriptors",
        "score_threshold": FAISS_SCORE_THRESHOLD,
    }
    with open(FAISS_METADATA_FILE, "w") as f:
        json.dump(metadata, f, indent=4)

    dbm.update_metadata("faiss")
    return True, f"FAISS Index berhasil dibangun: {len(labels)} sampel, {X.shape[1]} dimensi, threshold IP {FAISS_SCORE_THRESHOLD:.2f}."
