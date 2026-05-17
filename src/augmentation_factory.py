import pandas as pd
import numpy as np
import random
import uuid
import os
from tqdm import tqdm

import database_manager as dbm
import feature_engine as fe

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATABASE_DIR = os.path.join(ROOT_DIR, 'dataset_parquets')
ENABLE_V4_AUGMENTATION = os.environ.get("BISINDO_ENABLE_V4_AUGMENTATION", "0") == "1"

def parse_features(feature_str):
    return np.array(list(map(float, feature_str.split(','))))

def format_features(feature_array):
    return ','.join(map(str, feature_array))


def _augmentation_tracking_metadata(features: np.ndarray) -> dict:
    arr = np.asarray(features, dtype=np.float32)
    flags = arr[176:179] if arr.shape[0] >= 179 else np.ones(3, dtype=np.float32)
    left = arr[18:81].reshape(21, 3)
    right = arr[81:144].reshape(21, 3)
    left_rendered = bool(np.max(np.linalg.norm(left[:, :2], axis=1)) > 1e-6)
    right_rendered = bool(np.max(np.linalg.norm(right[:, :2], axis=1)) > 1e-6)
    metadata = {
        "pose_detected": bool(flags[fe.IDX_POSE] >= 0.5),
        "left": {
            "source": "augmentation" if left_rendered else "missing",
            "confidence": 0.75 if left_rendered else 0.0,
            "gap_age": 0 if flags[fe.IDX_LH] >= 0.5 else 999,
            "quality_reason": "synthetic_augmentation",
            "original_detected": bool(flags[fe.IDX_LH] >= 0.5),
            "rendered": left_rendered,
            "roi": None,
        },
        "right": {
            "source": "augmentation" if right_rendered else "missing",
            "confidence": 0.75 if right_rendered else 0.0,
            "gap_age": 0 if flags[fe.IDX_RH] >= 0.5 else 999,
            "quality_reason": "synthetic_augmentation",
            "original_detected": bool(flags[fe.IDX_RH] >= 0.5),
            "rendered": right_rendered,
            "roi": None,
        },
    }
    return fe.flatten_tracking_metadata(metadata)

# --- AUGMENTATION LOGICS ---
def add_gaussian_noise(sequence, noise_level=0.005):
    noise = np.random.normal(0, noise_level, sequence.shape)
    return sequence + noise

def _smooth_offsets(length: int, noise_level: float) -> np.ndarray:
    offsets = np.random.normal(0.0, noise_level, size=(length, 1, 3)).astype(np.float32)
    offsets[:, :, 2] *= 0.35
    if length < 3:
        return offsets

    kernel = np.array([1.0, 2.0, 3.0, 2.0, 1.0], dtype=np.float32)
    kernel /= kernel.sum()
    half = len(kernel) // 2
    smoothed = offsets.copy()
    for t in range(length):
        lo = max(0, t - half)
        hi = min(length, t + half + 1)
        k_lo = half - (t - lo)
        k_hi = k_lo + (hi - lo)
        weights = kernel[k_lo:k_hi].reshape(-1, 1, 1)
        smoothed[t] = (offsets[lo:hi] * weights).sum(axis=0) / weights.sum()
    return smoothed.astype(np.float32)

def _add_offset_to_nonzero_block(block: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    out = block.copy()
    nonzero = np.max(np.linalg.norm(block[:, :, :2], axis=2), axis=1) > 1e-6
    out[nonzero] = out[nonzero] + offsets[nonzero]
    return out

def translate_spatial_sequence(sequence, noise_level=0.012):
    out = sequence.copy()
    offsets = _smooth_offsets(len(out), noise_level)
    for sl, points in [(slice(0, 18), 6), (slice(18, 81), 21), (slice(81, 144), 21)]:
        block = out[:, sl].reshape(len(out), points, 3)
        block = _add_offset_to_nonzero_block(block, offsets)
        out[:, sl] = block.reshape(len(out), points * 3)
    return out

def jitter_hand_anchors(sequence, noise_level=0.006):
    out = sequence.copy()
    # Kept for compatibility, but V4 no longer calls this by default. Independent
    # hand-anchor jitter can create impossible hand/body phase shifts.
    for sl in (slice(18, 81), slice(81, 144)):
        block = out[:, sl].reshape(len(out), 21, 3)
        offsets = _smooth_offsets(len(out), noise_level)
        block = _add_offset_to_nonzero_block(block, offsets)
        out[:, sl] = block.reshape(len(out), 63)
    return out

def time_warp(sequence):
    length = len(sequence)
    if length < 5: return sequence
    new_length = int(length * np.random.uniform(0.8, 1.2))
    new_length = max(3, new_length)
    old_indices = np.arange(length)
    new_indices = np.linspace(0, length - 1, new_length)

    warped_seq = np.zeros((new_length, sequence.shape[1]), dtype=np.float32)
    numeric_dim = min(sequence.shape[1], 176)
    for i in range(numeric_dim):
        warped_seq[:, i] = np.interp(new_indices, old_indices, sequence[:, i])

    if sequence.shape[1] > 176:
        nearest = np.rint(new_indices).astype(int)
        nearest = np.clip(nearest, 0, length - 1)
        warped_seq[:, 176:] = sequence[nearest, 176:]

    return warped_seq

def frame_drop_duplicate(sequence, p_drop=0.05, p_dup=0.05):
    if len(sequence) < 5: return sequence
    new_seq = []
    for frame in sequence:
        rand_val = random.random()
        if rand_val < p_drop: continue 
        elif rand_val < p_drop + p_dup:
            new_seq.append(frame)
            new_seq.append(frame) 
        else:
            new_seq.append(frame)
    if len(new_seq) < 3: return sequence
    return np.array(new_seq)

def apply_random_augmentation(sequence):
    aug_seq = fe.sanitize_sequence(sequence.astype(np.float32, copy=True), zero_missing_hands=True)

    if random.random() < 0.55:
        aug_seq = translate_spatial_sequence(aug_seq, noise_level=0.006)

    if random.random() < 0.5:
        aug_seq[:, 144:176] = add_gaussian_noise(aug_seq[:, 144:176], noise_level=0.03)

    if random.random() < 0.35:
        aug_seq = time_warp(aug_seq)
    elif random.random() < 0.25:
        aug_seq = frame_drop_duplicate(aug_seq)
        
    return fe.sanitize_sequence(aug_seq, zero_missing_hands=True)
    
def generate_dataset(target_samples=200, splits_to_augment=['train']):
    """
    splits_to_augment: list string e.g., ['train', 'val'] (Menerima input dari UI)
    """
    if not ENABLE_V4_AUGMENTATION:
        return (
            False,
            "Augmentasi V4 dimatikan default karena augmentasi lama membuat lompatan temporal. "
            "Set BISINDO_ENABLE_V4_AUGMENTATION=1 setelah data V4 bersih tervalidasi."
        )

    if not os.path.exists(DATABASE_DIR):
        return False, "Folder database belum ada."

    parquet_files = [f for f in os.listdir(DATABASE_DIR) if f.endswith('.parquet')]
    if not parquet_files: return False, "Database kosong."

    total_generated = 0
    print(f"\n--- Memulai Pabrik Augmentasi untuk: {splits_to_augment} ---")
    
    for file in parquet_files:
        filepath = os.path.join(DATABASE_DIR, file)
        label = file.replace('.parquet', '')
        
        df = pd.read_parquet(filepath)
        if df.empty: continue
        if 'feature_version' not in df.columns:
            print(f"[{label.upper()}] skip: data belum {fe.FEATURE_SCHEMA}.")
            continue
        df = df[df['feature_version'] == fe.FEATURE_SCHEMA]
        if df.empty:
            print(f"[{label.upper()}] skip: tidak ada data {fe.FEATURE_SCHEMA}.")
            continue

        new_rows = []
        
        # Lakukan iterasi per split yang dipilih user (Bisa Train, Val, atau Test)
        for target_split in splits_to_augment:
            split_df = df[df['split'] == target_split]
            if split_df.empty: continue

            grouped = split_df.groupby('video_id')
            seqs = []
            base_vids_map = [] # Untuk tracking nama file asal
            
            for video_id, group in grouped:
                group = group.sort_values('frame_num')
                seq = np.array([parse_features(f) for f in group['features']])
                seqs.append(seq)
                base_vids_map.append(video_id)
                
            current_count = len(seqs)
            if current_count >= target_samples:
                print(f"[{label.upper()} - {target_split.upper()}] sudah {current_count} sampel. (Skip)")
                continue
                
            needed = target_samples - current_count
            print(f"[{label.upper()} - {target_split.upper()}] Butuh {needed} sintetis lagi...")
            
            for _ in tqdm(range(needed), desc=f"Augmenting {label} ({target_split})"):
                rand_idx = random.randint(0, len(seqs)-1)
                base_seq = seqs[rand_idx]
                base_vid = base_vids_map[rand_idx]
                
                aug_seq = apply_random_augmentation(base_seq)
                
                # LOGIKA PENAMAAN BARU UNTUK AUGMENTASI
                if '_manual_' in base_vid:
                    new_vid = base_vid.replace('_manual_', '_generate_')
                    # Tambahkan UUID agar unique jika base yang sama di-augment berkali-kali
                    new_vid = new_vid + f"_aug_{uuid.uuid4().hex[:4]}"
                else:
                    new_vid = f"{target_split}_generate_aug_{uuid.uuid4().hex[:8]}.avi"
                
                for frame_num, features in enumerate(aug_seq):
                    row = {
                        'video_id': new_vid,
                        'label': label,
                        'frame_num': frame_num,
                        'split': target_split, 
                        'feature_version': fe.FEATURE_SCHEMA,
                        'source_frame_num': frame_num,
                        'features': format_features(features)
                    }
                    row.update(_augmentation_tracking_metadata(features))
                    new_rows.append(row)
                total_generated += 1
            
        if new_rows:
            df_new = pd.DataFrame(new_rows)
            df_gabung = pd.concat([df, df_new], ignore_index=True)
            df_gabung.to_parquet(filepath, index=False)
            
    if total_generated > 0:
        dbm.update_metadata("db_update")
        return True, f"SELESAI! Berhasil men-generate {total_generated} data baru."
    else:
        return True, "Semua vocab target sudah cukup."

if __name__ == "__main__":
    pass
