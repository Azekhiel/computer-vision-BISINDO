import pandas as pd
import numpy as np
import random
import uuid
import os
from tqdm import tqdm

import database_manager as dbm

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATABASE_DIR = os.path.join(ROOT_DIR, 'dataset_parquets')

def parse_features(feature_str):
    return np.array(list(map(float, feature_str.split(','))))

def format_features(feature_array):
    return ','.join(map(str, feature_array))

# --- AUGMENTATION LOGICS ---
def add_gaussian_noise(sequence, noise_level=0.005):
    noise = np.random.normal(0, noise_level, sequence.shape)
    return sequence + noise

def scale_sequence(sequence, scale_range=(0.85, 1.15)):
    scale_factor = np.random.uniform(*scale_range)
    return sequence * scale_factor

def time_warp(sequence):
    length = len(sequence)
    if length < 5: return sequence
    new_length = int(length * np.random.uniform(0.8, 1.2))
    old_indices = np.arange(length)
    new_indices = np.linspace(0, length - 1, new_length)
    warped_seq = np.zeros((new_length, sequence.shape[1]))
    for i in range(sequence.shape[1]):
        warped_seq[:, i] = np.interp(new_indices, old_indices, sequence[:, i])
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
    aug_seq = sequence.copy()
    
    # KUNCI PERBAIKAN: Pecah 179-D menjadi 3 blok yang berbeda sifatnya
    spatial_features = aug_seq[:, :144]   # Koordinat XYZ (Aman untuk di-Scale & Noise)
    angles = aug_seq[:, 144:176]          # Sudut Sendi 2D (TIDAK BOLEH di-Scale, boleh di-Noise kecil)
    flags = aug_seq[:, 176:]              # Bendera Oklusi 0/1 (TIDAK BOLEH diubah sedikitpun)
    
    if random.random() < 0.7:
        spatial_features = add_gaussian_noise(spatial_features)
    if random.random() < 0.7:
        spatial_features = scale_sequence(spatial_features)
        
    # Opsional: Berikan sedikit noise pada sudut agar model lebih robust (sekitar 0.05 radian / ~2.8 derajat)
    if random.random() < 0.5:
        angles = add_gaussian_noise(angles, noise_level=0.05)
        
    # Gabungkan kembali menjadi array 179-D yang utuh
    aug_seq = np.concatenate([spatial_features, angles, flags], axis=1)
        
    # Efek Temporal (Warp Waktu & Frame Drop) dikenakan ke SELURUH array 179-D sekaligus
    if random.random() < 0.5:
        aug_seq = time_warp(aug_seq)
    elif random.random() < 0.5:
        aug_seq = frame_drop_duplicate(aug_seq)
        
    return aug_seq
    
def generate_dataset(target_samples=200, splits_to_augment=['train']):
    """
    splits_to_augment: list string e.g., ['train', 'val'] (Menerima input dari UI)
    """
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
                    new_rows.append({
                        'video_id': new_vid,
                        'label': label,
                        'frame_num': frame_num,
                        'split': target_split, 
                        'features': format_features(features)
                    })
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