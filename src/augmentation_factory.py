import pandas as pd
import numpy as np
import random
import uuid
import os
from tqdm import tqdm

# Import database manager untuk memberitahu sistem kalau ada data baru
import database_manager as dbm

DATABASE_FILE = 'dataset_dynamic.csv'
TARGET_SAMPLES = 200

def parse_features(feature_str):
    """Mengubah string koma-koma di CSV menjadi numpy array."""
    return np.array(list(map(float, feature_str.split(','))))

def format_features(feature_array):
    """Mengubah kembali numpy array menjadi string untuk disimpan di CSV."""
    return ','.join(map(str, feature_array))

# ==========================================
# 1. SPATIAL AUGMENTATIONS
# ==========================================
def add_gaussian_noise(sequence, noise_level=0.005):
    """Mensimulasikan tremor tangan / noise dari sensor kamera."""
    noise = np.random.normal(0, noise_level, sequence.shape)
    return sequence + noise

def scale_sequence(sequence, scale_range=(0.85, 1.15)):
    """Mensimulasikan orang dengan proporsi lengan/tubuh yang berbeda."""
    scale_factor = np.random.uniform(*scale_range)
    return sequence * scale_factor

# ==========================================
# 2. TEMPORAL AUGMENTATIONS (VARIABLE-LENGTH)
# ==========================================
def time_warp(sequence):
    """
    Mempercepat atau memperlambat video secara natural.
    Mengubah durasi video antara 80% (lebih cepat) hingga 120% (lebih lambat).
    """
    length = len(sequence)
    if length < 5:
        return sequence # Kalau terlalu pendek, abaikan
    
    # Pilih target panjang frame baru secara acak
    new_length = int(length * np.random.uniform(0.8, 1.2))
    
    old_indices = np.arange(length)
    new_indices = np.linspace(0, length - 1, new_length)
    
    # Interpolasi linear untuk semua 144 fitur secara bersamaan
    warped_seq = np.zeros((new_length, sequence.shape[1]))
    for i in range(sequence.shape[1]):
        warped_seq[:, i] = np.interp(new_indices, old_indices, sequence[:, i])
        
    return warped_seq

def frame_drop_duplicate(sequence, p_drop=0.05, p_dup=0.05):
    """Mensimulasikan lag kamera atau frame-drop yang umum terjadi."""
    if len(sequence) < 5:
        return sequence
        
    new_seq = []
    for frame in sequence:
        rand_val = random.random()
        if rand_val < p_drop:
            continue # Buang frame ini (Frame Drop)
        elif rand_val < p_drop + p_dup:
            new_seq.append(frame)
            new_seq.append(frame) # Duplikasi frame (Lag Freeze)
        else:
            new_seq.append(frame)
            
    # Fallback kalau apes membuang terlalu banyak frame
    if len(new_seq) < 3: 
        return sequence
        
    return np.array(new_seq)

# ==========================================
# MAIN FACTORY LOGIC
# ==========================================
def apply_random_augmentation(sequence):
    """Meracik kombinasi augmentasi secara acak untuk 1 sampel."""
    aug_seq = sequence.copy()
    
    # 70% peluang terkena efek spasial
    if random.random() < 0.7:
        aug_seq = add_gaussian_noise(aug_seq)
    if random.random() < 0.7:
        aug_seq = scale_sequence(aug_seq)
        
    # 50% peluang terkena efek temporal (pilih salah satu)
    if random.random() < 0.5:
        aug_seq = time_warp(aug_seq)
    elif random.random() < 0.5:
        aug_seq = frame_drop_duplicate(aug_seq)
        
    return aug_seq

def generate_dataset(target_samples=TARGET_SAMPLES):
    """
    Membaca database, HANYA mengambil data dengan split=='train',
    dan menggandakannya hingga mencapai target_samples per vocab.
    """
    if not os.path.exists(DATABASE_FILE):
        return False, "Database belum ada. Silakan rekam manual atau import folder terlebih dahulu."
        
    df = pd.read_csv(DATABASE_FILE)
    if df.empty:
        return False, "Database kosong."

    # FILTER KRUSIAL: Hanya ambil data TRAINING untuk digandakan
    train_df = df[df['split'] == 'train']
    
    if train_df.empty:
        return False, "Tidak ada data dengan split 'train'. Tambahkan data training terlebih dahulu."

    # Kelompokkan data training per vocab dan per ID Video
    grouped = train_df.groupby(['label', 'video_id'])
    sequences = {} 
    
    # Rekonstruksi baris-baris CSV menjadi bentuk Matriks Time-Series
    for (label, video_id), group in grouped:
        # Urutkan berdasarkan frame_num agar urutan waktu tidak acak-acakan
        group = group.sort_values('frame_num')
        seq = np.array([parse_features(f) for f in group['features']])
        
        if label not in sequences:
            sequences[label] = []
        sequences[label].append(seq)
        
    new_rows = []
    total_generated = 0
    
    print("\n--- Memulai Pabrik Augmentasi (Hanya Data TRAIN) ---")
    
    for label, seqs in sequences.items():
        current_count = len(seqs)
        
        if current_count >= target_samples:
            print(f"[{label.upper()}] sudah memiliki {current_count} sampel training. (Skip)")
            continue
            
        needed = target_samples - current_count
        print(f"[{label.upper()}] Butuh {needed} sampel sintetis lagi...")
        
        for _ in tqdm(range(needed), desc=f"Augmenting {label}"):
            # 1. Pilih sampel training asli secara acak sebagai basis
            base_seq = random.choice(seqs)
            
            # 2. Terapkan variasi augmentasi spasial-temporal
            aug_seq = apply_random_augmentation(base_seq)
            
            # 3. Beri ID video baru agar tidak tumpang tindih
            new_vid = f"{label}_train_aug_{uuid.uuid4().hex[:6]}"
            
            # 4. Pecah kembali matriksnya menjadi baris-baris untuk dimasukkan ke CSV
            for frame_num, features in enumerate(aug_seq):
                new_rows.append({
                    'video_id': new_vid,
                    'label': label,
                    'frame_num': frame_num,
                    'split': 'train', # Pastikan data generatif selalu dilabeli train
                    'features': format_features(features)
                })
            total_generated += 1
            
    # Jika ada data baru yang berhasil dibuat, append ke CSV
    if new_rows:
        df_new = pd.DataFrame(new_rows)
        df_new.to_csv(DATABASE_FILE, mode='a', header=False, index=False)
        
        # Beritahu sistem bahwa database baru saja berubah agar status model menjadi Outdated
        dbm.update_metadata("db_update")
        
        return True, f"SELESAI! Berhasil men-generate {total_generated} data training baru."
    else:
        return True, "Semua vocab sudah mencapai target. Tidak ada data baru yang dibuat."

if __name__ == "__main__":
    # Test jalankan file ini secara langsung
    status, msg = generate_dataset(TARGET_SAMPLES)
    print(msg)