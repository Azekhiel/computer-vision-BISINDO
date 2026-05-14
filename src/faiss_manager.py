import os
import pandas as pd
import numpy as np
import faiss

DATABASE_DIR = 'dataset_parquets'
MODEL_DIR = 'models'
FAISS_TARGET_FRAMES = 30

def interpolate_sequence(sequence, target_frames):
    """Interpolasi sequence ke jumlah frame yang tetap (default 30)"""
    seq_len = len(sequence)
    if seq_len == target_frames:
        return sequence
    
    x_old = np.linspace(0, 1, seq_len)
    x_new = np.linspace(0, 1, target_frames)
    
    new_seq = np.zeros((target_frames, sequence.shape[1]))
    for i in range(sequence.shape[1]):
        new_seq[:, i] = np.interp(x_new, x_old, sequence[:, i])
        
    return new_seq

def build_faiss_index():
    if not os.path.exists(DATABASE_DIR):
        return False, "Folder database tidak ditemukan."

    parquet_files = [f for f in os.listdir(DATABASE_DIR) if f.endswith('.parquet')]
    if not parquet_files:
        return False, "Database kosong."

    vectors = []
    labels = []
    
    for file in parquet_files:
        vocab = file.replace('.parquet', '')
        df = pd.read_parquet(os.path.join(DATABASE_DIR, file))
        
        # Kita pakai data train saja untuk FAISS
        df_train = df[df['split'] == 'train']
        if df_train.empty:
            continue
            
        grouped = df_train.groupby('video_id')
        for vid, group in grouped:
            group = group.sort_values('frame_num')
            
            # Ekstrak 147-Dimensi dari Parquet
            seq = np.array([list(map(float, f.split(','))) for f in group['features']], dtype=np.float32)
            
            # KUNCI PERBAIKAN: Potong 3 Bendera Oklusi di belakang agar tidak merusak metrik FAISS
            spatial_seq = seq[:, :144]
            
            std_seq = interpolate_sequence(spatial_seq, FAISS_TARGET_FRAMES).astype('float32')
            flat_vec = std_seq.flatten()
            vectors.append(flat_vec)
            labels.append(vocab)

    if not vectors:
        return False, "Tidak ada data training valid."

    X = np.array(vectors)
    faiss.normalize_L2(X)
    
    d = X.shape[1] # Pasti berukuran 30 * 144 = 4320
    index = faiss.IndexFlatIP(d) # Inner Product karena sudah L2 Normalized
    index.add(X)
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    faiss.write_index(index, os.path.join(MODEL_DIR, 'sign_language.index'))
    
    label_map = np.array(labels)
    np.save(os.path.join(MODEL_DIR, 'label_map.npy'), label_map)
    
    return True, f"FAISS Index berhasil dibangun dengan {len(labels)} sampel ({d} dimensi)."