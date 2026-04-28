import os
import numpy as np
import pandas as pd
import faiss

# Import modul internal
import database_manager as dbm

# ==========================================
# KONFIGURASI DIREKTORI
# ==========================================
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATABASE_DIR = os.path.join(ROOT_DIR, 'dataset_parquets')
MODEL_DIR = os.path.join(ROOT_DIR, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

FAISS_INDEX = os.path.join(MODEL_DIR, 'sign_language.index')
FAISS_LABELS = os.path.join(MODEL_DIR, 'label_map.npy')

# ==========================================
# FUNGSI UTILITAS
# ==========================================
def parse_features(feature_str):
    return np.array(list(map(float, feature_str.split(','))))

def interpolate_sequence(seq, target_len=30):
    """
    Menyamakan durasi frame isyarat menjadi persis 30 frame
    Ini wajib buat FAISS biar dimensi vektornya selalu sama (4320-D)
    """
    seq_len = len(seq)
    if seq_len == target_len:
        return seq
    
    indices = np.linspace(0, seq_len - 1, target_len)
    interpolated_seq = np.zeros((target_len, seq.shape[1]))
    
    for i in range(seq.shape[1]):
        interpolated_seq[:, i] = np.interp(indices, np.arange(seq_len), seq[:, i])
        
    return interpolated_seq

# ==========================================
# FUNGSI BUILD INDEX UTAMA
# ==========================================
def build_faiss_index():
    print("Mulai merakit index FAISS...")
    vocabs = dbm.get_vocab_list()
    
    vectors = []
    labels_list = []
    label_map_array = []
    current_label_id = 0
    
    for vocab in vocabs:
        if vocab == 'idle':
            print("Melewati kelas 'idle'...")
            continue
            
        filepath = os.path.join(DATABASE_DIR, f"{vocab}.parquet")
        if not os.path.exists(filepath):
            continue
            
        print(f"Mengekstrak data {vocab}...")
        label_map_array.append(vocab)
        
        df = pd.read_parquet(filepath)
        
        for vid, group in df.groupby('video_id'):
            group = group.sort_values('frame_num')
            seq = np.array([parse_features(f) for f in group['features']])
            
            std_seq = interpolate_sequence(seq, 30).astype('float32')
            flat_vec = std_seq.flatten()
            
            vectors.append(flat_vec)
            # Simpan ID kelasnya (0, 1, 2..), BUKAN urutan videonya
            labels_list.append(current_label_id)
            
        current_label_id += 1
        
    if not vectors:
        return False, "Data isyarat valid gak ditemuin buat dibikin index."
        
    # 1. Konversi ke Numpy Array
    vectors_np = np.array(vectors, dtype='float32')
    # WAJIB int64 untuk dipakai di FAISS IDMap
    ids_np = np.array(labels_list, dtype=np.int64) 
    
    # 2. Normalisasi L2 biar murni Cosine Similarity
    faiss.normalize_L2(vectors_np)
    
    # 3. KUNCI PERBAIKAN: Gunakan IndexIDMap
    d = vectors_np.shape[1] 
    base_index = faiss.IndexFlatIP(d)
    index = faiss.IndexIDMap(base_index) # Bungkus index dasar agar paham label kelas
    
    # 4. Masukkan vektor BERSAMAAN dengan ID labelnya
    index.add_with_ids(vectors_np, ids_np)
    
    # Simpan file ke harddisk
    faiss.write_index(index, FAISS_INDEX)
    np.save(FAISS_LABELS, np.array(label_map_array))
    
    return True, "Build index FAISS sukses dijalankan"

if __name__ == "__main__":
    status, msg = build_faiss_index()
    print(msg)