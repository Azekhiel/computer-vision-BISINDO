import pandas as pd
import numpy as np
import faiss
import os

import database_manager as dbm

# ==========================================
# KONFIGURASI PATH (Tahan Banting)
# ==========================================
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATABASE_DIR = os.path.join(ROOT_DIR, 'dataset_parquets')
MODELS_DIR = os.path.join(ROOT_DIR, 'models')

INDEX_FILE = os.path.join(MODELS_DIR, 'sign_language.index')
LABEL_FILE = os.path.join(MODELS_DIR, 'label_map.npy')

# Ukuran standarisasi untuk FAISS. 
# Berapapun durasi asli videonya, akan diinterpolasi menjadi 30 frame.
FAISS_TARGET_FRAMES = 30 
# Total dimensi = 30 frame x 144 fitur spasial = 4320 dimensi
FAISS_DIMENSION = FAISS_TARGET_FRAMES * 144 

def interpolate_sequence(sequence, target_length=FAISS_TARGET_FRAMES):
    """
    Menyamakan durasi frame menjadi target_length menggunakan interpolasi linear.
    Memastikan array yang masuk ke FAISS selalu memiliki dimensi yang persis sama,
    tanpa merusak alur waktu gerakan aslinya.
    """
    length = len(sequence)
    if length == target_length:
        return sequence
        
    old_indices = np.arange(length)
    new_indices = np.linspace(0, length - 1, target_length)
    
    # sequence.shape[1] adalah 144 dimensi spasial dari MediaPipe
    interpolated_seq = np.zeros((target_length, sequence.shape[1]))
    
    for i in range(sequence.shape[1]):
        interpolated_seq[:, i] = np.interp(new_indices, old_indices, sequence[:, i])
        
    return interpolated_seq

def parse_features(feature_str):
    """Mengubah string koma-koma di parquet menjadi numpy array."""
    return np.array(list(map(float, feature_str.split(','))))

def build_faiss_index():
    """
    Membangun ulang indeks pencarian vektor FAISS menggunakan data 'train' terbaru.
    Membaca data dari folder partisi parquet untuk mencegah Out Of Memory.
    """
    if not os.path.exists(DATABASE_DIR):
        return False, "Folder database belum ada. Silakan rekam atau import data terlebih dahulu."
        
    print("\n--- Membaca Database Partisi (Folder) ---")
    
    X_train = []
    y_train = []
    valid_vocabs = set()
    total_rows_loaded = 0
    
    try:
        # Iterasi membaca setiap file parquet di dalam folder
        for file in os.listdir(DATABASE_DIR):
            if file.endswith('.parquet'):
                filepath = os.path.join(DATABASE_DIR, file)
                df = pd.read_parquet(filepath)
                
                if df.empty: continue
                
                # Saring HANYA data train untuk masuk ke memori
                train_df = df[df['split'] == 'train']
                if train_df.empty: continue
                
                total_rows_loaded += len(train_df)
                
                # Pengelompokan berdasarkan video_id untuk merangkai deret waktu
                grouped = train_df.groupby(['label', 'video_id'])
                
                for (label, video_id), group in grouped:
                    # Pastikan frame berurutan
                    group = group.sort_values('frame_num')
                    
                    # Susun matriks sequence [Panjang_Frame_Asli x 144]
                    seq = np.array([parse_features(f) for f in group['features']])
                    
                    # Standarisasi panjang sequence menjadi ukuran tetap (30 frame)
                    std_seq = interpolate_sequence(seq, FAISS_TARGET_FRAMES)
                    
                    # Flatten matriks menjadi vektor 1D [4320 dimensi] untuk FAISS
                    flat_vector = std_seq.flatten()
                    
                    X_train.append(flat_vector)
                    y_train.append(label)
                    valid_vocabs.add(label)

    except Exception as e:
        return False, f"Gagal membaca memori: {e}"

    if not X_train:
        return False, "Tidak ada data dengan split 'train' untuk di-build."

    print(f"Berhasil memuat {total_rows_loaded} baris data latih ke memori.")
    print("\n--- Memulai Build Index FAISS ---")

    # Konversi ke array float32 (wajib untuk operasi matriks FAISS)
    X_matrix = np.array(X_train).astype('float32')
    y_labels = np.array(y_train)
    
    print(f"Dimensi Matriks FAISS Terbentuk: {X_matrix.shape}")

    # Normalisasi L2 pada setiap vektor fitur untuk perhitungan Cosine Similarity.
    # Ini sangat penting agar model lebih fokus pada 'bentuk' pose ketimbang jarak absolut tubuh ke kamera.
    faiss.normalize_L2(X_matrix)
    
    # Inisialisasi indeks pencarian vektor L2 CPU dasar
    cpu_index = faiss.IndexFlatL2(FAISS_DIMENSION)
    
    # ==========================================
    # AUTO-DEVICE DETECTOR (CUDA GPU ALLOCATION)
    # ==========================================
    final_index = None
    try:
        # Mencoba mengakses VRAM GPU
        res = faiss.StandardGpuResources()
        gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
        
        # Masukkan matriks data ke memori GPU
        gpu_index.add(X_matrix)
        
        # Ekstrak kembali ke memori CPU agar bisa di-save ke hard disk dengan aman
        final_index = faiss.index_gpu_to_cpu(gpu_index)
        print("[AKSELERASI] FAISS sukses menggunakan akselerasi CUDA GPU.")
    except (AttributeError, Exception) as e:
        # Fallback jika tidak ada NVIDIA GPU atau faiss-gpu tidak terinstall
        cpu_index.add(X_matrix)
        final_index = cpu_index
        print("[AKSELERASI] GPU tidak terdeteksi/error. FAISS berjalan di CPU biasa.")

    # Pastikan folder models ada
    os.makedirs(MODELS_DIR, exist_ok=True)
        
    # Simpan binary index FAISS dan pemetaan label numpy-nya
    faiss.write_index(final_index, INDEX_FILE)
    np.save(LABEL_FILE, y_labels)
    
    # Beritahu sistem utama bahwa FAISS sudah menggunakan data ter-update
    dbm.update_metadata("faiss")
    
    msg = f"Index FAISS berhasil dibangun!\nTotal Vocab: {len(valid_vocabs)}\nTotal Vektor Tersimpan: {len(y_labels)}"
    return True, msg

def get_faiss_status():
    """Mengembalikan status index FAISS saat ini dari metadata manager."""
    status_dict = dbm.check_model_status()
    return status_dict.get("faiss", "Unknown")

if __name__ == "__main__":
    # Test eksekusi mandiri untuk memastikan struktur folder benar
    status, pesan = build_faiss_index()
    print(pesan)