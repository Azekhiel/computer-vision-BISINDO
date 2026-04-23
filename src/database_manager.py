import pandas as pd
import os
import json
from datetime import datetime

DATABASE_FILE = 'dataset_dynamic.csv'
METADATA_FILE = 'models/model_status.json'

def init_database():
    """
    Inisialisasi database CSV dan metadata JSON jika belum ada.
    """
    if not os.path.exists(DATABASE_FILE):
        # Struktur kolom baru yang mendukung durasi dinamis dan pemisahan train/val/test
        df = pd.DataFrame(columns=['video_id', 'label', 'frame_num', 'split', 'features'])
        df.to_csv(DATABASE_FILE, index=False)
        
    if not os.path.exists('models'):
        os.makedirs('models')
        
    if not os.path.exists(METADATA_FILE):
        update_metadata("init")

def update_metadata(action="db_update"):
    """
    Mencatat waktu setiap kali ada perubahan pada database atau model di-build.
    action bisa berupa: 'db_update', 'faiss', 'lstm', 'transformer', atau 'init'.
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, 'r') as f:
            data = json.load(f)
    else:
        # Template dasar jika file JSON belum ada
        data = {
            "last_database_update": "None", 
            "faiss_built_at": "None", 
            "lstm_trained_at": "None", 
            "transformer_trained_at": "None"
        }
        
    if action == "db_update" or action == "init": 
        data["last_database_update"] = now
    elif action == "faiss": 
        data["faiss_built_at"] = now
    elif action == "lstm": 
        data["lstm_trained_at"] = now
    elif action == "transformer": 
        data["transformer_trained_at"] = now
        
    with open(METADATA_FILE, 'w') as f:
        json.dump(data, f, indent=4)

def check_model_status():
    """
    Mengecek apakah model yang ada sudah menggunakan data paling update.
    Sangat berguna untuk memberi notifikasi ke user di UI.
    """
    if not os.path.exists(METADATA_FILE): 
        return {"faiss": "Unknown", "lstm": "Unknown", "transformer": "Unknown"}
    
    with open(METADATA_FILE, 'r') as f:
        data = json.load(f)
        
    if data["last_database_update"] == "None":
        return {"faiss": "Belum Ada", "lstm": "Belum Ada", "transformer": "Belum Ada"}
        
    db_time = datetime.strptime(data["last_database_update"], "%Y-%m-%d %H:%M:%S")
    
    status = {}
    for model in ["faiss", "lstm", "transformer"]:
        model_time_str = data[f"{model}_built_at" if model == "faiss" else f"{model}_trained_at"]
        
        if model_time_str == "None":
            status[model] = "Belum Ada (Butuh Build) ❌"
        else:
            mod_time = datetime.strptime(model_time_str, "%Y-%m-%d %H:%M:%S")
            # Jika database lebih baru dari model, berarti model sudah usang (outdated)
            if mod_time < db_time:
                status[model] = "Outdated (Data Baru Tersedia) ⚠️"
            else:
                status[model] = "Up-to-Date ✅"
                
    return status

def get_database_stats():
    """
    Mengambil statistik komprehensif dari database untuk ditampilkan di Dashboard UI.
    Memisahkan perhitungan antara train (asli vs generate), val, dan test.
    """
    if not os.path.exists(DATABASE_FILE): 
        return {}
    
    df = pd.read_csv(DATABASE_FILE)
    if df.empty: 
        return {}
    
    stats = {}
    # Kita groupby label untuk menghitung per vocab
    grouped = df.groupby('label')
    
    for label, group in grouped:
        # Hitung jumlah sampel (berdasarkan video_id unik, bukan baris frame)
        train_df = group[group['split'] == 'train']
        val_df = group[group['split'] == 'val']
        test_df = group[group['split'] == 'test']
        
        train_count = train_df['video_id'].nunique()
        val_count = val_df['video_id'].nunique()
        test_count = test_df['video_id'].nunique()
        
        # Mengecek berapa banyak data training yang asli vs hasil augmentasi (berakhiran '_aug_')
        train_asli = train_df[~train_df['video_id'].str.contains('_aug_')]['video_id'].nunique()
        train_gen = train_count - train_asli
        
        stats[label] = {
            "Total Train": train_count,
            "Train (Asli)": train_asli,
            "Train (Generate)": train_gen,
            "Total Val": val_count,
            "Total Test": test_count
        }
        
    return stats

def get_vocab_list():
    """Mengembalikan list nama vocab yang tersedia di database dan mengurutkannya."""
    if not os.path.exists(DATABASE_FILE): 
        return []
    
    df = pd.read_csv(DATABASE_FILE)
    if df.empty: 
        return []
        
    vocabs = df['label'].unique().tolist()
    vocabs.sort()
    return vocabs

def get_samples_by_vocab(vocab_name):
    """Mengambil daftar video_id yang dimiliki oleh suatu vocab (berguna untuk UI hapus sampel spesifik)."""
    if not os.path.exists(DATABASE_FILE): return []
    
    df = pd.read_csv(DATABASE_FILE)
    if df.empty: return []
    
    samples = df[df['label'] == vocab_name]['video_id'].unique().tolist()
    return samples

# ==========================================
# CRUD OPERATIONS (Create, Read, Update, Delete)
# ==========================================

def delete_vocab(vocab_name):
    """Menghapus semua sampel yang terkait dengan sebuah vocab."""
    if not os.path.exists(DATABASE_FILE): return False, "Database tidak ditemukan."
    
    df = pd.read_csv(DATABASE_FILE)
    if df.empty: return False, "Database kosong."
    
    initial_rows = len(df)
    # Filter dataset, buang yang labelnya sama dengan vocab_name
    df = df[df['label'] != vocab_name]
    
    if len(df) == initial_rows:
        return False, f"Vocab '{vocab_name}' tidak ditemukan."
        
    df.to_csv(DATABASE_FILE, index=False)
    update_metadata("db_update") # Beri tahu sistem bahwa data berubah
    return True, f"Seluruh data untuk '{vocab_name}' berhasil dihapus."

def delete_sample(video_id):
    """Menghapus spesifik 1 sampel video (berguna jika ada sampel rekaman yang gerakannya jelek)."""
    if not os.path.exists(DATABASE_FILE): return False, "Database tidak ditemukan."
    
    df = pd.read_csv(DATABASE_FILE)
    if df.empty: return False, "Database kosong."
    
    initial_rows = len(df)
    df = df[df['video_id'] != video_id]
    
    if len(df) == initial_rows:
        return False, f"Sampel '{video_id}' tidak ditemukan."
        
    df.to_csv(DATABASE_FILE, index=False)
    update_metadata("db_update")
    return True, f"Sampel '{video_id}' berhasil dihapus."

def rename_vocab(old_name, new_name):
    """Mengubah label suatu vocab menjadi nama baru di seluruh baris database."""
    if not os.path.exists(DATABASE_FILE): return False, "Database tidak ditemukan."
    
    df = pd.read_csv(DATABASE_FILE)
    if df.empty: return False, "Database kosong."
    
    # Standarisasi nama baru
    new_name = new_name.strip().replace(" ", "_").lower()
    
    # Cek apakah nama baru sudah dipakai oleh vocab lain
    if new_name in df['label'].unique():
        return False, f"Vocab dengan nama '{new_name}' sudah ada. Gunakan nama lain."
        
    # Lakukan proses ubah nama
    df.loc[df['label'] == old_name, 'label'] = new_name
    
    # Karena video_id biasanya memuat nama vocab (misal: "halo_1"), kita sekalian ubah prefix video_id-nya
    # supaya tetap rapi dan konsisten
    def update_video_id(vid_id):
        if vid_id.startswith(f"{old_name}_"):
            return vid_id.replace(f"{old_name}_", f"{new_name}_", 1)
        return vid_id
        
    df['video_id'] = df['video_id'].apply(update_video_id)
    
    df.to_csv(DATABASE_FILE, index=False)
    update_metadata("db_update")
    return True, f"Vocab '{old_name}' berhasil diubah menjadi '{new_name}'."

if __name__ == "__main__":
    # Script untuk memastikan file terinisialisasi saat pertama kali dijalankan
    init_database()
    print("Database terinisialisasi. Status model saat ini:")
    print(json.dumps(check_model_status(), indent=4))