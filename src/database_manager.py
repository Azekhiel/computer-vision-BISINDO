import pandas as pd
import os
import json
from datetime import datetime
import glob

import feature_engine as fe

# ==========================================
# KONFIGURASI PATH (Tahan Banting)
# ==========================================
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATABASE_DIR = os.path.join(ROOT_DIR, 'dataset_parquets')
MODELS_DIR = os.path.join(ROOT_DIR, 'models')
METADATA_FILE = os.path.join(MODELS_DIR, 'model_status.json')

def init_database():
    """
    Inisialisasi direktori partisi parquet dan metadata JSON jika belum ada.
    """
    os.makedirs(DATABASE_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
        
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
            status[model] = "Belum Ada (Butuh Build)"
        else:
            mod_time = datetime.strptime(model_time_str, "%Y-%m-%d %H:%M:%S")
            # Jika database lebih baru dari model, berarti model sudah usang
            if mod_time < db_time:
                status[model] = "Outdated (Data Baru Tersedia)"
            else:
                status[model] = "Up-to-Date"
                
    return status

def get_database_stats():
    """
    Mengambil statistik dengan cara membaca setiap file parquet di dalam folder.
    Sangat ringan karena yang dibaca hanya per-file kecil.
    """
    stats = {}
    if not os.path.exists(DATABASE_DIR): 
        return stats
    
    # Looping setiap file parquet di folder
    for file in os.listdir(DATABASE_DIR):
        if file.endswith('.parquet'):
            vocab_name = file.replace('.parquet', '')
            filepath = os.path.join(DATABASE_DIR, file)
            
            try:
                df = pd.read_parquet(filepath)
                if df.empty: continue
                df = fe.filter_current_feature_rows(df)
                if df.empty: continue
                
                train_df = df[df['split'] == 'train']
                val_df = df[df['split'] == 'val']
                test_df = df[df['split'] == 'test']
                
                train_count = train_df['video_id'].nunique()
                val_count = val_df['video_id'].nunique()
                test_count = test_df['video_id'].nunique()
                
                # Cek jumlah augmentasi
                train_asli = train_df[~train_df['video_id'].astype(str).str.contains('_aug_')]['video_id'].nunique()
                train_gen = train_count - train_asli
                
                stats[vocab_name] = {
                    "Total Train": train_count,
                    "Train (Asli)": train_asli,
                    "Train (Generate)": train_gen,
                    "Total Val": val_count,
                    "Total Test": test_count
                }
            except Exception:
                pass # Jika file corrupt, lewati saja
                
    return stats

def get_vocab_list():
    """Mengembalikan list nama vocab hanya dengan membaca nama file di folder (Sangat Cepat)."""
    if not os.path.exists(DATABASE_DIR): 
        return []
    
    vocabs = []
    for file in os.listdir(DATABASE_DIR):
        if file.endswith('.parquet'):
            vocabs.append(file.replace('.parquet', ''))
            
    vocabs.sort()
    return vocabs

def get_samples_by_vocab(vocab_name):
    """Mengambil daftar video_id yang dimiliki oleh suatu vocab."""
    filepath = os.path.join(DATABASE_DIR, f"{vocab_name}.parquet")
    if not os.path.exists(filepath): return []
    
    df = pd.read_parquet(filepath)
    if df.empty: return []
    df = fe.filter_current_feature_rows(df)
    if df.empty: return []
    
    return df['video_id'].unique().tolist()

# ==========================================
# CRUD OPERATIONS (Create, Read, Update, Delete)
# ==========================================

def delete_vocab(vocab_name):
    """Menghapus sebuah vocab cukup dengan menghapus file parquet-nya."""
    filepath = os.path.join(DATABASE_DIR, f"{vocab_name}.parquet")
    
    if not os.path.exists(filepath):
        return False, f"Vocab '{vocab_name}' tidak ditemukan."
        
    try:
        os.remove(filepath)
        update_metadata("db_update")
        return True, f"Seluruh data untuk '{vocab_name}' berhasil dihapus."
    except Exception as e:
        return False, f"Gagal menghapus file: {e}"

def delete_sample(video_id):
    """Mencari video_id di seluruh partisi dan menghapusnya dari DataFrame bersangkutan."""
    if not os.path.exists(DATABASE_DIR): return False, "Database tidak ditemukan."
    
    for file in os.listdir(DATABASE_DIR):
        if file.endswith('.parquet'):
            filepath = os.path.join(DATABASE_DIR, file)
            try:
                df = pd.read_parquet(filepath)
                if video_id in df['video_id'].values:
                    initial_rows = len(df)
                    df = df[df['video_id'] != video_id]
                    
                    if len(df) < initial_rows:
                        # Timpa kembali file parquet-nya
                        df.to_parquet(filepath, index=False)
                        update_metadata("db_update")
                        return True, f"Sampel '{video_id}' berhasil dihapus."
            except Exception:
                continue
                
    return False, f"Sampel '{video_id}' tidak ditemukan."

def rename_vocab(old_name, new_name):
    """Mengubah nama label dan nama file parquet-nya."""
    new_name = new_name.strip().replace(" ", "_").lower()
    
    old_filepath = os.path.join(DATABASE_DIR, f"{old_name}.parquet")
    new_filepath = os.path.join(DATABASE_DIR, f"{new_name}.parquet")
    
    if not os.path.exists(old_filepath): 
        return False, f"Vocab '{old_name}' tidak ditemukan."
        
    if os.path.exists(new_filepath):
        return False, f"Vocab '{new_name}' sudah ada. Gunakan nama lain."
        
    try:
        # Buka data lama
        df = pd.read_parquet(old_filepath)
        if df.empty:
            return False, "Database kosong."
            
        # Ubah label
        df['label'] = new_name
        
        # Update prefix video_id
        def update_video_id(vid_id):
            if str(vid_id).startswith(f"{old_name}_"):
                return str(vid_id).replace(f"{old_name}_", f"{new_name}_", 1)
            return vid_id
            
        df['video_id'] = df['video_id'].apply(update_video_id)
        
        # Simpan sebagai file baru
        df.to_parquet(new_filepath, index=False)
        
        # Hapus file lama
        os.remove(old_filepath)
        
        update_metadata("db_update")
        return True, f"Vocab '{old_name}' berhasil diubah menjadi '{new_name}'."
        
    except Exception as e:
        return False, f"Terjadi kesalahan saat mengubah nama: {e}"

if __name__ == "__main__":
    init_database()
    print("Database terinisialisasi. Status model saat ini:")
    print(json.dumps(check_model_status(), indent=4))
