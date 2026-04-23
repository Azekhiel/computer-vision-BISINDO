import os
import cv2
import pandas as pd
import numpy as np
import mediapipe as mp
import uuid
from tqdm import tqdm

import feature_engine as fe
import database_manager as dbm

mp_holistic = mp.solutions.holistic
DATABASE_FILE = 'dataset_dynamic.csv'

def auto_trim_sequence(sequence_data, threshold=0.015):
    """
    Mendeteksi kapan gerakan dimulai dan berakhir menggunakan kalkulasi
    jarak vektor spasial. Membuang frame diam di awal dan akhir video.
    """
    if len(sequence_data) < 5:
        return sequence_data # Terlalu pendek untuk di-trim

    movement_scores = []
    for i in range(1, len(sequence_data)):
        score = fe.calculate_movement_score(sequence_data[i-1], sequence_data[i])
        movement_scores.append(score)
    
    # Cari di mana gerakan pertama kali melebihi threshold
    start_idx = 0
    for i, score in enumerate(movement_scores):
        if score > threshold:
            # Ambil sedikit buffer (1 frame sebelum gerakan dimulai)
            start_idx = max(0, i - 1)
            break
            
    # Cari di mana gerakan berakhir (berhenti) dari belakang
    end_idx = len(sequence_data) - 1
    for i in range(len(movement_scores)-1, -1, -1):
        if movement_scores[i] > threshold:
            # Beri buffer 1 frame setelah gerakan selesai
            end_idx = min(len(sequence_data) - 1, i + 2)
            break
            
    # Jika tidak ada gerakan yang melebihi threshold (mungkin gerakannya super halus), 
    # kembalikan seluruh sequence
    if start_idx >= end_idx:
        return sequence_data
        
    return sequence_data[start_idx:end_idx+1]

def process_media_file(filepath, vocab_label, sample_id, split_type, holistic_model):
    """
    Membaca 1 file video/GIF, mengekstrak fitur 144-dimensi, memotong frame diam,
    dan mengembalikan DataFrame siap simpan beserta label split-nya.
    """
    cap = cv2.VideoCapture(filepath)
    raw_sequence = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic_model.process(frame_rgb)
        
        # Ekstrak 144-D fitur relatif
        keypoints = fe.extract_keypoints_relative(results)
        raw_sequence.append(keypoints)
        
    cap.release()
    
    if len(raw_sequence) == 0:
        return None, "File kosong atau format tidak terbaca oleh OpenCV"
        
    # Terapkan Auto-Trimmer untuk mendapatkan sequence inti yang padat
    trimmed_sequence = auto_trim_sequence(raw_sequence)
    
    # Validasi panjang frame
    if len(trimmed_sequence) < 5:
        return None, "Gerakan terlalu pendek / tidak terdeteksi pose manusia"
        
    # Bentuk format row untuk database
    df_rows = []
    for frame_num, features in enumerate(trimmed_sequence):
        df_rows.append({
            'video_id': sample_id,
            'label': vocab_label,
            'frame_num': frame_num,
            'split': split_type,           # <- Fitur baru: menandai Train/Val/Test
            'features': ','.join(map(str, features))
        })
        
    return pd.DataFrame(df_rows), "Sukses"

def bulk_import(parent_folder, split_type='train'):
    """
    Membaca folder utama yang berisi sub-folder vocab.
    Semua video/GIF di dalam folder tersebut akan dilabeli sesuai nama sub-folder,
    dan dimasukkan ke database dengan kategori split_type ('train', 'val', 'test').
    """
    # Validasi input split
    if split_type not in ['train', 'val', 'test']:
        return False, "Parameter split_type harus 'train', 'val', atau 'test'."

    if not os.path.exists(parent_folder):
        return False, f"Folder '{parent_folder}' tidak ditemukan."
        
    # Pastikan database sudah terinisialisasi
    dbm.init_database()
    
    all_dataframes = []
    total_files_processed = 0
    total_success = 0
    
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        vocab_folders = [f for f in os.listdir(parent_folder) if os.path.isdir(os.path.join(parent_folder, f))]
        
        if not vocab_folders:
            return False, f"Tidak ada folder vocab di dalam '{parent_folder}'."

        for folder_name in vocab_folders:
            # Standarisasi nama vocab
            vocab_label = folder_name.strip().replace(" ", "_").lower()
            vocab_path = os.path.join(parent_folder, folder_name)
            
            files = [f for f in os.listdir(vocab_path) if os.path.isfile(os.path.join(vocab_path, f))]
            if not files:
                continue
                
            print(f"\nMemproses Vocab: [{vocab_label.upper()}] untuk split: [{split_type.upper()}]")
            
            for filename in tqdm(files, desc=f"Importing {vocab_label}"):
                total_files_processed += 1
                filepath = os.path.join(vocab_path, filename)
                
                # Buat video_id yang unik tapi informatif
                raw_name = os.path.splitext(filename)[0].strip().replace(" ", "_")
                unique_hex = uuid.uuid4().hex[:4]
                # Format: nama_vocab_split_namasampel_hex (misal: halo_val_sampel1_a1b2)
                sample_id = f"{vocab_label}_{split_type}_{raw_name}_{unique_hex}"
                
                df_sample, msg = process_media_file(filepath, vocab_label, sample_id, split_type, holistic)
                
                if df_sample is not None:
                    all_dataframes.append(df_sample)
                    total_success += 1
                else:
                    print(f"  -> [SKIP] {filename}: {msg}")

    if all_dataframes:
        final_df = pd.concat(all_dataframes, ignore_index=True)
        # Append ke CSV tanpa menimpa data yang sudah ada
        final_df.to_csv(DATABASE_FILE, mode='a', header=False, index=False)
        
        # PENTING: Lapor ke metadata bahwa ada data baru masuk
        dbm.update_metadata("db_update")
        
        msg = f"Berhasil mengimpor {total_success} dari {total_files_processed} file ke database sebagai '{split_type}'."
        print(f"\n[SELESAI] {msg}")
        return True, msg
    else:
        return False, "Tidak ada data valid yang bisa diekstrak dari folder tersebut."

if __name__ == "__main__":
    # Script untuk mengetes proses import mandiri tanpa UI
    # Misal kita punya 3 folder: "data_raw/train", "data_raw/val", "data_raw/test"
    
    print("Contoh Penggunaan Script Data Ingestion:")
    folder_train = "data_raw/train_vocab"
    
    if os.path.exists(folder_train):
        print(f"Mencoba import folder {folder_train} sebagai data TRAIN...")
        bulk_import(folder_train, split_type='train')
    else:
        print(f"Bikin folder '{folder_train}' dulu, lalu isi dengan folder-folder vocab, lalu jalankan lagi script ini.")