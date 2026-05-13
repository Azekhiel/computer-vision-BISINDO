import os
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
import uuid

# Import modul internal
import feature_engine as fe
import database_manager as dbm

mp_holistic = mp.solutions.holistic

# ==========================================
# KONFIGURASI DIREKTORI & PARAMETER
# ==========================================
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATABASE_DIR = os.path.join(ROOT_DIR, 'dataset_parquets')

# Parameter Auto-Trim (Hanya untuk isyarat valid, BUKAN untuk 'idle')
START_THRESH = 0.020
STOP_THRESH = 0.010

def auto_trim_sequence(sequence, scores):
    """
    Memotong frame di awal (sebelum tangan diangkat) 
    dan di akhir (setelah tangan diturunkan) berdasarkan skor pergerakan.
    """
    if not sequence or len(sequence) < 5:
        return sequence

    start_idx = 0
    end_idx = len(sequence) - 1

    # Cari titik mulai (dari depan ke belakang)
    for i, score in enumerate(scores):
        if score > START_THRESH:
            # Ambil sedikit frame ancang-ancang
            start_idx = max(0, i - 2)
            break

    # Cari titik berhenti (dari belakang ke depan)
    for i in range(len(scores) - 1, -1, -1):
        if scores[i] > STOP_THRESH:
            # Ambil sedikit frame sisa gerakan
            end_idx = min(len(sequence) - 1, i + 2)
            break

    if start_idx >= end_idx:
        return sequence # Failsafe kalau gerakannya aneh

    return sequence[start_idx:end_idx+1]

def process_video(video_path, vocab_name, holistic):
    """
    Ekstrak fitur dari 1 video. Punya jalur khusus untuk kelas 'idle'.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    raw_sequence = []
    movement_scores = []
    prev_vector = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Konversi warna untuk MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(frame_rgb)
        
        # Ekstrak 144-D fitur relatif
        keypoints = fe.extract_keypoints_relative(results)
        
        # Hitung skor gerakan untuk keperluan auto-trim nanti
        score = fe.calculate_movement_score(prev_vector, keypoints)
        
        raw_sequence.append(keypoints)
        movement_scores.append(score)
        prev_vector = keypoints

    cap.release()

    if not raw_sequence:
        return None

    # ==========================================
    # LOGIKA BYPASS UNTUK KELAS IDLE
    # ==========================================
    if vocab_name == 'idle':
        # Jangan dipotong! Ambil semua frame apa adanya biar Segmenter bisa belajar gerakan diam
        final_sequence = raw_sequence
    else:
        # Potong awal dan akhir gerakan yang gak penting buat 14 kelas isyarat
        final_sequence = auto_trim_sequence(raw_sequence, movement_scores)
        
    # Filter kalau videonya terlalu pendek setelah dipotong
    if len(final_sequence) < 8:
        return None
        
    return final_sequence

def is_vocab_folder(path):
    """
    Cek apakah folder berisi file video langsung (berarti ini folder vocab).
    """
    files = os.listdir(path)
    video_extensions = ('.mp4', '.avi', '.mov', '.gif')
    return any(f.lower().endswith(video_extensions) for f in files)

def bulk_import(source_paths, split_type="train"):
    """
    Import cerdas: bisa menerima list folder atau satu folder besar.
    source_paths: bisa berupa string path tunggal atau list of strings.
    """
    if isinstance(source_paths, str):
        source_paths = [source_paths]

    folders_to_process = []

    for path in source_paths:
        if not os.path.exists(path): continue
        
        if is_vocab_folder(path):
            # Jika folder langsung berisi video, masukkan ke list proses
            folders_to_process.append(path)
        else:
            # Jika folder berisi sub-folder, masukkan semua sub-foldernya
            sub_folders = [os.path.join(path, f) for f in os.listdir(path) 
                           if os.path.isdir(os.path.join(path, f))]
            folders_to_process.extend(sub_folders)

    if not folders_to_process:
        return False, "Tidak ada folder valid yang ditemukan."

    # Proses ekstraksi fitur (menggunakan logika Holistic yang sudah ada)
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        for vocab_path in folders_to_process:
            vocab = os.path.basename(vocab_path)
            video_files = [f for f in os.listdir(vocab_path) if f.endswith(('.mp4', '.avi', '.mov'))]
            
            if not video_files: continue

            all_vocab_data = []
            for video_file in video_files:
                video_path = os.path.join(vocab_path, video_file)
                video_id = f"{vocab}_{split_type}_manual_{uuid.uuid4().hex[:8]}"
                sequence = process_video(video_path, vocab, holistic)
                
                if sequence is not None:
                    for frame_num, features in enumerate(sequence):
                        all_vocab_data.append({
                            'video_id': video_id, 'label': vocab, 'frame_num': frame_num,
                            'split': split_type, 'features': ','.join(map(str, features))
                        })

            if all_vocab_data:
                df_new = pd.DataFrame(all_vocab_data)
                parquet_path = os.path.join(DATABASE_DIR, f"{vocab}.parquet")
                if os.path.exists(parquet_path):
                    df_combined = pd.concat([pd.read_parquet(parquet_path), df_new], ignore_index=True)
                    df_combined.to_parquet(parquet_path, index=False)
                else:
                    df_new.to_parquet(parquet_path, index=False)
                    
    dbm.update_metadata("db_update")
    return True, f"Berhasil memproses {len(folders_to_process)} folder kosakata."

if __name__ == "__main__":
    # Ganti string di bawah kalau mau nge-test run langsung dari file ini
    source_folder = os.path.join(ROOT_DIR, 'raw_videos') 
    status, msg = bulk_import(source_folder, "train") # Ditambah argumen default biar gak error
    print(msg)