import os
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
import uuid
import concurrent.futures
from collections import defaultdict
from tqdm import tqdm

import feature_engine as fe
import database_manager as dbm

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATABASE_DIR = os.path.join(ROOT_DIR, 'dataset_parquets')

# ==========================================
# KONFIGURASI MULTIPROCESSING & TRIMMER
# ==========================================
# N-WORKERS: Jumlah video yang diekstrak secara bersamaan.
# Jetson Orin Nano punya 6 Core, set di angka 4 atau 5 agar UI tetap responsif.
MAX_WORKERS = 5  

START_THRESH = 0.020
STOP_THRESH = 0.010

# --- Worker Initialization (Kunci Kecepatan Multiprocessing) ---
worker_holistic = None

def init_worker():
    """
    Fungsi ini dipanggil HANYA SEKALI saat Core CPU (Worker) baru dihidupkan.
    MediaPipe di-load ke RAM per-Core, bukan per-video.
    """
    global worker_holistic
    # Kita matikan log error C++ di tiap worker agar terminal tetap rapi
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
    os.environ['GLOG_minloglevel'] = '2'
    
    worker_holistic = mp.solutions.holistic.Holistic(
        min_detection_confidence=0.5, 
        min_tracking_confidence=0.5
    )

def auto_trim_sequence(sequence, scores):
    if not sequence or len(sequence) < 5: return sequence
    start_idx = 0
    end_idx = len(sequence) - 1
    for i, score in enumerate(scores):
        if score > START_THRESH:
            start_idx = max(0, i - 2)
            break
    for i in range(len(scores) - 1, -1, -1):
        if scores[i] > STOP_THRESH:
            end_idx = min(len(sequence) - 1, i + 2)
            break
    if start_idx >= end_idx: return sequence 
    return sequence[start_idx:end_idx+1]

def process_media_task(args):
    """Fungsi mandiri yang dieksekusi oleh masing-masing Core CPU."""
    file_path, vocab_name, split_type, video_id = args
    global worker_holistic
    
    ext = os.path.splitext(file_path)[1].lower()
    img_exts = ['.jpg', '.jpeg', '.png', '.gif']
    
    # --- LOGIKA GAMBAR STATIS ---
    if ext in img_exts:
        frame = cv2.imread(file_path)
        if frame is None: return video_id, vocab_name, split_type, None, "Gagal baca gambar"
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = worker_holistic.process(frame_rgb)
        keypoints = fe.extract_keypoints_relative(results)
        
        final_sequence = [keypoints for _ in range(30)]
        return video_id, vocab_name, split_type, final_sequence, "OK"

    # --- LOGIKA VIDEO NORMAL ---
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened(): return video_id, vocab_name, split_type, None, "Gagal buka video"

    raw_sequence = []
    movement_scores = []
    prev_vector = None
    
    while True:
        ret, frame = cap.read()
        if not ret: break
            
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = worker_holistic.process(frame_rgb)
        
        keypoints = fe.extract_keypoints_relative(results)
        score = fe.calculate_movement_score(prev_vector, keypoints)
        
        raw_sequence.append(keypoints)
        movement_scores.append(score)
        prev_vector = keypoints

    cap.release()

    if not raw_sequence: return video_id, vocab_name, split_type, None, "0 frame terbaca"

    if vocab_name == 'idle':
        final_sequence = raw_sequence
    else:
        final_sequence = auto_trim_sequence(raw_sequence, movement_scores)
        
    if len(final_sequence) < 8:
        return video_id, vocab_name, split_type, None, f"Sisa {len(final_sequence)} frame"
        
    return video_id, vocab_name, split_type, final_sequence, "OK"

def is_vocab_folder(path):
    files = os.listdir(path)
    media_extensions = ('.mkv', '.mp4', '.avi', '.mov', '.webm', '.jpg', '.jpeg', '.png', '.gif')
    return any(f.lower().endswith(media_extensions) for f in files)

def bulk_import(source_paths, default_split="train", mp_device="CPU"):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
    os.environ['GLOG_minloglevel'] = '2'
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1' if mp_device == "CPU" else '0'

    if isinstance(source_paths, str): source_paths = [source_paths]

    grouped_folders = defaultdict(list)

    # --- 1. Rounting & Mapping Folder ---
    for path in source_paths:
        if not os.path.exists(path): continue
        
        basename = os.path.basename(os.path.normpath(path)).lower()

        if is_vocab_folder(path):
            parent_name = os.path.basename(os.path.dirname(path)).lower()
            split = parent_name if parent_name in ['train', 'val', 'test'] else default_split
            grouped_folders[split].append(path)
            
        elif basename in ['train', 'val', 'test']:
            vocab_dirs = [os.path.join(path, v) for v in os.listdir(path) if os.path.isdir(os.path.join(path, v))]
            for vocab_path in vocab_dirs:
                if is_vocab_folder(vocab_path):
                    grouped_folders[basename].append(vocab_path)
                    
        else:
            sub_dirs = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
            split_subdirs = [d for d in sub_dirs if d.lower() in ['train', 'val', 'test']]
            
            if len(split_subdirs) > 0:
                for split_name in split_subdirs:
                    split_path = os.path.join(path, split_name)
                    vocab_dirs = [os.path.join(split_path, v) for v in os.listdir(split_path) if os.path.isdir(os.path.join(split_path, v))]
                    for vocab_path in vocab_dirs:
                        if is_vocab_folder(vocab_path):
                            grouped_folders[split_name.lower()].append(vocab_path)
            else:
                for sub_dir in sub_dirs:
                    vocab_path = os.path.join(path, sub_dir)
                    if is_vocab_folder(vocab_path):
                        grouped_folders[default_split].append(vocab_path)

    # --- 2. Mengumpulkan Semua Pekerjaan (Tasks) ---
    tasks = []
    vocab_existing_ids = {} # Cache untuk menyimpan ID video yang sudah ada di database
    
    for split_type in ['train', 'val', 'test', 'lainnya']:
        if split_type not in grouped_folders and split_type != 'lainnya': continue
        
        for vocab_path in grouped_folders.get(split_type, []):
            vocab = os.path.basename(vocab_path)
            
            # Cek Duplikasi (Hanya Load 1x per Vocab)
            if vocab not in vocab_existing_ids:
                existing_ids = set()
                parquet_path = os.path.join(DATABASE_DIR, f"{vocab}.parquet")
                if os.path.exists(parquet_path):
                    try:
                        df_existing = pd.read_parquet(parquet_path, columns=['video_id'])
                        existing_ids = set(df_existing['video_id'].unique())
                    except Exception:
                        pass
                vocab_existing_ids[vocab] = existing_ids
                
            media_files = [f for f in os.listdir(vocab_path) if f.lower().endswith(('.mkv','.mp4', '.avi', '.mov', '.webm', '.jpg', '.jpeg', '.png', '.gif'))]
            
            for media_file in media_files:
                file_path = os.path.join(vocab_path, media_file)
                video_id = f"{split_type}_manual_{media_file}"
                
                if video_id in vocab_existing_ids[vocab]:
                    continue # Lewati jika sudah pernah diekstrak
                    
                tasks.append((file_path, vocab, split_type, video_id))
                
    if not tasks:
        return True, "Semua file sudah ada di database (Tidak ada file baru)."

    # --- 3. Eksekusi Paralel (Multiprocessing) ---
    print("\n" + "="*60)
    print(f"🚀 MEMULAI EKSTRAKSI {len(tasks)} FILE ({MAX_WORKERS} PEKERJA PARALEL)")
    print("="*60)

    # Tempat menampung hasil sementara sebelum ditulis ke Parquet
    results_by_vocab = defaultdict(list)
    failed_count = 0
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS, initializer=init_worker) as executor:
        # Menjalankan fungsi task pada semua data dengan progress bar TQDM
        for result in tqdm(executor.map(process_media_task, tasks), total=len(tasks), desc="Progress Keseluruhan"):
            video_id, vocab, split_type, sequence, msg = result
            
            if sequence is not None:
                for frame_num, features in enumerate(sequence):
                    results_by_vocab[vocab].append({
                        'video_id': video_id, 'label': vocab, 'frame_num': frame_num,
                        'split': split_type, 'features': ','.join(map(str, features))
                    })
            else:
                failed_count += 1

    # --- 4. Menyimpan Hasil Ekstraksi Secara Aman ---
    print("\n💾 Menyimpan hasil ekstraksi ke Parquet...")
    for vocab, rows in results_by_vocab.items():
        df_new = pd.DataFrame(rows)
        parquet_path = os.path.join(DATABASE_DIR, f"{vocab}.parquet")
        if os.path.exists(parquet_path):
            df_combined = pd.concat([pd.read_parquet(parquet_path), df_new], ignore_index=True)
            df_combined.to_parquet(parquet_path, index=False)
        else:
            df_new.to_parquet(parquet_path, index=False)
            
    dbm.update_metadata("db_update")
    
    pesan_akhir = f"Selesai! {len(tasks) - failed_count} file berhasil diekstrak."
    if failed_count > 0: pesan_akhir += f" ({failed_count} gagal karena terlalu pendek/rusak)."
    
    print("\n✅ " + pesan_akhir)
    return True, pesan_akhir

if __name__ == "__main__":
    pass