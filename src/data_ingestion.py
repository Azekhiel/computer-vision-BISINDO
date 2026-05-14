import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['GLOG_minloglevel'] = '2'

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
# KONFIGURASI
# ==========================================
MAX_WORKERS = 5

START_THRESH = 0.015  # Sedikit diturunkan karena skor post-smooth lebih rendah
STOP_THRESH = 0.008
TRIM_PAD = 2      

DUPLICATE_THRESH = 1e-5   
MIN_FRAMES = 8
STATIC_IMAGE_REPEAT = 30

_worker_holistic = None

def _init_worker():
    global _worker_holistic
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['GLOG_minloglevel'] = '2'
    _worker_holistic = mp.solutions.holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

def _is_duplicate_frame(prev_vec: np.ndarray, curr_vec: np.ndarray) -> bool:
    if prev_vec is None: return False
    return float(np.sum(np.abs(curr_vec - prev_vec))) < DUPLICATE_THRESH

def auto_trim_sequence(sequence: list, scores: list[float]) -> list:
    if not sequence or len(sequence) < 5: return sequence

    n = len(sequence)
    start_idx, end_idx = 0, n - 1

    for i, s in enumerate(scores):
        if s > START_THRESH:
            start_idx = max(0, i - TRIM_PAD)
            break
            
    for i in range(n - 1, -1, -1):
        if scores[i] > STOP_THRESH:
            end_idx = min(n - 1, i + TRIM_PAD)
            break

    if start_idx >= end_idx: return sequence   
    return sequence[start_idx : end_idx + 1]

def _process_media_task(args):
    file_path, vocab_name, split_type, video_id = args
    global _worker_holistic

    ext = os.path.splitext(file_path)[1].lower()
    img_exts = {'.jpg', '.jpeg', '.png', '.gif'}
    builder = fe.SequenceBuilder()   

    # --- GAMBAR STATIS ---
    if ext in img_exts:
        frame = cv2.imread(file_path)
        if frame is None: return video_id, vocab_name, split_type, None, "Gagal baca gambar"

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = _worker_holistic.process(frame_rgb)
        vector, mask = fe.extract_keypoints_relative(results)

        for _ in range(STATIC_IMAGE_REPEAT): builder.add_frame(vector, mask)
        sequence, _ = builder.build()
        return video_id, vocab_name, split_type, sequence, "OK"

    # --- VIDEO NORMAL ---
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened(): return video_id, vocab_name, split_type, None, "Gagal buka video"

    prev_vec = None

    while True:
        ret, frame = cap.read()
        if not ret: break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = _worker_holistic.process(frame_rgb)
        vector, mask = fe.extract_keypoints_relative(results)

        if _is_duplicate_frame(prev_vec, vector): continue

        builder.add_frame(vector, mask)
        prev_vec = vector

    cap.release()

    if not builder._vectors: return video_id, vocab_name, split_type, None, "0 frame terbaca"

    raw_sequence, smooth_scores = builder.build()

    if vocab_name == 'idle': final_sequence = raw_sequence
    else: final_sequence = auto_trim_sequence(raw_sequence, smooth_scores)

    if len(final_sequence) < MIN_FRAMES:
        return video_id, vocab_name, split_type, None, f"Sisa {len(final_sequence)} frame"

    return video_id, vocab_name, split_type, final_sequence, "OK"


# ==========================================
# HELPER ROUTING (SAMA SEPERTI SEBELUMNYA)
# ==========================================
_MEDIA_EXTS = ('.mkv', '.mp4', '.avi', '.mov', '.webm', '.jpg', '.jpeg', '.png', '.gif')

def is_vocab_folder(path: str) -> bool:
    try: return any(f.lower().endswith(_MEDIA_EXTS) for f in os.listdir(path))
    except PermissionError: return False

def bulk_import(source_paths, default_split: str = "train", mp_device: str = "CPU"):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['GLOG_minloglevel'] = '2'
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1' if mp_device == "CPU" else '0'

    if isinstance(source_paths, str): source_paths = [source_paths]

    grouped_folders = defaultdict(list)
    total_folders_found = 0

    for path in source_paths:
        if not os.path.exists(path): continue
        basename = os.path.basename(os.path.normpath(path)).lower()

        if is_vocab_folder(path):
            parent_name = os.path.basename(os.path.dirname(path)).lower()
            split = parent_name if parent_name in ['train', 'val', 'test'] else default_split
            grouped_folders[split].append(path)
            total_folders_found += 1
            
        elif basename in ['train', 'val', 'test']:
            vocab_dirs = [os.path.join(path, v) for v in os.listdir(path) if os.path.isdir(os.path.join(path, v))]
            for vocab_path in vocab_dirs:
                if is_vocab_folder(vocab_path):
                    grouped_folders[basename].append(vocab_path)
                    total_folders_found += 1
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
                            total_folders_found += 1
            else:
                for sub_dir in sub_dirs:
                    vocab_path = os.path.join(path, sub_dir)
                    if is_vocab_folder(vocab_path):
                        grouped_folders[default_split].append(vocab_path)
                        total_folders_found += 1

    if total_folders_found == 0: return False, "Tidak ada folder video/vocab valid."

    print("\n" + "="*50)
    print(f"🚀 MEMULAI PROSES IMPORT MASSAL (Worker: {MAX_WORKERS})")
    print("="*50)

    tasks = []
    vocab_existing_ids = {}

    for split_type in ['train', 'val', 'test', 'lainnya']:
        if split_type not in grouped_folders and split_type != 'lainnya': continue
        
        for vocab_path in grouped_folders.get(split_type, []):
            vocab = os.path.basename(vocab_path)
            
            if vocab not in vocab_existing_ids:
                existing_ids = set()
                parquet_path = os.path.join(DATABASE_DIR, f"{vocab}.parquet")
                if os.path.exists(parquet_path):
                    try:
                        df_existing = pd.read_parquet(parquet_path, columns=['video_id'])
                        existing_ids = set(df_existing['video_id'].unique())
                    except Exception: pass
                vocab_existing_ids[vocab] = existing_ids
                
            media_files = [f for f in os.listdir(vocab_path) if f.lower().endswith(_MEDIA_EXTS)]
            
            for media_file in media_files:
                file_path = os.path.join(vocab_path, media_file)
                video_id = f"{split_type}_manual_{media_file}"
                
                if video_id in vocab_existing_ids[vocab]: continue 
                tasks.append((file_path, vocab, split_type, video_id))

    if not tasks: return True, "Semua file sudah ada di database."

    results_by_vocab = defaultdict(list)
    failed_count = 0
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS, initializer=_init_worker) as executor:
        for result in tqdm(executor.map(_process_media_task, tasks), total=len(tasks), desc="Progress Keseluruhan"):
            vid, vocab, split_type, sequence, msg = result
            
            if sequence is not None:
                for frame_num, features in enumerate(sequence):
                    results_by_vocab[vocab].append({
                        'video_id': vid, 'label': vocab, 'frame_num': frame_num,
                        'split': split_type, 'features': ','.join(map(str, features))
                    })
            else:
                failed_count += 1

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
    if failed_count > 0: pesan_akhir += f" ({failed_count} gagal)."
    
    print("\n✅ " + pesan_akhir)
    return True, pesan_akhir

if __name__ == "__main__":
    pass