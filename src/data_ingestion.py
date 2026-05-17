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
GIF_DIR = os.path.join(ROOT_DIR, 'assets', 'gifs')

# ==========================================
# KONFIGURASI
# ==========================================
MAX_WORKERS = max(1, int(os.environ.get("BISINDO_MAX_WORKERS", "2")))

START_THRESH = 0.015  # Sedikit diturunkan karena skor post-smooth lebih rendah
STOP_THRESH = 0.008
TRIM_PAD = 2      

DUPLICATE_THRESH = 1e-5   
MIN_FRAMES = 8
STATIC_IMAGE_REPEAT = 30

_worker_holistic = None
_worker_hands = None


def _create_hands_solution():
    try:
        return mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=0,
            min_detection_confidence=0.35,
            min_tracking_confidence=0.25,
        )
    except TypeError:
        return mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.35,
            min_tracking_confidence=0.25,
        )

def _init_worker():
    global _worker_holistic, _worker_hands
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['GLOG_minloglevel'] = '2'
    _worker_holistic = mp.solutions.holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.35, # Pertahanan Oklusi
        smooth_landmarks=True,
        model_complexity=0
    )
    _worker_hands = _create_hands_solution()

def _is_duplicate_frame(prev_sig: np.ndarray, curr_sig: np.ndarray) -> bool:
    if prev_sig is None: return False
    return float(np.sum(np.abs(curr_sig - prev_sig))) < DUPLICATE_THRESH


def _mask_intervals(mask: np.ndarray) -> list[tuple[int, int]]:
    intervals = []
    start = None
    for idx, active in enumerate(list(np.asarray(mask, dtype=bool)) + [False]):
        if active and start is None:
            start = idx
        elif start is not None and not active:
            intervals.append((start, idx - 1))
            start = None
    return intervals


def _remove_short_non_original_segments(sequence, metadata, min_len: int = 3):
    if not sequence:
        return sequence, metadata
    arr = np.asarray(sequence, dtype=np.float32).copy()
    if arr.ndim != 2 or arr.shape[1] < fe.N_TOTAL_WITH_FLAGS:
        return sequence, metadata
    cleaned_metadata = list(metadata or [{} for _ in range(len(arr))])
    side_defs = [
        ("left", fe.IDX_LH, fe.SLICE_LH, slice(144, 160), (2, 4)),
        ("right", fe.IDX_RH, fe.SLICE_RH, slice(160, 176), (3, 5)),
    ]
    for side, flag_idx, hand_slice, angle_slice, pose_points in side_defs:
        mask = arr[:, fe.SLICE_FLAGS][:, flag_idx] >= 0.5
        for start, end in _mask_intervals(mask):
            if end - start + 1 >= int(min_len):
                continue
            has_original = False
            for meta in cleaned_metadata[start : end + 1]:
                hand = meta.get(side, {}) if isinstance(meta, dict) else {}
                if bool(hand.get("original_detected", False)):
                    has_original = True
                    break
            if has_original:
                continue
            arr[start : end + 1, hand_slice] = 0.0
            arr[start : end + 1, angle_slice] = 0.0
            for pose_idx in pose_points:
                p0 = fe.SLICE_POSE.start + pose_idx * 3
                arr[start : end + 1, p0 : p0 + 3] = 0.0
            arr[start : end + 1, fe.SLICE_FLAGS.start + flag_idx] = 0.0
            for meta in cleaned_metadata[start : end + 1]:
                if not isinstance(meta, dict):
                    continue
                hand = meta.get(side, {})
                if isinstance(hand, dict):
                    hand.update(
                        {
                            "accepted": False,
                            "rendered": False,
                            "source": "trimmed_short_segment",
                            "confidence": 0.0,
                            "quality_reason": "post_trim_short_segment",
                            "roi": None,
                        }
                    )
    return [arr[i].astype(np.float32) for i in range(len(arr))], cleaned_metadata


def auto_trim_sequence(sequence: list, scores: list[float]) -> list:
    start_idx, end_idx = auto_trim_bounds(len(sequence), scores)
    return sequence[start_idx : end_idx + 1]


def auto_trim_bounds(
    length: int,
    scores: list[float],
    sequence: list | None = None,
    metadata: list[dict] | None = None,
) -> tuple[int, int]:
    if length < 5:
        return 0, max(0, length - 1)

    n = int(length)
    start_idx, end_idx = 0, n - 1

    if sequence is not None:
        try:
            arr = np.asarray(sequence, dtype=np.float32)
            flags = arr[:, fe.SLICE_FLAGS] if arr.ndim == 2 and arr.shape[1] >= fe.N_TOTAL_WITH_FLAGS else None
            if flags is not None:
                visible = (flags[:, fe.IDX_LH] >= 0.5) | (flags[:, fe.IDX_RH] >= 0.5)
                moving = np.asarray(scores, dtype=np.float32) > START_THRESH
                salient = np.zeros(n, dtype=bool)
                if metadata is not None and len(metadata) == n:
                    for meta_idx, meta in enumerate(metadata):
                        if not isinstance(meta, dict):
                            continue
                        for side in ("left", "right"):
                            hand = meta.get(side, {})
                            if not isinstance(hand, dict) or not bool(hand.get("accepted", False)):
                                continue
                            roi = hand.get("roi")
                            if not roi or len(roi) != 4:
                                continue
                            center_y = (float(roi[1]) + float(roi[3])) * 0.5
                            if center_y < 0.74:
                                salient[meta_idx] = True
                                break
                salient_moving = moving & salient
                idx = np.where(salient_moving)[0]
                if idx.size >= 3:
                    return max(0, int(idx[0]) - TRIM_PAD), min(n - 1, int(idx[-1]) + TRIM_PAD)
                idx = np.where(moving)[0]
                if idx.size:
                    visible_idx = np.where(visible)[0]
                    if visible_idx.size:
                        start_hint = max(int(idx[0]), int(visible_idx[0]))
                        end_hint = min(int(idx[-1]), int(visible_idx[-1]))
                        return max(0, start_hint - TRIM_PAD), min(n - 1, end_hint + TRIM_PAD)
                    return max(0, int(idx[0]) - TRIM_PAD), min(n - 1, int(idx[-1]) + TRIM_PAD)
                idx = np.where(visible)[0]
                if idx.size:
                    return max(0, int(idx[0]) - TRIM_PAD), min(n - 1, int(idx[-1]) + TRIM_PAD)
        except Exception:
            pass

    for i, s in enumerate(scores):
        if s > START_THRESH:
            start_idx = max(0, i - TRIM_PAD)
            break
            
    for i in range(n - 1, -1, -1):
        if scores[i] > STOP_THRESH:
            end_idx = min(n - 1, i + TRIM_PAD)
            break

    if start_idx >= end_idx: return 0, n - 1
    return start_idx, end_idx


def _extract_tracked_observation(frame, tracker):
    tracking_frame = fe.enhance_frame_for_tracking(frame)
    frame_rgb = cv2.cvtColor(tracking_frame, cv2.COLOR_BGR2RGB)
    results = _worker_holistic.process(frame_rgb)
    base = fe.extract_frame_observation(results)
    hands_results = []
    if _worker_hands is not None and tracker.needs_fallback(base):
        hands_results = tracker.fallback_candidates(frame_rgb, _worker_hands, base)
    return fe.extract_tracked_frame_observation(tracking_frame, results, hands_results=hands_results, tracker=tracker)


def _extract_offline_frame_observation(frame, frame_idx: int):
    tracking_frame = fe.enhance_frame_for_tracking(frame)
    tracking_gray = cv2.cvtColor(tracking_frame, cv2.COLOR_BGR2GRAY)
    enhanced_rgb = cv2.cvtColor(tracking_frame, cv2.COLOR_BGR2RGB)
    raw_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = _worker_holistic.process(enhanced_rgb)
    observation = fe.extract_frame_observation(results)
    candidates = []
    if _worker_hands is not None:
        enhanced_candidates = fe.detect_full_frame_hand_candidates(
            enhanced_rgb,
            _worker_hands,
            source="hands_enhanced",
            variant="enhanced_full",
            frame_idx=frame_idx,
        )
        candidates.extend(enhanced_candidates)
        holistic_hands = int(bool(observation.mask[fe.IDX_LH])) + int(bool(observation.mask[fe.IDX_RH]))
        if len(enhanced_candidates) < 2 and holistic_hands < 2:
            candidates.extend(
                fe.detect_full_frame_hand_candidates(
                    raw_rgb,
                    _worker_hands,
                    source="hands_raw",
                    variant="raw_full",
                    frame_idx=frame_idx,
                )
            )
    return observation, candidates, tracking_gray

def _process_media_task(args):
    file_path, vocab_name, split_type, video_id = args
    global _worker_holistic

    ext = os.path.splitext(file_path)[1].lower()
    img_exts = {'.jpg', '.jpeg', '.png', '.gif'}
    builder = fe.SequenceBuilder()   

    # --- GAMBAR STATIS ---
    if ext in img_exts:
        frame = cv2.imread(file_path)
        if frame is None: return video_id, vocab_name, split_type, None, None, None, "Gagal baca gambar"

        observation, candidates, gray = _extract_offline_frame_observation(frame, 0)
        tracked = fe.OfflineTrackletSolver().solve([observation], [candidates], [gray])[0]

        for _ in range(STATIC_IMAGE_REPEAT): builder.add_observation(tracked)
        sequence, _ = builder.build()
        metadata = builder.last_build_metadata
        source_indices = [0] * len(sequence)
        return video_id, vocab_name, split_type, sequence, metadata, source_indices, "OK"

    # --- VIDEO NORMAL ---
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened(): return video_id, vocab_name, split_type, None, None, None, "Gagal buka video"

    source_indices = []
    source_frame_idx = -1
    observations = []
    candidates_by_frame = []
    gray_frames = []

    while True:
        ret, frame = cap.read()
        if not ret: break
        source_frame_idx += 1

        observation, candidates, gray = _extract_offline_frame_observation(frame, source_frame_idx)
        observations.append(observation)
        candidates_by_frame.append(candidates)
        gray_frames.append(gray)
        source_indices.append(source_frame_idx)

    cap.release()

    if not observations: return video_id, vocab_name, split_type, None, None, None, "0 frame terbaca"

    tracked_observations = fe.OfflineTrackletSolver().solve(observations, candidates_by_frame, gray_frames)
    for observation in tracked_observations:
        builder.add_observation(observation)

    raw_sequence, smooth_scores = builder.build()
    raw_metadata = builder.last_build_metadata

    if vocab_name == 'idle':
        start_idx, end_idx = 0, len(raw_sequence) - 1
    else:
        start_idx, end_idx = auto_trim_bounds(len(raw_sequence), smooth_scores, raw_sequence, raw_metadata)
    final_sequence = raw_sequence[start_idx : end_idx + 1]
    final_metadata = raw_metadata[start_idx : end_idx + 1]
    final_source_indices = source_indices[start_idx : end_idx + 1]
    final_sequence, final_metadata = _remove_short_non_original_segments(final_sequence, final_metadata)

    if len(final_sequence) < MIN_FRAMES:
        return video_id, vocab_name, split_type, None, None, None, f"Sisa {len(final_sequence)} frame"

    return video_id, vocab_name, split_type, final_sequence, final_metadata, final_source_indices, "OK"


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
                        df_existing = pd.read_parquet(parquet_path)
                        df_existing = fe.filter_current_feature_rows(df_existing)
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
            vid, vocab, split_type, sequence, metadata, source_indices, msg = result
            
            if sequence is not None:
                for frame_num, features in enumerate(sequence):
                    row = {
                        'video_id': vid, 'label': vocab, 'frame_num': frame_num,
                        'split': split_type, 'feature_version': fe.FEATURE_SCHEMA,
                        'source_frame_num': int(source_indices[frame_num]) if source_indices else frame_num,
                        'features': ','.join(map(str, features))
                    }
                    if metadata and frame_num < len(metadata):
                        row.update(fe.flatten_tracking_metadata(metadata[frame_num]))
                    results_by_vocab[vocab].append(row)
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
        gif_path = os.path.join(GIF_DIR, f"{vocab}.gif")
        if os.path.exists(gif_path):
            os.remove(gif_path)
            
    dbm.update_metadata("db_update")
    pesan_akhir = f"Selesai! {len(tasks) - failed_count} file berhasil diekstrak."
    if failed_count > 0: pesan_akhir += f" ({failed_count} gagal)."
    
    print("\n✅ " + pesan_akhir)
    return True, pesan_akhir

if __name__ == "__main__":
    pass
