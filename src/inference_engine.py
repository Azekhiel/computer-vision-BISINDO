import cv2
import mediapipe as mp
import numpy as np
import os
import json
import torch
import faiss
import time
from collections import deque
import pyttsx3
import threading
import queue
import pythoncom  # WAJIB untuk mencegah macet/deadlock di Windows

# Import modul internal
import feature_engine as fe
import faiss_manager as fm
import lstm_manager as lm
import transformer_manager as tm
import segmenter_manager as sgm

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# ==========================================
# KONFIGURASI DIREKTORI & PARAMETER
# ==========================================
MODEL_DIR = 'models'
FAISS_INDEX = os.path.join(MODEL_DIR, 'sign_language.index')
FAISS_LABELS = os.path.join(MODEL_DIR, 'label_map.npy')

LSTM_WEIGHTS = os.path.join(MODEL_DIR, 'lstm_weights.pth')
LSTM_LABELS = os.path.join(MODEL_DIR, 'lstm_labels.json')

TRANSFORMER_WEIGHTS = os.path.join(MODEL_DIR, 'transformer_weights.pth')
TRANSFORMER_LABELS = os.path.join(MODEL_DIR, 'transformer_labels.json')

SEGMENTER_WEIGHTS = os.path.join(MODEL_DIR, 'segmenter_weights.pth')

# Parameter Smart VAD (Two-Stage Pipeline)
SEGMENTER_WINDOW = 15      # Jumlah frame yang dianalisis satpam secara konstan
SEGMENTER_CONFIDENCE = 0.6 # Minimal yakin 60% bahwa itu isyarat valid, bukan noise
MAX_IDLE_FRAMES = 5        # Toleransi tangan diam sebelum rekaman benar-benar dipotong
MIN_VALID_FRAMES = 8       # Minimal durasi isyarat untuk dikirim ke Classifier

# ==========================================
# TEXT-TO-SPEECH (TTS) WORKER
# ==========================================
tts_queue = queue.Queue()

def tts_worker():
    """Berjalan di background thread. Membaca teks di antrean tanpa memblokir kamera."""
    # Mendaftarkan thread ini agar diizinkan memakai komponen audio Windows (COM)
    pythoncom.CoInitialize() 
    
    while True:
        text = tts_queue.get()
        if text is None: # Sinyal untuk mematikan thread
            break
            
        # Inisialisasi di DALAM loop agar mesin di-reset setiap kali mau ngomong
        engine = pyttsx3.init()
        
        voices = engine.getProperty('voices')
        for voice in voices:
            if 'indonesia' in voice.name.lower() or 'id' in voice.languages:
                engine.setProperty('voice', voice.id)
                break
                
        engine.setProperty('rate', 150) 
        
        # Eksekusi suara
        engine.say(text)
        engine.runAndWait()
        
        # Hapus engine dari memori agar tidak macet untuk tebakan kata berikutnya
        del engine 
        tts_queue.task_done()

# ==========================================
# FUNGSI LOADER MODEL
# ==========================================
def load_segmenter_model():
    """Memuat model Satpam/VAD dari segmenter_manager.py"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(SEGMENTER_WEIGHTS):
        return None, device, "Model Segmenter belum dilatih! Jalankan Tahap 1 di UI."
    
    model = sgm.VADSegmenterModel(input_dim=144, hidden_dim=64)
    model.load_state_dict(torch.load(SEGMENTER_WEIGHTS, map_location=device))
    model.to(device).eval()
    return model, device, "OK"

def load_classifier_model(model_type):
    """Memuat model Penebak Utama (FAISS/LSTM/Transformer)"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if model_type == 'faiss':
        if not os.path.exists(FAISS_INDEX) or not os.path.exists(FAISS_LABELS):
            return None, None, None, "Index FAISS belum di-build."
        model = faiss.read_index(FAISS_INDEX)
        labels = np.load(FAISS_LABELS)
        return model, labels, None, "OK"
        
    weights_path = LSTM_WEIGHTS if model_type == 'lstm' else TRANSFORMER_WEIGHTS
    labels_path = LSTM_LABELS if model_type == 'lstm' else TRANSFORMER_LABELS
    
    if not os.path.exists(weights_path) or not os.path.exists(labels_path):
        return None, None, None, f"Model {model_type.upper()} belum dilatih."
        
    with open(labels_path, 'r') as f:
        label_map = {int(k): v for k, v in json.load(f).items()}
        
    num_classes = len(label_map)
    if model_type == 'lstm':
        model = lm.BiLSTMAttentionModel(input_dim=144, hidden_dim=256, num_classes=num_classes, num_layers=2)
    else:
        model = tm.TransformerSignModel(input_dim=144, d_model=256, nhead=8, num_layers=3, dim_feedforward=512, num_classes=num_classes)
        
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device).eval()
    return model, label_map, device, "OK"

# ==========================================
# MAIN INFERENCE LOOP
# ==========================================
def run_live_inference(selected_model='faiss'):
    print(f"\n--- Memulai Smart Two-Stage Inference: {selected_model.upper()} ---")
    
    # 1. Load Model Segmenter (Satpam)
    segmenter, device_seg, msg_seg = load_segmenter_model()
    if segmenter is None:
        return False, msg_seg

    # 2. Load Model Penebak Utama
    classifier, label_map, device_cls, msg_cls = load_classifier_model(selected_model)
    if classifier is None:
        return False, msg_cls

    # Nyalakan pekerja suara di background
    threading.Thread(target=tts_worker, daemon=True).start()

    cap = cv2.VideoCapture(0)
    
    # State Machine Variables
    segmenter_buffer = deque(maxlen=SEGMENTER_WINDOW)
    recording_buffer = []
    is_recording = False
    idle_counter = 0
    
    current_prediction = "SIAP. SILAKAN BERGERAK."
    confidence_score = 0.0
    seg_prob = 0.0 

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while True:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            
            results = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            keypoints = fe.extract_keypoints_relative(results)
            
            # Masukkan selalu frame ke memori pendek Segmenter
            segmenter_buffer.append(keypoints)

            # ==========================================
            # STAGE 1: SEGMENTASI (Pendeteksi Gerakan)
            # ==========================================
            is_sign_detected = False
            if len(segmenter_buffer) == SEGMENTER_WINDOW:
                seg_input = torch.tensor(np.array(segmenter_buffer), dtype=torch.float32).unsqueeze(0).to(device_seg)
                
                with torch.no_grad():
                    seg_out = segmenter(seg_input)
                    seg_prob = torch.sigmoid(seg_out).item()
                    
                is_sign_detected = seg_prob >= SEGMENTER_CONFIDENCE

            # ==========================================
            # LOGIKA PEREKAMAN (State Machine)
            # ==========================================
            if not is_recording:
                if is_sign_detected:
                    is_recording = True
                    recording_buffer = list(segmenter_buffer)
                    idle_counter = 0
                    current_prediction = "MEREKAM..."
            else:
                recording_buffer.append(keypoints)
                
                if is_sign_detected:
                    idle_counter = 0 
                else:
                    idle_counter += 1 
                    
                # Jika tangan diam melewati batas toleransi
                if idle_counter >= MAX_IDLE_FRAMES:
                    is_recording = False
                    
                    # ==========================================
                    # STAGE 2: KLASIFIKASI (Menebak Makna)
                    # ==========================================
                    if len(recording_buffer) >= MIN_VALID_FRAMES:
                        seq_array = np.array(recording_buffer)
                        
                        if selected_model == 'faiss':
                            std_seq = fm.interpolate_sequence(seq_array, 30).astype('float32')
                            flat_vec = std_seq.flatten().reshape(1, -1)
                            faiss.normalize_L2(flat_vec)
                            
                            distances, indices = classifier.search(flat_vec, k=1)
                            
                            if distances[0][0] < 1.3:
                                pred_id = indices[0][0]
                                pred_label = label_map[pred_id]
                                current_prediction = pred_label.upper()
                                confidence_score = 1.0 - (distances[0][0] / 2.0)
                                
                                # Lempar teks ke antrean suara (ganti underscore jadi spasi)
                                tts_queue.put(pred_label.replace("_", " "))
                            else:
                                current_prediction = "TIDAK DIKENAL"
                        else:
                            # Logika Penebak Deep Learning (LSTM/Transformer)
                            tensor_seq = torch.tensor(seq_array, dtype=torch.float32).unsqueeze(0).to(device_cls)
                            tensor_len = torch.tensor([len(seq_array)]).to(device_cls)
                            
                            with torch.no_grad():
                                outputs = classifier(tensor_seq, tensor_len)
                                probs = torch.softmax(outputs, dim=1)
                                conf, idx = torch.max(probs, 1)
                                
                                if conf.item() > 0.65:
                                    pred_id = idx.item()
                                    pred_label = label_map[pred_id]
                                    current_prediction = pred_label.upper()
                                    confidence_score = conf.item()
                                    
                                    # Lempar teks ke antrean suara
                                    tts_queue.put(pred_label.replace("_", " "))
                                else:
                                    current_prediction = "TIDAK YAKIN"
                    else:
                        current_prediction = "GERAKAN TERLALU PENDEK"
                        
                    recording_buffer = [] # Reset buffer

            # ==========================================
            # UI OVERLAY
            # ==========================================
            bar_color = (0, 0, 255) if is_recording else (0, 255, 0)
            seg_w = int(seg_prob * 200)
            cv2.rectangle(frame, (20, 80), (20 + seg_w, 95), bar_color, -1)
            cv2.rectangle(frame, (20, 80), (220, 95), (255, 255, 255), 1) 
            cv2.putText(frame, f"SATPAM/VAD: {seg_prob*100:.1f}%", (20, 115), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

            header_color = (0, 165, 255) if is_recording else (245, 117, 16)
            cv2.rectangle(frame, (0,0), (w, 60), header_color, -1)
            cv2.putText(frame, f"STATUS: {current_prediction}", (20, 42), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255,255,255), 3)

            cv2.imshow('BISINDO Live Translator', frame)
            if cv2.waitKey(10) & 0xFF == ord('q'): break

    # Matikan webcam dan hentikan pekerja suara
    cap.release()
    cv2.destroyAllWindows()
    tts_queue.put(None) 
    return True, "Inferensi Selesai."