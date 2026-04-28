import cv2
import mediapipe as mp
import numpy as np
import os
import json
import torch
import faiss
from collections import Counter
import time

# Import modul internal
import feature_engine as fe
import faiss_manager as fm
import lstm_manager as lm
import transformer_manager as tm

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# ==========================================
# KONFIGURASI INFERENCE (CALIBRATED)
# ==========================================
MODEL_DIR = 'models'
FAISS_INDEX = os.path.join(MODEL_DIR, 'sign_language.index')
FAISS_LABELS = os.path.join(MODEL_DIR, 'label_map.npy')

# KUNCI PERBAIKAN: Threshold harus kecil (0.01 - 0.04)
START_THRESHOLD = 0.65  # Hanya rekam jika skor > 0.65 (gerakan sangat mantap)
STOP_THRESHOLD = 0.55
IDLE_FRAMES_WAIT = 3
MAX_RECORDING_FRAMES = 90 

# ==========================================
# FUNGSI LOADER & PRE-PROCESSING
# ==========================================
def load_faiss_model():
    if not os.path.exists(FAISS_INDEX) or not os.path.exists(FAISS_LABELS):
        return None, None, "Index FAISS belum di-build."
    index = faiss.read_index(FAISS_INDEX)
    labels = np.load(FAISS_LABELS)
    return index, labels, "OK"

def load_pytorch_model(model_type):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    print(f"\n--- Memulai Seamless Live Inference: {selected_model.upper()} ---")
    
    if selected_model == 'faiss':
        model_obj, label_map, msg = load_faiss_model()
        device = None
    else:
        model_obj, label_map, device, msg = load_pytorch_model(selected_model)
        
    if model_obj is None:
        print(f"[ERROR] {msg}")
        return False, msg

    cap = cv2.VideoCapture(0)
    sequence_buffer, prev_vector = [], None
    is_detecting, idle_counter = False, 0
    current_prediction, confidence_score, last_pred_time = "IDLE", 0.0, 0
    
    # Hasil prediksi sebelumnya untuk smoothing
    pred_history = []

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while True:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            
            results = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            # Ekstrak Fitur (Pastikan ini SAMA dengan saat training)
            keypoints = fe.extract_keypoints_relative(results)
            gerak_score = fe.calculate_movement_score(prev_vector, keypoints)
            if gerak_score > 0.001:
                print(f"DEBUG Score: {gerak_score:.4f} | State: {'RECORD' if is_detecting else 'IDLE'}")
            prev_vector = keypoints

            # --- LOGIKA VAD ---
            if not is_detecting:
                if gerak_score > START_THRESHOLD:
                    is_detecting, sequence_buffer, idle_counter = True, [keypoints], 0
                    current_prediction = "MEREKAM..."
            else:
                sequence_buffer.append(keypoints)
                if gerak_score < STOP_THRESHOLD:
                    idle_counter += 1
                else:
                    idle_counter = 0 
                    
                if idle_counter >= IDLE_FRAMES_WAIT or len(sequence_buffer) >= MAX_RECORDING_FRAMES:
                    is_detecting = False
                    if len(sequence_buffer) >= 8: # Minimal 8 frame agar valid
                        seq_array = np.array(sequence_buffer)
                        
                        if selected_model == 'faiss':
                            # Normalisasi L2 sangat penting untuk FAISS Cosine Similarity
                            std_seq = fm.interpolate_sequence(seq_array, 30).astype('float32')
                            flat_vec = std_seq.flatten().reshape(1, -1)
                            faiss.normalize_L2(flat_vec)
                            distances, indices = model_obj.search(flat_vec, k=1)
                            
                            # Jarak L2 (D) biasanya < 1.0 untuk hasil yang akurat
                            if distances[0][0] < 1.1:
                                current_prediction = label_map[indices[0][0]].upper()
                                confidence_score = 1.0 - (distances[0][0] / 2.0)
                            else:
                                current_prediction = "TIDAK DIKENAL"
                        else:
                            # Logika Deep Learning (LSTM/Transformer)
                            tensor_seq = torch.tensor(seq_array, dtype=torch.float32).unsqueeze(0).to(device)
                            tensor_len = torch.tensor([len(seq_array)]).to(device)
                            with torch.no_grad():
                                outputs = model_obj(tensor_seq, tensor_len)
                                probs = torch.softmax(outputs, dim=1)
                                conf, idx = torch.max(probs, 1)
                                confidence_score = conf.item()
                                if confidence_score > 0.70:
                                    current_prediction = label_map[idx.item()].upper()
                                else:
                                    current_prediction = "TIDAK YAKIN"
                        
                        last_pred_time = time.time()
                    sequence_buffer = []

            # --- UI FEEDBACK ---
            # Activity Bar (Kiri Atas)
            bar_color = (0, 0, 255) if is_detecting else (0, 255, 0)
            score_w = int(min(gerak_score * 3000, 200))
            cv2.rectangle(frame, (20, 80), (20 + score_w, 95), bar_color, -1)
            cv2.putText(frame, f"ACTIVITY: {gerak_score:.3f}", (20, 115), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

            # Header Status
            header_color = (0, 165, 255) if is_detecting else (245, 117, 16)
            cv2.rectangle(frame, (0,0), (w, 60), header_color, -1)
            cv2.putText(frame, f"STATUS: {current_prediction}", (20, 42), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255,255,255), 3)

            cv2.imshow('BISINDO Live Translator', frame)
            if cv2.waitKey(10) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()