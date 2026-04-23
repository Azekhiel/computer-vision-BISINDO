import cv2
import mediapipe as mp
import numpy as np
import os
import json
import torch
import faiss
from collections import Counter
import time

# Import modul internal kita
import feature_engine as fe
import faiss_manager as fm
import lstm_manager as lm
import transformer_manager as tm

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# ==========================================
# KONFIGURASI INFERENCE
# ==========================================
MODEL_DIR = 'models'
FAISS_INDEX = os.path.join(MODEL_DIR, 'sign_language.index')
FAISS_LABELS = os.path.join(MODEL_DIR, 'label_map.npy')

LSTM_WEIGHTS = os.path.join(MODEL_DIR, 'lstm_weights.pth')
LSTM_LABELS = os.path.join(MODEL_DIR, 'lstm_labels.json')

TRANSFORMER_WEIGHTS = os.path.join(MODEL_DIR, 'transformer_weights.pth')
TRANSFORMER_LABELS = os.path.join(MODEL_DIR, 'transformer_labels.json')

# Threshold Pergerakan (VAD)
START_THRESHOLD = 0.015  # Seberapa cepat tangan gerak buat mulai ngerekam
STOP_THRESHOLD = 0.008   # Seberapa pelan tangan gerak buat nandain gerakan selesai
IDLE_FRAMES_WAIT = 7     # Tunggu N frame diam sebelum mutusin gerakan beneran beres (biar gak kepotong)
MIN_FRAMES_VALID = 5     # Gerakan minimal 5 frame baru dianggep valid

# ==========================================
# FUNGSI LOADER MODEL
# ==========================================
def load_faiss_model():
    if not os.path.exists(FAISS_INDEX) or not os.path.exists(FAISS_LABELS):
        return None, None, "Index FAISS belum di-build."
        
    index = faiss.read_index(FAISS_INDEX)
    labels = np.load(FAISS_LABELS)
    return index, labels, "OK"

def load_pytorch_model(model_type):
    """Memuat bobot dan arsitektur untuk LSTM atau Transformer."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    weights_path = LSTM_WEIGHTS if model_type == 'lstm' else TRANSFORMER_WEIGHTS
    labels_path = LSTM_LABELS if model_type == 'lstm' else TRANSFORMER_LABELS
    
    if not os.path.exists(weights_path) or not os.path.exists(labels_path):
        return None, None, None, f"Model {model_type.upper()} belum dilatih."
        
    with open(labels_path, 'r') as f:
        # Konversi key string "0" jadi integer 0
        label_map_str = json.load(f)
        label_map = {int(k): v for k, v in label_map_str.items()}
        
    num_classes = len(label_map)
    
    if model_type == 'lstm':
        model = lm.BiLSTMAttentionModel(input_dim=144, hidden_dim=256, num_classes=num_classes, num_layers=2)
    else:
        model = tm.TransformerSignModel(input_dim=144, d_model=256, nhead=8, num_layers=3, dim_feedforward=512, num_classes=num_classes)
        
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval() # Set ke mode evaluasi
    
    return model, label_map, device, "OK"

# ==========================================
# MAIN INFERENCE LOOP (SEAMLESS)
# ==========================================
def run_live_inference(selected_model='faiss'):
    """
    Menjalankan kamera live dengan deteksi gerakan cerdas (VAD).
    selected_model bisa berupa: 'faiss', 'lstm', 'transformer'.
    """
    print(f"\n--- Memulai Seamless Live Inference menggunakan: {selected_model.upper()} ---")
    
    # 1. Load Model yang dipilih user
    if selected_model == 'faiss':
        model_obj, label_map, msg = load_faiss_model()
    else:
        model_obj, label_map, device, msg = load_pytorch_model(selected_model)
        
    if model_obj is None:
        print(f"[ERROR] {msg}")
        return False, msg

    # 2. Inisialisasi State Machine VAD
    cap = cv2.VideoCapture(0)
    
    sequence_buffer = []
    prev_vector = None
    is_detecting = False
    idle_counter = 0
    
    current_prediction = "IDLE"
    last_pred_time = 0
    confidence_score = 0.0

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while True:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)
            
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image_rgb)
            
            # Gambar visual kerangka
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            
            # Ekstrak fitur 144D
            keypoints = fe.extract_keypoints_relative(results)
            
            # Cek kecepatan gerakan
            gerak_score = fe.calculate_movement_score(prev_vector, keypoints)
            prev_vector = keypoints
            
            # ==========================================
            # LOGIKA VAD (SEAMLESS ACTION SPOTTING)
            # ==========================================
            if not is_detecting:
                if gerak_score > START_THRESHOLD:
                    is_detecting = True
                    sequence_buffer = [keypoints]
                    idle_counter = 0
                    current_prediction = "Merekam Gerakan..."
            else:
                sequence_buffer.append(keypoints)
                
                # Kalau mulai melambat/berhenti
                if gerak_score < STOP_THRESHOLD:
                    idle_counter += 1
                else:
                    idle_counter = 0 # Reset kalau ternyata lanjut gerak lagi
                    
                # Eksekusi prediksi kalau udah diam selama N frame
                if idle_counter >= IDLE_FRAMES_WAIT:
                    is_detecting = False
                    
                    if len(sequence_buffer) >= MIN_FRAMES_VALID:
                        seq_array = np.array(sequence_buffer)
                        
                        # --- INFERENSI BERDASARKAN MODEL ---
                        if selected_model == 'faiss':
                            # Standarisasi jadi 30 frame
                            std_seq = fm.interpolate_sequence(seq_array, 30).astype('float32')
                            flat_vec = std_seq.flatten().reshape(1, -1)
                            faiss.normalize_L2(flat_vec)
                            
                            distances, indices = model_obj.search(flat_vec, k=3)
                            
                            if distances[0][0] < 1.3: # Threshold kedekatan L2
                                predicted_labels = [label_map[idx] for idx in indices[0]]
                                # Mayoritas dari 3 terdekat
                                best_pred = Counter(predicted_labels).most_common(1)[0][0]
                                current_prediction = best_pred.upper()
                                confidence_score = 1.0 - (distances[0][0] / 2.0)
                            else:
                                current_prediction = "TIDAK DIKENAL"
                                
                        else: # LSTM / Transformer
                            # Format input untuk PyTorch: [Batch, Seq_Len, Features]
                            tensor_seq = torch.tensor(seq_array, dtype=torch.float32).unsqueeze(0).to(device)
                            tensor_len = torch.tensor([len(seq_array)]).to(device)
                            
                            with torch.no_grad():
                                logits = model_obj(tensor_seq, tensor_len)
                                probs = torch.softmax(logits, dim=1)
                                conf, predicted_idx = torch.max(probs, 1)
                                
                                confidence_score = conf.item()
                                
                                if confidence_score > 0.65: # Threshold probabilitas PyTorch
                                    current_prediction = label_map[predicted_idx.item()].upper()
                                else:
                                    current_prediction = "TIDAK DIKENAL"
                                    
                        last_pred_time = time.time()
                    sequence_buffer = [] # Kosongkan buffer untuk kata selanjutnya
            
            # --- UI OVERLAY DI OPENCV ---
            # Jika prediksinya sudah lewat 3 detik, kembalikan teks ke IDLE
            if time.time() - last_pred_time > 3.0 and not is_detecting:
                current_prediction = "IDLE (Diam)"
                confidence_score = 0.0

            # Bar status atas
            status_color = (0, 165, 255) if is_detecting else (245, 117, 16)
            cv2.rectangle(frame, (0,0), (640, 60), status_color, -1)
            
            # Teks Prediksi
            cv2.putText(frame, f"HASIL: {current_prediction}", (15, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 3, cv2.LINE_AA)
            
            # Teks Confidence dan Model
            cv2.putText(frame, f"Model: {selected_model.upper()} | Conf: {confidence_score:.2f} | FPS: {int(cap.get(cv2.CAP_PROP_FPS))}", 
                        (15, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, 'Tekan "Q" untuk keluar', (450, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
            
            cv2.imshow(f'Seamless Sign Language Translator ({selected_model.upper()})', frame)
            
            if cv2.waitKey(10) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()
    return True, "Sesi Live Test selesai."

if __name__ == "__main__":
    # Test eksekusi mandiri. Ganti parameter dengan 'lstm' atau 'transformer' untuk coba model lain
    run_live_inference(selected_model='faiss')