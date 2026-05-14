import cv2
import mediapipe as mp
import numpy as np
import os
import json
import torch

# Pencegah Crash PyTorch di Jetson
torch.backends.cudnn.enabled = False

import faiss
import time
from collections import deque
import pyttsx3
import threading
import queue
import platform
try:
    import pythoncom
except ImportError:
    pythoncom = None

import feature_engine as fe
import faiss_manager as fm
import lstm_manager as lm
import transformer_manager as tm
import segmenter_manager as sgm

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

MODEL_DIR = 'models'
FAISS_INDEX = os.path.join(MODEL_DIR, 'sign_language.index')
FAISS_LABELS = os.path.join(MODEL_DIR, 'label_map.npy')

LSTM_WEIGHTS = os.path.join(MODEL_DIR, 'lstm_weights.pth')
LSTM_LABELS = os.path.join(MODEL_DIR, 'lstm_labels.json')

TRANSFORMER_WEIGHTS = os.path.join(MODEL_DIR, 'transformer_weights.pth')
TRANSFORMER_LABELS = os.path.join(MODEL_DIR, 'transformer_labels.json')

SEGMENTER_WEIGHTS = os.path.join(MODEL_DIR, 'segmenter_weights.pth')

SEGMENTER_WINDOW = 15      
SEGMENTER_CONFIDENCE = 0.6 
MAX_IDLE_FRAMES = 5        
MIN_VALID_FRAMES = 8       

tts_queue = queue.Queue()

def tts_worker():
    if pythoncom is not None:
        pythoncom.CoInitialize() 
    while True:
        text = tts_queue.get()
        if text is None: break
        engine = pyttsx3.init()
        voices = engine.getProperty('voices')
        for voice in voices:
            if 'indonesia' in voice.name.lower() or 'id' in voice.languages:
                engine.setProperty('voice', voice.id)
                break
        engine.setProperty('rate', 150) 
        engine.say(text)
        engine.runAndWait()
        del engine 
        tts_queue.task_done()

def load_segmenter_model(use_cpu=False):
    device = torch.device("cpu" if use_cpu else ("cuda" if torch.cuda.is_available() else "cpu"))
    if not os.path.exists(SEGMENTER_WEIGHTS):
        return None, device, "Model Segmenter belum dilatih! Jalankan Tahap 1 di UI."
    
    # KUNCI PERBAIKAN: Satpam VAD membaca 147 Dimensi
    model = sgm.VADSegmenterModel(input_dim=147, hidden_dim=64)
    model.load_state_dict(torch.load(SEGMENTER_WEIGHTS, map_location=device))
    model.to(device).eval()
    return model, device, "OK"

def load_classifier_model(model_type, use_cpu=False):
    device = torch.device("cpu" if use_cpu else ("cuda" if torch.cuda.is_available() else "cpu"))
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
    
    # KUNCI PERBAIKAN: LSTM dan Transformer membaca 147 Dimensi
    if model_type == 'lstm':
        model = lm.BiLSTMAttentionModel(input_dim=147, hidden_dim=256, num_classes=num_classes, num_layers=2)
    else:
        model = tm.TransformerSignModel(input_dim=147, d_model=256, nhead=8, num_layers=3, dim_feedforward=512, num_classes=num_classes)
        
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device).eval()
    return model, label_map, device, "OK"

def run_live_inference(selected_model='faiss', mp_device='CPU'):
    print(f"\n--- Memulai Smart Two-Stage Inference: {selected_model.upper()} ---")
    use_cpu_for_ai = (mp_device == 'CPU')

    segmenter, device_seg, msg_seg = load_segmenter_model(use_cpu_for_ai)
    if segmenter is None: return False, msg_seg

    classifier, label_map, device_cls, msg_cls = load_classifier_model(selected_model, use_cpu_for_ai)
    if classifier is None: return False, msg_cls

    threading.Thread(target=tts_worker, daemon=True).start()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    segmenter_buffer = deque(maxlen=SEGMENTER_WINDOW)
    builder = fe.SequenceBuilder()
    combo_buffer = []   
    
    is_recording_word = False
    last_active_time = time.time()
    current_idle_time = 0.0
    current_prediction = "SIAP. SILAKAN BERGERAK."
    seg_prob = 0.0 

    with mp_holistic.Holistic(
        min_detection_confidence=0.5, 
        min_tracking_confidence=0.35, # Pertahanan Oklusi
        smooth_landmarks=True,
        model_complexity=0 
    ) as holistic:
        while True:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.resize(frame, (640, 480))
            
            # Tanpa di-flip (Mirror Dihapus untuk Konsistensi Spasial)
            h, w, _ = frame.shape
            
            results = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            # KUNCI PERBAIKAN: Unpack 4 elemen dengan benar
            vector, mask, pose_lw, pose_rw = fe.extract_keypoints_relative(results)
            segmenter_buffer.append((vector, mask, pose_lw, pose_rw))

            is_sign_detected = False
            if len(segmenter_buffer) == SEGMENTER_WINDOW:
                
                # KUNCI PERBAIKAN: Menggabungkan vektor(144) + mask(3) untuk input Satpam
                vad_input = np.array([np.concatenate([v, m.astype(np.float32)]) for v, m, plw, prw in segmenter_buffer])
                seg_input = torch.tensor(vad_input, dtype=torch.float32).unsqueeze(0).to(device_seg)
                
                with torch.no_grad():
                    seg_out = segmenter(seg_input)
                    seg_prob = torch.sigmoid(seg_out).item()
                is_sign_detected = seg_prob >= SEGMENTER_CONFIDENCE

            current_time = time.time()

            if is_sign_detected:
                if not is_recording_word:
                    is_recording_word = True
                    builder.reset()
                    for v, m, plw, prw in segmenter_buffer:
                        builder.add_frame(v, m, plw, prw)
                    current_prediction = "MEREKAM KATA..."
                else:
                    builder.add_frame(vector, mask, pose_lw, pose_rw)
                    
                last_active_time = current_time
                current_idle_time = 0.0
            else:
                if is_recording_word:
                    builder.add_frame(vector, mask, pose_lw, pose_rw)
                    current_idle_time = current_time - last_active_time
                    
                    if current_idle_time >= 1.0:
                        is_recording_word = False
                        
                        seq_list, _ = builder.build()
                        
                        if len(seq_list) >= MIN_VALID_FRAMES:
                            seq_array = np.array(seq_list) # Sequence (N, 147)
                            
                            if selected_model == 'faiss':
                                # FAISS hanya makan vektor Spasial 144
                                spatial_seq = seq_array[:, :144] 
                                std_seq = fm.interpolate_sequence(spatial_seq, 30).astype('float32')
                                flat_vec = std_seq.flatten().reshape(1, -1)
                                faiss.normalize_L2(flat_vec)
                                distances, indices = classifier.search(flat_vec, k=1)
                                
                                if distances[0][0] < 1.3:
                                    pred_label = label_map[indices[0][0]]
                                    combo_buffer.append(pred_label.upper())
                                    current_prediction = f"+ {pred_label.upper()}"
                                else: current_prediction = "TIDAK DIKENAL"
                                
                            else:
                                # LSTM / Transformer makan fitur penuh 147
                                tensor_seq = torch.tensor(seq_array, dtype=torch.float32).unsqueeze(0).to(device_cls)
                                tensor_len = torch.tensor([len(seq_array)]).to(device_cls)
                                with torch.no_grad():
                                    outputs = classifier(tensor_seq, tensor_len)
                                    probs = torch.softmax(outputs, dim=1)
                                    conf, idx = torch.max(probs, 1)
                                    if conf.item() > 0.65:
                                        pred_label = label_map[idx.item()]
                                        combo_buffer.append(pred_label.upper())
                                        current_prediction = f"+ {pred_label.upper()}"
                                    else: current_prediction = "TIDAK YAKIN"
                                    
                        else: current_prediction = "GERAKAN TERLALU PENDEK"
                        builder.reset() 
                        
                else: 
                    current_idle_time = current_time - last_active_time
                    if current_idle_time >= 5.0 and len(combo_buffer) > 0:
                        kalimat = " ".join(combo_buffer)
                        current_prediction = f"KALIMAT: {kalimat}"
                        tts_queue.put(kalimat.replace("_", " "))
                        combo_buffer = []
                        last_active_time = current_time

            bar_color = (0, 0, 255) if is_recording_word else (0, 255, 0)
            seg_w = int(seg_prob * 200)
            cv2.rectangle(frame, (20, 80), (20 + seg_w, 95), bar_color, -1)
            cv2.rectangle(frame, (20, 80), (220, 95), (255, 255, 255), 1) 
            cv2.putText(frame, f"SATPAM/VAD: {seg_prob*100:.1f}%", (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

            header_color = (0, 165, 255) if is_recording_word else (245, 117, 16)
            cv2.rectangle(frame, (0,0), (w, 60), header_color, -1)
            cv2.putText(frame, f"STATUS: {current_prediction}", (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255,255,255), 3)

            y_pos = 150
            if not is_recording_word and current_idle_time > 0:
                cv2.putText(frame, f"Stopwatch Turun: {current_idle_time:.1f} detik", (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                y_pos += 30

            if len(combo_buffer) > 0:
                cv2.putText(frame, f"Isi Buffer ({len(combo_buffer)} kata tersimpan):", (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                y_pos += 25
                kata_list = " - ".join(combo_buffer)
                cv2.putText(frame, kata_list, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 255, 50), 2)

            cv2.imshow('BISINDO Live Translator', frame)
            if cv2.waitKey(10) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()
    tts_queue.put(None) 
    return True, "Inferensi Selesai."