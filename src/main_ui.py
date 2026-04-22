import tkinter as tk
from tkinter import messagebox, simpledialog
import cv2
import pandas as pd
import numpy as np
import os
import faiss
from collections import Counter
from PIL import Image, ImageTk

# import 2 modul bikinan kita tadi
import feature_engine as fe
import faiss_manager as fm

import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

CSV_FILE = 'raw_dataset.csv'
VOCAB_FILE = 'vocab_list.txt'
INDEX_FILE = 'sign_language.index'
LABEL_FILE = 'label_map.npy'
GIF_DIR = 'generated_gifs'

NUM_FRAMES = 20

# bikin file n folder kalo blm ada
for path in [CSV_FILE, VOCAB_FILE]:
    if not os.path.exists(path):
        with open(path, 'w') as f:
            if path == CSV_FILE: f.write("video_id,label,frame_num,features\n")

if not os.path.exists(GIF_DIR):
    os.makedirs(GIF_DIR)

COLOR_POSE = (255, 100, 0)
COLOR_FACE = (200, 200, 200)
COLOR_LEFT_HAND = (0, 0, 255)
COLOR_RIGHT_HAND = (0, 200, 0)

def load_vocabs():
    with open(VOCAB_FILE, 'r') as f:
        return [line.strip() for line in f.readlines() if line.strip()]

def save_vocabs(vocab_list):
    with open(VOCAB_FILE, 'w') as f:
        for v in vocab_list: f.write(f"{v}\n")

def create_gif_from_csv(vocab_name):
    df = pd.read_csv(CSV_FILE)
    df_vocab = df[df['label'] == vocab_name]
    if df_vocab.empty: return False 
        
    first_video_id = df_vocab['video_id'].iloc[0]
    df_sample = df_vocab[df_vocab['video_id'] == first_video_id]
    
    frames_img = []
    canvas_size = 400
    # koordinat skrg relatif, jadi kita geser offset 200 pixel biar posisinya ke tengah layar
    center_offset = 200
    
    for _, row in df_sample.iterrows():
        features = list(map(float, row['features'].split(',')))
        canvas = np.ones((canvas_size, canvas_size, 3), dtype=np.uint8) * 255
        
        # pose
        for connection in mp_holistic.POSE_CONNECTIONS:
            start_idx, end_idx = connection[0] * 3, connection[1] * 3
            x1, y1 = features[start_idx], features[start_idx+1]
            x2, y2 = features[end_idx], features[end_idx+1]
            if x1 != 0 and y1 != 0 and connection[0] < 25 and connection[1] < 25:
                px1 = int(x1 * canvas_size) + center_offset
                py1 = int(y1 * canvas_size) + center_offset
                px2 = int(x2 * canvas_size) + center_offset
                py2 = int(y2 * canvas_size) + center_offset
                cv2.line(canvas, (px1, py1), (px2, py2), COLOR_POSE, 2)

        # muka dan tangan
        for connection in mp_holistic.FACEMESH_TESSELATION:
            start_idx, end_idx = 99 + (connection[0]*3), 99 + (connection[1]*3)
            x1, y1, x2, y2 = features[start_idx], features[start_idx+1], features[end_idx], features[end_idx+1]
            if x1 != 0 and y1 != 0: 
                cv2.line(canvas, (int(x1*canvas_size)+center_offset, int(y1*canvas_size)+center_offset), 
                         (int(x2*canvas_size)+center_offset, int(y2*canvas_size)+center_offset), COLOR_FACE, 1)

        for idx_offset, color in [(1503, COLOR_LEFT_HAND), (1566, COLOR_RIGHT_HAND)]:
            for connection in mp_holistic.HAND_CONNECTIONS:
                start_idx, end_idx = idx_offset + (connection[0]*3), idx_offset + (connection[1]*3)
                x1, y1, x2, y2 = features[start_idx], features[start_idx+1], features[end_idx], features[end_idx+1]
                if x1 != 0 and y1 != 0: 
                    cv2.line(canvas, (int(x1*canvas_size)+center_offset, int(y1*canvas_size)+center_offset), 
                             (int(x2*canvas_size)+center_offset, int(y2*canvas_size)+center_offset), color, 2)
                
        rgb_frame = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        frames_img.append(Image.fromarray(rgb_frame))
        
    gif_path = f"{GIF_DIR}/{vocab_name}.gif"
    frames_img[0].save(gif_path, save_all=True, append_images=frames_img[1:], duration=60, loop=0)
    return True

def start_recording(vocab_name, current_sample_count):
    cap = cv2.VideoCapture(0)
    frames_data = []

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while True:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1) 
            cv2.putText(frame, f'SIAP REKAM: {vocab_name} | SAMPEL {current_sample_count + 1}', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            cv2.putText(frame, 'Tekan "S" buat rekam, "Q" batal', (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
            cv2.imshow('Recorder', frame)
            
            key = cv2.waitKey(10) & 0xFF
            if key == ord('s'): break
            elif key == ord('q'): 
                cap.release()
                cv2.destroyAllWindows()
                return

        for frame_num in range(NUM_FRAMES):
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image_rgb)
            
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            
            # pake engine baru yg relatif
            keypoints = fe.extract_keypoints_relative(results)
            frames_data.append(keypoints)
            
            cv2.putText(frame, f'MEREKAM... {frame_num+1}/{NUM_FRAMES}', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.imshow('Recorder', frame)
            cv2.waitKey(30) 

    cap.release()
    cv2.destroyAllWindows()

    if len(frames_data) == NUM_FRAMES:
        video_id = f"{vocab_name}_{current_sample_count}"
        df_new = pd.DataFrame({
            'video_id': [video_id] * NUM_FRAMES,
            'label': [vocab_name] * NUM_FRAMES,
            'frame_num': range(NUM_FRAMES),
            'features': [','.join(map(str, f)) for f in frames_data]
        })
        df_new.to_csv(CSV_FILE, mode='a', header=False, index=False)
        if current_sample_count == 0: create_gif_from_csv(vocab_name)
        messagebox.showinfo("Sukses", f"Sampel {vocab_name} tersimpan!")

def run_live_test():
    if not os.path.exists(INDEX_FILE) or not os.path.exists(LABEL_FILE):
        return messagebox.showerror("Error", "Database FAISS belom ada!")

    index = faiss.read_index(INDEX_FILE)
    labels = np.load(LABEL_FILE)
    
    cap = cv2.VideoCapture(0)
    
    sequence_buffer = [] 
    prediction_history = []
    current_prediction = "IDLE (Diam)"
    prev_vector = None
    
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while True:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)
            
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image_rgb)
            
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            
            # ambil relatif
            keypoints = fe.extract_keypoints_relative(results)
            sequence_buffer.append(keypoints)
            
            # deteksi gerak (ngecek seberapa kenceng tangan gerak)
            gerak_score = fe.calculate_movement_score(prev_vector, keypoints)
            prev_vector = keypoints

            if len(sequence_buffer) == NUM_FRAMES:
                # kalo orangnya lagi gerak kenceng, tahan dulu, jangan ditebak
                if gerak_score > 0.15:
                    current_prediction = "Mendeteksi transisi..."
                else:
                    vector = np.concatenate(sequence_buffer).astype('float32').reshape(1, -1)
                    faiss.normalize_L2(vector)
                    
                    distances, indices = index.search(vector, k=5)
                    
                    # pake thresholding, kalo jarak terlalu jauh brarti ga jelas gerakannya
                    if distances[0][0] < 1.2: 
                        predicted_labels = [labels[idx] for idx in indices[0]]
                        tebakan_sementara = Counter(predicted_labels).most_common(1)[0][0]
                        prediction_history.append(tebakan_sementara)
                        
                        # batesin history cuma 10 frame terakhir
                        if len(prediction_history) > 10:
                            prediction_history.pop(0)
                            
                        # majority voting buat mutusin hasil akhir
                        if len(prediction_history) == 10:
                            suara_terbanyak = Counter(prediction_history).most_common(1)[0]
                            # harus minimal 7 frame sepakat baru diprint
                            if suara_terbanyak[1] >= 7:
                                current_prediction = suara_terbanyak[0]
                    else:
                        current_prediction = "IDLE (Gerakan Ngasal)"
                        prediction_history.clear()
                        
                sequence_buffer.pop(0) 
            
            cv2.rectangle(frame, (0,0), (640, 50), (245, 117, 16), -1)
            cv2.putText(frame, f"PREDIKSI: {current_prediction.upper()}", (10, 35), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            cv2.putText(frame, 'Tekan "Q" buat keluar', (10, 460), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
            
            cv2.imshow('Live Test Sign Language', frame)
            if cv2.waitKey(10) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()


class AppUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Sign Language TA Panel - ITB")
        self.root.geometry("900x650")
        
        self.selected_vocab = ""
        self.gif_frames = []
        self.gif_job = None 
        self.vocab_list = load_vocabs()
        
        frame_kiri = tk.Frame(root, width=250, bg="#f0f0f0", relief="groove", bd=2)
        frame_kiri.pack(side="left", fill="y", padx=10, pady=10)
        
        tk.Label(frame_kiri, text="Database Vocab", font=("Arial", 12, "bold"), bg="#f0f0f0").pack(pady=5)
        
        scrollbar = tk.Scrollbar(frame_kiri)
        scrollbar.pack(side="right", fill="y")
        self.listbox = tk.Listbox(frame_kiri, font=("Arial", 11), height=20, yscrollcommand=scrollbar.set, exportselection=False)
        self.refresh_listbox()
        self.listbox.pack(side="top", fill="both", expand=True, padx=5)
        scrollbar.config(command=self.listbox.yview)
        self.listbox.bind('<<ListboxSelect>>', self.on_select_vocab)
        
        frame_btn_kiri = tk.Frame(frame_kiri, bg="#f0f0f0")
        frame_btn_kiri.pack(side="bottom", fill="x", pady=10)
        
        tk.Button(frame_btn_kiri, text="+ Tambah", bg="#d1e7dd", command=self.add_vocab).pack(side="left", padx=5, fill="x", expand=True)
        tk.Button(frame_btn_kiri, text="- Hapus", bg="#f8d7da", command=self.delete_vocab).pack(side="right", padx=5, fill="x", expand=True)

        frame_kanan = tk.Frame(root, width=250)
        frame_kanan.pack(side="right", fill="y", padx=10, pady=10)
        
        tk.Label(frame_kanan, text="Aksi Eksekusi", font=("Arial", 12, "bold")).pack(pady=5)
        
        btn_style = {"font": ("Arial", 11, "bold"), "pady": 8, "width": 18}
        
        self.btn_rekam = tk.Button(frame_kanan, text="Rekam 1 Sampel", bg="#fff3cd", state="disabled", command=self.btn_rekam_click, **btn_style)
        self.btn_rekam.pack(pady=10)
        
        tk.Label(frame_kanan, text="-"*20).pack(pady=10)
        
        self.btn_generate = tk.Button(frame_kanan, text="Build Index FAISS", bg="#cfe2f3", command=self.btn_generate_click, **btn_style)
        self.btn_generate.pack(pady=10)
        
        self.btn_test = tk.Button(frame_kanan, text="LIVE TEST (FAISS)", bg="#0d6efd", fg="white", command=run_live_test, **btn_style)
        self.btn_test.pack(pady=10)
        
        frame_tengah = tk.Frame(root)
        frame_tengah.pack(side="left", fill="both", expand=True, padx=10, pady=10)
        
        self.lbl_judul = tk.Label(frame_tengah, text="Pilih vocab dari list kiri", font=("Arial", 16, "bold"))
        self.lbl_judul.pack(pady=5)
        
        frame_gif = tk.Frame(frame_tengah, width=350, height=350, bg="white", relief="sunken", bd=2)
        frame_gif.pack(pady=5)
        frame_gif.pack_propagate(False) 
        
        self.lbl_gif = tk.Label(frame_gif, text="[ Preview GIF ]", bg="white")
        self.lbl_gif.pack(expand=True, fill="both")
        
        frame_stats = tk.Frame(frame_tengah, relief="ridge", bd=2, bg="#e9ecef")
        frame_stats.pack(fill="x", pady=10, ipady=5)
        
        self.lbl_stat_manual = tk.Label(frame_stats, text="Manual Terekam: 0", font=("Arial", 11), bg="#e9ecef", fg="blue")
        self.lbl_stat_manual.pack(side="left", expand=True)
        
        self.lbl_stat_gen = tk.Label(frame_stats, text="Generated (FAISS): 0", font=("Arial", 11), bg="#e9ecef", fg="green")
        self.lbl_stat_gen.pack(side="left", expand=True)

    def refresh_listbox(self):
        self.listbox.delete(0, tk.END)
        for v in self.vocab_list: self.listbox.insert(tk.END, v)

    def add_vocab(self):
        new_v = simpledialog.askstring("Tambah Vocab", "Masukkan nama kosakata baru (tanpa spasi):")
        if new_v:
            new_v = new_v.replace(" ", "_").lower()
            if new_v not in self.vocab_list:
                self.vocab_list.append(new_v)
                self.vocab_list.sort()
                save_vocabs(self.vocab_list)
                self.refresh_listbox()
            
            idx = self.vocab_list.index(new_v)
            self.listbox.selection_clear(0, tk.END)
            self.listbox.selection_set(idx)
            self.listbox.event_generate("<<ListboxSelect>>")

    def delete_vocab(self):
        if not self.selected_vocab: return
        if messagebox.askyesno("Hapus", f"Yakin hapus '{self.selected_vocab}'?"):
            self.vocab_list.remove(self.selected_vocab)
            save_vocabs(self.vocab_list)
            self.refresh_listbox()
            
            df = pd.read_csv(CSV_FILE)
            df = df[df['label'] != self.selected_vocab]
            df.to_csv(CSV_FILE, index=False)
            
            gif_path = f"{GIF_DIR}/{self.selected_vocab}.gif"
            if os.path.exists(gif_path): os.remove(gif_path)
            
            self.selected_vocab = ""
            self.lbl_judul.config(text="Pilih vocab dari list kiri")
            self.update_stats()
            self.lbl_gif.config(image='', text="[ Preview GIF ]", bg="white")
            self.btn_rekam.config(state="disabled")

    def update_stats(self):
        if not self.selected_vocab:
            self.lbl_stat_manual.config(text="Manual Terekam: 0")
            self.lbl_stat_gen.config(text="Generated (FAISS): 0")
            return
            
        df = pd.read_csv(CSV_FILE)
        jml_manual = len(df[df['label'] == self.selected_vocab]) // NUM_FRAMES
        
        jml_gen = 0
        if os.path.exists(LABEL_FILE):
            labels = np.load(LABEL_FILE)
            jml_gen = np.sum(labels == self.selected_vocab)
            
        self.lbl_stat_manual.config(text=f"Manual Terekam: {jml_manual}")
        self.lbl_stat_gen.config(text=f"Generated (FAISS): {jml_gen}")

    def animate_gif(self, ind):
        if not self.gif_frames: return
        frame = self.gif_frames[ind]
        self.lbl_gif.configure(image=frame)
        self.lbl_gif.image = frame
        ind = (ind + 1) % len(self.gif_frames)
        self.gif_job = self.root.after(60, self.animate_gif, ind) 

    def on_select_vocab(self, event):
        selection = event.widget.curselection()
        if not selection: return
        
        self.selected_vocab = event.widget.get(selection[0])
        self.lbl_judul.config(text=f"Kosakata: {self.selected_vocab.upper()}")
        self.update_stats()
        
        self.btn_rekam.config(state="normal")
        
        if self.gif_job is not None: self.root.after_cancel(self.gif_job)
            
        gif_path = f"{GIF_DIR}/{self.selected_vocab}.gif"
        jml_manual = len(pd.read_csv(CSV_FILE)[pd.read_csv(CSV_FILE)['label'] == self.selected_vocab]) // NUM_FRAMES
        
        if not os.path.exists(gif_path) and jml_manual > 0:
            create_gif_from_csv(self.selected_vocab)
            
        if os.path.exists(gif_path):
            self.gif_frames = []
            try:
                gif_img = Image.open(gif_path)
                while True:
                    frame = gif_img.copy().convert('RGB').resize((350, 350))
                    self.gif_frames.append(ImageTk.PhotoImage(frame))
                    gif_img.seek(len(self.gif_frames)) 
            except EOFError: pass 
            self.animate_gif(0)
        else:
            self.lbl_gif.config(image='', text="Belom ada data mentah, rekam minimal 1", bg="white")

    def btn_rekam_click(self):
        jml_sekarang = len(pd.read_csv(CSV_FILE)[pd.read_csv(CSV_FILE)['label'] == self.selected_vocab]) // NUM_FRAMES
        start_recording(self.selected_vocab, jml_sekarang)
        self.update_stats()
        self.listbox.event_generate("<<ListboxSelect>>")

    def btn_generate_click(self):
        status, msg = fm.build_all_faiss_index(CSV_FILE, INDEX_FILE, LABEL_FILE)
        if status:
            messagebox.showinfo("Beres", msg)
        else:
            messagebox.showerror("Gagal", msg)
        self.update_stats()

if __name__ == "__main__":
    root = tk.Tk()
    app = AppUI(root)
    root.mainloop()