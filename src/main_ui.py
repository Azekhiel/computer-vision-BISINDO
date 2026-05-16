import tkinter as tk
from tkinter import ttk, messagebox, simpledialog, filedialog
import cv2
import pandas as pd
import numpy as np
import os
import threading
import queue
from PIL import Image, ImageTk
import uuid

import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

# Import Modul Internal
import feature_engine as fe
import database_manager as dbm
import data_ingestion as di
import augmentation_factory as af
import faiss_manager as fm
import inference_engine as ie
import visualization_utils as vu  

# ==========================================
# KONFIGURASI PATH (Tahan Banting & Partisi)
# ==========================================
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATABASE_DIR = os.path.join(ROOT_DIR, 'dataset_parquets')
GIF_DIR = os.path.join(ROOT_DIR, 'assets', 'gifs')
MODEL_DIR = os.path.join(ROOT_DIR, 'models')

os.makedirs(DATABASE_DIR, exist_ok=True)
os.makedirs(GIF_DIR, exist_ok=True)

def record_manual_dynamic(vocab_name, split_type):
    """Merekam gerakan secara manual dengan durasi bebas yang diakhiri secara manual."""
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Gunakan SequenceBuilder untuk Rekam Manual
    builder = fe.SequenceBuilder()
    is_recording = False

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
            
            color = (0, 0, 255) if is_recording else (245, 117, 16)
            cv2.rectangle(frame, (0,0), (640, 60), color, -1)
            
            status_text = f"MEREKAM: {vocab_name.upper()}" if is_recording else f"SIAP: {vocab_name.upper()} ({split_type.upper()})"
            cv2.putText(frame, status_text, (10,35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            cv2.putText(frame, 'Tekan "S" untuk Mulai/Stop. "Q" untuk Batal.', (10,55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
            
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image_rgb)
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            
            observation = fe.extract_frame_observation(results)
            
            if is_recording:
                builder.add_observation(observation)
                cv2.putText(frame, f"Frame: {len(builder._vectors)}", (500,35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

            cv2.imshow('Manual Recorder', frame)
            
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'): 
                cap.release()
                cv2.destroyAllWindows()
                return False, "Dibatalkan user"
            elif key == ord('s'): 
                if not is_recording:
                    is_recording = True
                else:
                    break 

    cap.release()
    cv2.destroyAllWindows()

    # Ekstrak data yang sudah diperhalus dari builder
    raw_sequence, smooth_scores = builder.build()

    if len(raw_sequence) < 5:
        return False, "Gerakan terlalu pendek."

    if vocab_name == 'idle':
        trimmed = raw_sequence
    else:
        # Gunakan auto_trim_sequence dari data_ingestion (agar logikanya sama persis)
        trimmed = di.auto_trim_sequence(raw_sequence, smooth_scores)

    if len(trimmed) < 5: return False, "Gerakan terlalu pendek setelah di-trim."

    video_id = f"{vocab_name}_{split_type}_manual_{uuid.uuid4().hex[:6]}"
    df_rows = []
    for f_num, features in enumerate(trimmed):
        df_rows.append({
            'video_id': video_id, 'label': vocab_name, 'frame_num': f_num,
            'split': split_type, 'feature_version': fe.FEATURE_SCHEMA,
            'features': ','.join(map(str, features))
        })
        
    df_new = pd.DataFrame(df_rows)
    file_vocab = os.path.join(DATABASE_DIR, f"{vocab_name}.parquet")
    
    if os.path.exists(file_vocab):
        df_lama = pd.read_parquet(file_vocab)
        df_gabung = pd.concat([df_lama, df_new], ignore_index=True)
        df_gabung.to_parquet(file_vocab, index=False)
    else:
        df_new.to_parquet(file_vocab, index=False)

    gif_path = os.path.join(GIF_DIR, f"{vocab_name}.gif")
    if os.path.exists(gif_path):
        os.remove(gif_path)
        
    dbm.update_metadata("db_update")
    return True, f"Sampel {split_type.upper()} tersimpan ({len(trimmed)} frame)."


class AppUI:
    def __init__(self, root):
        self.root = root
        self.root.title("BISINDO MLOps Dashboard - ITB")
        self.root.geometry("1100x700")
        
        dbm.init_database()
        
        self.selected_vocab = ""
        self.gif_frames = []
        self.gif_job = None 
        self.live_worker = None
        self.live_status_queue = queue.Queue()
        self.live_poll_job = None
        
        self.setup_ui()
        self.refresh_ui()

    def setup_ui(self):
        # --- PANEL KIRI: Database ---
        frame_kiri = tk.Frame(self.root, width=250, bg="#f8f9fa", relief="groove", bd=2)
        frame_kiri.pack(side="left", fill="y", padx=10, pady=10)
        
        tk.Label(frame_kiri, text="Daftar Vocab", font=("Arial", 12, "bold"), bg="#f8f9fa").pack(pady=5)
        
        scrollbar = tk.Scrollbar(frame_kiri)
        scrollbar.pack(side="right", fill="y")
        self.listbox = tk.Listbox(frame_kiri, font=("Arial", 11), height=25, yscrollcommand=scrollbar.set, exportselection=False)
        self.listbox.pack(side="top", fill="both", expand=True, padx=5)
        scrollbar.config(command=self.listbox.yview)
        self.listbox.bind('<<ListboxSelect>>', self.on_select_vocab)
        
        frame_btn_kiri = tk.Frame(frame_kiri, bg="#f8f9fa")
        frame_btn_kiri.pack(side="bottom", fill="x", pady=10)
        
        tk.Button(frame_btn_kiri, text="+ Tambah", bg="#d1e7dd", command=self.add_vocab).pack(side="left", padx=5, fill="x", expand=True)
        tk.Button(frame_btn_kiri, text="- Hapus", bg="#f8d7da", command=self.delete_vocab).pack(side="right", padx=5, fill="x", expand=True)

        # --- PANEL TENGAH: Dashboard ---
        frame_tengah = tk.Frame(self.root)
        frame_tengah.pack(side="left", fill="both", expand=True, padx=10, pady=10)
        
        self.lbl_judul = tk.Label(frame_tengah, text="Pilih vocab untuk melihat detail", font=("Arial", 16, "bold"))
        self.lbl_judul.pack(pady=5)
        
        # Area Status Model (Traffic Light)
        frame_status = tk.LabelFrame(frame_tengah, text="Status Model Sistem", font=("Arial", 10, "bold"), bg="#e9ecef", padx=10, pady=5)
        frame_status.pack(fill="x", pady=5)
        
        self.lbl_stat_seg = tk.Label(frame_status, text="Seg: -", font=("Arial", 10, "bold"), bg="#e9ecef")
        self.lbl_stat_seg.pack(side="left", expand=True)
        self.lbl_stat_faiss = tk.Label(frame_status, text="FAISS: -", font=("Arial", 10), bg="#e9ecef")
        self.lbl_stat_faiss.pack(side="left", expand=True)
        self.lbl_stat_lstm = tk.Label(frame_status, text="LSTM: -", font=("Arial", 10), bg="#e9ecef")
        self.lbl_stat_lstm.pack(side="left", expand=True)
        self.lbl_stat_trans = tk.Label(frame_status, text="Transf: -", font=("Arial", 10), bg="#e9ecef")
        self.lbl_stat_trans.pack(side="left", expand=True)
        
        # GIF Preview
        frame_gif = tk.Frame(frame_tengah, width=350, height=350, bg="white", relief="sunken", bd=2)
        frame_gif.pack(pady=10)
        frame_gif.pack_propagate(False) 
        self.lbl_gif = tk.Label(frame_gif, text="[ Preview GIF ]", bg="white")
        self.lbl_gif.pack(expand=True, fill="both")
        
        # Area Statistik Vocab
        frame_stats = tk.LabelFrame(frame_tengah, text="Statistik Data Vocab", font=("Arial", 10, "bold"), bd=2, bg="#fff3cd", padx=10, pady=10)
        frame_stats.pack(fill="x", pady=5)
        
        self.lbl_stat_train = tk.Label(frame_stats, text="Train (Asli/Gen): 0 / 0", font=("Arial", 11), bg="#fff3cd")
        self.lbl_stat_train.pack(side="left", expand=True)
        self.lbl_stat_val = tk.Label(frame_stats, text="Validation: 0", font=("Arial", 11), bg="#fff3cd")
        self.lbl_stat_val.pack(side="left", expand=True)
        self.lbl_stat_test = tk.Label(frame_stats, text="Testing: 0", font=("Arial", 11), bg="#fff3cd")
        self.lbl_stat_test.pack(side="left", expand=True)

        # --- PANEL KANAN: Aksi Eksekusi ---
        frame_kanan = tk.Frame(self.root, width=250)
        frame_kanan.pack(side="right", fill="y", padx=10, pady=10)
        
        btn_style = {"font": ("Arial", 10, "bold"), "pady": 5, "width": 20}
        btn_style_small = {"font": ("Arial", 9), "pady": 3, "width": 20}
        
        # Pilihan Hardware AI Backend (CPU / GPU)
        frame_mp = tk.Frame(frame_kanan)
        frame_mp.pack(pady=(0,10))
        tk.Label(frame_mp, text="AI Backend:", font=("Arial", 10, "bold")).pack(side="left")
        self.combo_mp_device = ttk.Combobox(frame_mp, values=["CPU", "GPU"], state="readonly", width=8, font=("Arial", 10))
        self.combo_mp_device.set("CPU")
        self.combo_mp_device.pack(side="left", padx=5)

        tk.Label(frame_kanan, text="1. Manajemen Data", font=("Arial", 11, "bold")).pack(pady=(5,0))
        tk.Button(frame_kanan, text="Import Folder (Massal)", bg="#e2e3e5", command=self.btn_import_click, **btn_style).pack(pady=5)
        self.btn_rekam = tk.Button(frame_kanan, text="Rekam Manual 1 Sampel", bg="#cff4fc", state="disabled", command=self.btn_rekam_click, **btn_style)
        self.btn_rekam.pack(pady=5)
        tk.Button(frame_kanan, text="Generate Augmentasi", bg="#fff3cd", command=self.btn_generate_click, **btn_style).pack(pady=5)
        
        ttk.Separator(frame_kanan, orient='horizontal').pack(fill='x', pady=10)
        
        tk.Label(frame_kanan, text="2. Latih AI Engine", font=("Arial", 11, "bold")).pack(pady=(5,0))
        # Tombol Two Stage Pipeline
        tk.Button(frame_kanan, text="Tahap 1: Latih Satpam (VAD)", bg="#ffc107", command=self.btn_seg_click, **btn_style_small).pack(pady=5)
        tk.Button(frame_kanan, text="Tahap 2: Build FAISS", bg="#cfe2f3", command=self.btn_faiss_click, **btn_style_small).pack(pady=5)
        tk.Button(frame_kanan, text="Tahap 2: Train Bi-LSTM", bg="#d1e7dd", command=self.btn_lstm_click, **btn_style_small).pack(pady=5)
        tk.Button(frame_kanan, text="Tahap 2: Train Transformer", bg="#f8d7da", command=self.btn_trans_click, **btn_style_small).pack(pady=5)
        
        ttk.Separator(frame_kanan, orient='horizontal').pack(fill='x', pady=10)
        
        tk.Label(frame_kanan, text="3. Deployment", font=("Arial", 11, "bold")).pack(pady=(5,0))
        self.combo_model = ttk.Combobox(frame_kanan, values=["faiss", "lstm", "transformer"], state="readonly", font=("Arial", 11))
        self.combo_model.set("faiss")
        self.combo_model.pack(pady=5)
        
        self.btn_live = tk.Button(frame_kanan, text="LIVE TEST SEAMLESS", bg="#0d6efd", fg="white", font=("Arial", 11, "bold"), pady=10, width=18, command=self.btn_live_test_click)
        self.btn_live.pack(pady=10)
        self.lbl_live_status = tk.Label(frame_kanan, text="Live: idle", font=("Arial", 9), fg="#666666", wraplength=220, justify="left")
        self.lbl_live_status.pack(pady=(0, 10), fill="x")

    def ask_split_type(self):
        """Jendela Pop-up untuk menanyakan Split Default/Fallback"""
        win = tk.Toplevel(self.root)
        win.title("Pilih Tujuan Data (Fallback)")
        win.geometry("380x180")
        
        tk.Label(win, text="Tujuan Split Data (Fallback Default):", font=("Arial", 10, "bold")).pack(pady=(10,0))
        tk.Label(win, text="*Diabaikan jika sub-folder terdeteksi bernama 'train' / 'val' / 'test'", font=("Arial", 8, "italic"), fg="#666666").pack(pady=(0,10))
        
        var = tk.StringVar(value="train")
        tk.Radiobutton(win, text="Data TRAINING", variable=var, value="train").pack()
        tk.Radiobutton(win, text="Data VALIDATION", variable=var, value="val").pack()
        tk.Radiobutton(win, text="Data TESTING", variable=var, value="test").pack()
        
        result = [None]
        def submit():
            result[0] = var.get()
            win.destroy()
            
        tk.Button(win, text="Lanjutkan", command=submit, bg="#0d6efd", fg="white", font=("Arial", 9, "bold")).pack(pady=10)
        self.root.wait_window(win)
        return result[0]

    def refresh_ui(self):
        vocabs = dbm.get_vocab_list()
        self.listbox.delete(0, tk.END)
        for v in vocabs: self.listbox.insert(tk.END, v)
            
        statuses = dbm.check_model_status()
        
        def format_status(label, text):
            color = "red" if "Outdated" in text or "Belum" in text else "green"
            label.config(text=f"{text}", fg=color)
            
        # Cek status segmenter (manual karena belum masuk dbm stats)
        seg_path = os.path.join(MODEL_DIR, 'segmenter_weights.pth')
        seg_stat = "OK" if os.path.exists(seg_path) else "Belum Dilatih"
        format_status(self.lbl_stat_seg, f"Seg: {seg_stat}")
        
        format_status(self.lbl_stat_faiss, f"FAISS: {statuses.get('faiss', '-')}")
        format_status(self.lbl_stat_lstm, f"LSTM: {statuses.get('lstm', '-')}")
        format_status(self.lbl_stat_trans, f"Transf: {statuses.get('transformer', '-')}")
        
        if self.selected_vocab:
            threading.Thread(target=self._load_vocab_data_async, args=(self.selected_vocab,), daemon=True).start()
        else:
            self.lbl_stat_train.config(text="Train (Asli/Gen): 0 / 0")
            self.lbl_stat_val.config(text="Validation: 0")
            self.lbl_stat_test.config(text="Testing: 0")

    def on_select_vocab(self, event):
        selection = event.widget.curselection()
        if not selection: return
        
        self.selected_vocab = event.widget.get(selection[0])
        self.lbl_judul.config(text=f"Kosakata: {self.selected_vocab.upper()}")
        self.btn_rekam.config(state="normal")
        
        self.lbl_stat_train.config(text="Train (Asli/Gen): Memuat...")
        self.lbl_stat_val.config(text="Validation: Memuat...")
        self.lbl_stat_test.config(text="Testing: Memuat...")
        self.lbl_gif.config(image='', text="Memuat animasi...", bg="white")
        
        if self.gif_job is not None: 
            self.root.after_cancel(self.gif_job)
            self.gif_job = None
            
        threading.Thread(target=self._load_vocab_data_async, args=(self.selected_vocab,), daemon=True).start()

    def _load_vocab_data_async(self, vocab):
        stats = dbm.get_database_stats()
        v_stat = stats.get(vocab, {})
        
        t_asli = v_stat.get("Train (Asli)", 0)
        t_gen = v_stat.get("Train (Generate)", 0)
        val_c = v_stat.get("Total Val", 0)
        test_c = v_stat.get("Total Test", 0)
        
        gif_path = vu.generate_vocab_gif(vocab)
        loaded_frames = []
        
        if gif_path and os.path.exists(gif_path):
            try:
                # --- TAMBAHAN KODE ANTI-ERROR VERSI PILLOW ---
                try:
                    resample_method = Image.Resampling.LANCZOS
                except AttributeError:
                    resample_method = Image.LANCZOS # Fallback untuk Pillow lawas
                # ---------------------------------------------
                
                gif_img = Image.open(gif_path)
                for i in range(gif_img.n_frames):
                    gif_img.seek(i)
                    frame = gif_img.copy().convert('RGB').resize((350, 350), resample_method)
                    loaded_frames.append(frame)
            except Exception as e:
                print(f"Peringatan saat membaca GIF {vocab}: {e}")
                
        self.root.after(0, lambda: self._update_ui_after_load(t_asli, t_gen, val_c, test_c, loaded_frames))

    def _update_ui_after_load(self, t_asli, t_gen, val_c, test_c, loaded_frames):
        self.lbl_stat_train.config(text=f"Train (Asli/Gen): {t_asli} / {t_gen}")
        self.lbl_stat_val.config(text=f"Validation: {val_c}")
        self.lbl_stat_test.config(text=f"Testing: {test_c}")
        
        if loaded_frames:
            self.gif_frames = [ImageTk.PhotoImage(img) for img in loaded_frames]
            self.animate_gif(0)
        else:
            self.lbl_gif.config(image='', text="Belum ada data V3.1, re-import/rekam ulang minimal 1", bg="white")

    def animate_gif(self, ind):
        if not self.gif_frames: return
        frame = self.gif_frames[ind]
        self.lbl_gif.configure(image=frame)
        self.lbl_gif.image = frame
        ind = (ind + 1) % len(self.gif_frames)
        self.gif_job = self.root.after(60, self.animate_gif, ind) 

    def add_vocab(self):
        new_v = simpledialog.askstring("Tambah Vocab", "Masukkan nama kosakata baru:\n(Bikin 'idle' untuk rekaman kosong)")
        if new_v:
            new_v = new_v.strip().replace(" ", "_").lower()
            self.listbox.insert(tk.END, new_v)
            self.listbox.selection_clear(0, tk.END)
            self.listbox.selection_set(tk.END)
            self.listbox.event_generate("<<ListboxSelect>>")

    def delete_vocab(self):
        if not self.selected_vocab: return
        if messagebox.askyesno("Hapus", f"Hapus SELURUH data '{self.selected_vocab}'?"):
            dbm.delete_vocab(self.selected_vocab)
            gif_path = os.path.join(GIF_DIR, f"{self.selected_vocab}.gif")
            if os.path.exists(gif_path): os.remove(gif_path)
            
            self.selected_vocab = ""
            self.lbl_judul.config(text="Pilih vocab untuk melihat detail")
            self.lbl_gif.config(image='', text="[ Preview GIF ]")
            self.btn_rekam.config(state="disabled")
            self.refresh_ui()

    def btn_import_click(self):
        # 1. Pilih Folder Induk
        parent_folder = filedialog.askdirectory(title="Pilih Folder Utama atau Folder Vocab")
        if not parent_folder: return

        # 2. Cek apakah ini folder vocab langsung atau Root Folder
        is_direct = di.is_vocab_folder(parent_folder)
        final_selection = []
        
        if is_direct:
            final_selection = [parent_folder]
        else:
            sub_folders = [f for f in os.listdir(parent_folder) 
                           if os.path.isdir(os.path.join(parent_folder, f))]
            
            if not sub_folders:
                messagebox.showwarning("Kosong", "Folder ini tidak berisi sub-folder maupun video langsung.")
                return

            # WINDOW POP-UP CHECKBOX LIST BARU
            win = tk.Toplevel(self.root)
            win.title("Pilih Folder yang akan di-Import")
            win.geometry("450x550")
            
            tk.Label(win, text="Pilih sub-folder yang akan diproses:", font=("Arial", 11, "bold")).pack(pady=10)
            
            # Frame untuk tombol Select All & Deselect All
            btn_frame = tk.Frame(win)
            btn_frame.pack(fill="x", padx=10, pady=5)
            
            # Area Canvas + Scrollbar agar list panjang bisa digulir
            container = tk.Frame(win, bd=2, relief="sunken")
            container.pack(fill="both", expand=True, padx=10, pady=5)
            
            canvas = tk.Canvas(container, borderwidth=0, highlightthickness=0)
            scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
            scrollable_frame = tk.Frame(canvas)
            
            scrollable_frame.bind(
                "<Configure>",
                lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
            )
            
            # Bikin scroll pakai mouse wheel (scroll tengah mouse)
            def _on_mousewheel(event):
                canvas.yview_scroll(int(-1*(event.delta/120)), "units")
            canvas.bind_all("<MouseWheel>", _on_mousewheel)
            
            canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
            canvas.configure(yscrollcommand=scrollbar.set)
            
            canvas.pack(side="left", fill="both", expand=True)
            scrollbar.pack(side="right", fill="y")
            
            # Render Checkboxes
            check_vars = {}
            for f in sorted(sub_folders):  # Urut abjad biar rapi
                var = tk.BooleanVar(value=True)  # Default: Tercentang semua
                check_vars[f] = var
                cb = tk.Checkbutton(scrollable_frame, text=f, variable=var, font=("Arial", 10))
                cb.pack(anchor="w", padx=10, pady=2)
                
            def select_all():
                for v in check_vars.values(): v.set(True)
            def deselect_all():
                for v in check_vars.values(): v.set(False)
                
            tk.Button(btn_frame, text="Pilih Semua", command=select_all, bg="#e2e3e5").pack(side="left", expand=True, fill="x", padx=2)
            tk.Button(btn_frame, text="Batal Pilih Semua", command=deselect_all, bg="#e2e3e5").pack(side="left", expand=True, fill="x", padx=2)
            
            # Eksekusi setelah menekan konfirmasi
            def confirm():
                for f, var in check_vars.items():
                    if var.get():
                        final_selection.append(os.path.join(parent_folder, f))
                win.destroy()
                
            tk.Button(win, text="Import Terpilih", command=confirm, bg="#0d6efd", fg="white", font=("Arial", 10, "bold")).pack(pady=10)
            
            self.root.wait_window(win)
            
            # Lepas binding mousewheel supaya tidak error setelah window pop-up tertutup
            canvas.unbind_all("<MouseWheel>")

        if not final_selection: return

        # 3. Tanyakan Fallback Split (tetap perlu untuk folder yang namanya bukan train/val/test)
        split_type = self.ask_split_type()
        if not split_type: return
        
        # 4. Lempar ke data_ingestion backend
        status, msg = di.bulk_import(final_selection, split_type, self.combo_mp_device.get())
        if status: messagebox.showinfo("Sukses", msg)
        else: messagebox.showwarning("Info", msg)
        self.refresh_ui()
        
    def btn_rekam_click(self):
        if not self.selected_vocab: return
        split_type = self.ask_split_type()
        if not split_type: return
        
        status, msg = record_manual_dynamic(self.selected_vocab, split_type)
        messagebox.showinfo("Status", msg)
        self.refresh_ui()
        self.listbox.event_generate("<<ListboxSelect>>") 

    def btn_generate_click(self):
        # Tampilkan Jendela Checkbox
        win = tk.Toplevel(self.root)
        win.title("Opsi Augmentasi")
        win.geometry("300x250")
        
        tk.Label(win, text="Pilih Split yang akan di-augmentasi:", font=("Arial", 10, "bold")).pack(pady=10)
        
        var_train = tk.BooleanVar(value=True)
        var_val = tk.BooleanVar(value=False)
        var_test = tk.BooleanVar(value=False)
        
        tk.Checkbutton(win, text="Data TRAINING", variable=var_train).pack(anchor="w", padx=50)
        tk.Checkbutton(win, text="Data VALIDATION", variable=var_val).pack(anchor="w", padx=50)
        tk.Checkbutton(win, text="Data TESTING", variable=var_test).pack(anchor="w", padx=50)
        
        def run_aug():
            selected = []
            if var_train.get(): selected.append('train')
            if var_val.get(): selected.append('val')
            if var_test.get(): selected.append('test')
            
            if not selected:
                messagebox.showwarning("Peringatan", "Pilih minimal 1 split!")
                return
                
            win.destroy()
            status, msg = af.generate_dataset(200, splits_to_augment=selected)
            messagebox.showinfo("Status Augmentasi", msg)
            self.refresh_ui()

        tk.Button(win, text="Mulai Generate", bg="#fff3cd", font=("Arial", 10, "bold"), command=run_aug).pack(pady=20)

    def run_threaded_task(self, target_func, success_msg):
        def task():
            status, msg = target_func()
            self.root.after(0, lambda: messagebox.showinfo(success_msg, msg))
            self.root.after(0, self.refresh_ui)
        threading.Thread(target=task, daemon=True).start()

    @staticmethod
    def _lazy_module_task(module_name, function_name):
        def task():
            try:
                module = __import__(module_name)
                func = getattr(module, function_name)
                return func()
            except Exception as exc:
                return False, f"Gagal memuat/menjalankan {module_name}.{function_name}: {exc}"
        return task

    def btn_seg_click(self):
        messagebox.showinfo("Info", "Training Segmenter VAD berjalan di background. Cek terminal.")
        self.run_threaded_task(self._lazy_module_task("segmenter_manager", "train_segmenter"), "Training Segmenter")

    def btn_faiss_click(self):
        self.run_threaded_task(fm.build_faiss_index, "Build FAISS")

    def btn_lstm_click(self):
        messagebox.showinfo("Info", "Training Bi-LSTM akan berjalan di background.")
        self.run_threaded_task(self._lazy_module_task("lstm_manager", "train_lstm_model"), "Training LSTM")

    def btn_trans_click(self):
        messagebox.showinfo("Info", "Training Transformer SOTA akan berjalan di background.")
        self.run_threaded_task(self._lazy_module_task("transformer_manager", "train_transformer_model"), "Training Transformer")

    def btn_live_test_click(self):
        if self.live_worker is not None and self.live_worker.is_alive():
            self.live_worker.stop()
            self.btn_live.config(text="MENUTUP LIVE...", state="disabled")
            self.lbl_live_status.config(text="Live: stopping")
            return

        selected = self.combo_model.get()
        mp_device = self.combo_mp_device.get()
        self.live_status_queue = queue.Queue()
        self.live_worker = ie.start_live_inference(selected, mp_device, self.live_status_queue)
        self.btn_live.config(text="STOP LIVE TEST", bg="#dc3545", state="normal")
        self.lbl_live_status.config(text="Live: starting")
        self._poll_live_status()

    def _poll_live_status(self):
        while not self.live_status_queue.empty():
            item = self.live_status_queue.get()
            event = item.get("event")
            if event == "done":
                ok = bool(item.get("ok", True))
                msg = item.get("message", "Inferensi selesai.")
                self.btn_live.config(text="LIVE TEST SEAMLESS", bg="#0d6efd", state="normal")
                self.lbl_live_status.config(text=f"Live: {msg}")
                self.live_worker = None
                if not ok:
                    messagebox.showwarning("Live Inference", msg)
                return

            prediction = item.get("prediction", "-")
            vad = float(item.get("vad_probability", 0.0))
            recording = "recording" if item.get("recording") else "idle"
            self.lbl_live_status.config(text=f"Live: {recording} | VAD {vad*100:.1f}%\n{prediction}")

        if self.live_worker is not None and self.live_worker.is_alive():
            self.live_poll_job = self.root.after(250, self._poll_live_status)
        else:
            self.btn_live.config(text="LIVE TEST SEAMLESS", bg="#0d6efd", state="normal")
            self.live_worker = None

if __name__ == "__main__":
    root = tk.Tk()
    app = AppUI(root)
    root.mainloop()
