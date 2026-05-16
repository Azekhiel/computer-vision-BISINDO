import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
import os
import threading
from PIL import Image, ImageTk

import faiss_manager as fm
import database_manager as dbm
import feature_engine as fe
import visualization_utils as vu

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATABASE_DIR = os.path.join(ROOT_DIR, 'dataset_parquets')
AUDIT_GIF_DIR = os.path.join(ROOT_DIR, 'assets', 'audit_gifs')
os.makedirs(AUDIT_GIF_DIR, exist_ok=True)

def parse_features(feature_str):
    return np.array(list(map(float, feature_str.split(','))))

class AuditUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Audit Kualitas Augmentasi BISINDO")
        self.root.geometry("850x550")
        
        self.furthest_samples = []
        self.gif_frames = []
        self.gif_job = None
        
        self.setup_ui()
        self.load_vocabs()

    def setup_ui(self):
        top_frame = tk.Frame(self.root, pady=10)
        top_frame.pack(fill="x")
        
        tk.Label(top_frame, text="Pilih Kosakata:", font=("Arial", 11)).pack(side="left", padx=10)
        self.combo_vocab = ttk.Combobox(top_frame, state="readonly", width=20)
        self.combo_vocab.pack(side="left", padx=5)
        
        btn_audit = tk.Button(top_frame, text="Analisis 10 Terjauh", bg="#ffc107", command=self.run_audit)
        btn_audit.pack(side="left", padx=10)

        mid_frame = tk.Frame(self.root, pady=10)
        mid_frame.pack(fill="both", expand=True, padx=10)
        
        left_frame = tk.Frame(mid_frame)
        left_frame.pack(side="left", fill="both", expand=True)
        
        tk.Label(left_frame, text="10 Augmentasi Paling Melenceng (Skor L2)", font=("Arial", 10, "bold")).pack()
        tk.Label(left_frame, text="Gunakan Shift/Ctrl untuk memilih banyak item sekaligus", font=("Arial", 8, "italic"), fg="gray").pack()
        
        columns = ("ID Video", "Jarak L2", "Status GIF")
        self.tree = ttk.Treeview(left_frame, columns=columns, show="headings", height=15, selectmode="extended")
        self.tree.heading("ID Video", text="ID Video")
        self.tree.heading("Jarak L2", text="Jarak L2")
        self.tree.heading("Status GIF", text="Status GIF")
        
        self.tree.column("ID Video", width=200)
        self.tree.column("Jarak L2", width=70, anchor="center")
        self.tree.column("Status GIF", width=80, anchor="center")
        self.tree.pack(fill="both", expand=True, pady=5)
        
        btn_frame = tk.Frame(left_frame)
        btn_frame.pack(fill="x", pady=5)
        
        btn_generate = tk.Button(btn_frame, text="Hasilkan GIF (Terpilih)", bg="#198754", fg="white", command=self.generate_selected)
        btn_generate.pack(side="left", fill="x", expand=True, padx=(0, 5))
        
        btn_play = tk.Button(btn_frame, text="Putar GIF (1 Terpilih)", bg="#0d6efd", fg="white", command=self.play_selected)
        btn_play.pack(side="left", fill="x", expand=True, padx=(5, 0))

        right_frame = tk.Frame(mid_frame, width=300, bg="white", relief="sunken", bd=2)
        right_frame.pack(side="right", fill="y", padx=10)
        right_frame.pack_propagate(False)
        
        self.lbl_gif = tk.Label(right_frame, text="[ Preview GIF ]", bg="white")
        self.lbl_gif.pack(expand=True, fill="both")
        
        self.lbl_status = tk.Label(self.root, text="Siap.", fg="gray")
        self.lbl_status.pack(side="bottom", anchor="w", padx=10, pady=5)

    def load_vocabs(self):
        vocabs = dbm.get_vocab_list()
        self.combo_vocab['values'] = vocabs
        if vocabs:
            self.combo_vocab.set(vocabs[0])

    def run_audit(self):
        vocab = self.combo_vocab.get()
        if not vocab: return
        
        self.lbl_status.config(text="Sedang mencari Nearest Neighbor Euclidean...")
        for item in self.tree.get_children():
            self.tree.delete(item)
            
        threading.Thread(target=self._calc_distances, args=(vocab,), daemon=True).start()

    def _calc_distances(self, vocab_name):
        filepath = os.path.join(DATABASE_DIR, f"{vocab_name}.parquet")
        if not os.path.exists(filepath):
            self.root.after(0, lambda: self.lbl_status.config(text="File tidak ditemukan."))
            return

        df = pd.read_parquet(filepath)
        df = fe.filter_current_feature_rows(df)
        if df.empty:
            self.root.after(0, lambda: self.lbl_status.config(text="Tidak ada data V3.1 untuk vocab ini."))
            return
        asli_df = df[~df['video_id'].astype(str).str.contains('_aug_')]
        aug_df = df[df['video_id'].astype(str).str.contains('_aug_')]

        if asli_df.empty or aug_df.empty:
            self.root.after(0, lambda: self.lbl_status.config(text="Data asli atau augmentasi tidak cukup."))
            return

        asli_seqs = []
        for vid, group in asli_df.groupby('video_id'):
            group = group.sort_values('frame_num')
            seq = np.array([parse_features(f) for f in group['features']])
            descriptor = fm.make_temporal_descriptor(seq[:, :176])
            asli_seqs.append(descriptor)

        aug_distances = []
        for vid, group in aug_df.groupby('video_id'):
            group = group.sort_values('frame_num')
            seq = np.array([parse_features(f) for f in group['features']])
            descriptor = fm.make_temporal_descriptor(seq[:, :176])
            
            distances_to_asli = [np.linalg.norm(descriptor - asli_seq) for asli_seq in asli_seqs]
            min_dist = min(distances_to_asli)
            
            aug_distances.append({"vid": vid, "dist": min_dist, "seq": seq})

        aug_distances.sort(key=lambda x: x['dist'], reverse=True)
        self.furthest_samples = aug_distances[:10]

        self.root.after(0, self._update_tree)

    def _update_tree(self):
        for item in self.furthest_samples:
            vid = item['vid']
            dist = f"{item['dist']:.2f}"
            status = "Ada" if os.path.exists(os.path.join(AUDIT_GIF_DIR, f"{vid}.gif")) else "Belum"
            self.tree.insert("", tk.END, values=(vid, dist, status))
            
        self.lbl_status.config(text="Analisis selesai. Pilih sampel di tabel lalu klik Hasilkan GIF.")

    def generate_selected(self):
        selected_items = self.tree.selection()
        if not selected_items:
            messagebox.showinfo("Info", "Pilih minimal 1 sampel di tabel.")
            return

        vocab = self.combo_vocab.get()
        vids_to_gen = []
        for item in selected_items:
            vid = self.tree.item(item)['values'][0]
            status = self.tree.item(item)['values'][2]
            if status != "Ada":
                vids_to_gen.append((vid, item))

        if not vids_to_gen:
            self.lbl_status.config(text="Semua GIF yang dipilih sudah pernah di-generate sebelumnya.")
            return

        self.lbl_status.config(text=f"Merender {len(vids_to_gen)} GIF... Proses ini berjalan di latar belakang.")
        threading.Thread(target=self._process_generation, args=(vids_to_gen, vocab), daemon=True).start()

    def _process_generation(self, vids_data, vocab):
        for i, (vid, tree_item) in enumerate(vids_data):
            self.root.after(0, lambda v=vid, idx=i+1, tot=len(vids_data): self.lbl_status.config(text=f"Merender GIF ({idx}/{tot}): {v}"))
            
            target_data = next((item for item in self.furthest_samples if item['vid'] == vid), None)
            if target_data:
                self._create_gif_file(vid, target_data['seq'], vocab)
                
                self.root.after(0, lambda item=tree_item, v=vid, d=target_data['dist']: self.tree.item(item, values=(v, f"{d:.2f}", "Ada")))
        
        self.root.after(0, lambda: self.lbl_status.config(text="Semua GIF yang dipilih berhasil di-render! Silakan pilih 1 lalu klik Putar GIF."))

    def _create_gif_file(self, vid, sequence, vocab):
        gif_path = os.path.join(AUDIT_GIF_DIR, f"{vid}.gif")
        if os.path.exists(gif_path):
            return 
        vu.render_sequence_gif(
            sequence=sequence,
            gif_path=gif_path,
            title=f"Audit: {vocab}",
            feature_version=fe.FEATURE_SCHEMA,
            interval=50,
        )

    def play_selected(self):
        selected_items = self.tree.selection()
        if not selected_items:
            messagebox.showinfo("Info", "Pilih 1 sampel untuk diputar.")
            return
        
        item_values = self.tree.item(selected_items[0])['values']
        vid = item_values[0]
        status = item_values[2]
        
        if status != "Ada":
            messagebox.showwarning("GIF Belum Siap", f"GIF untuk video '{vid}' belum di-render.\n\nSilakan klik 'Hasilkan GIF' terlebih dahulu.")
            return

        if self.gif_job:
            self.root.after_cancel(self.gif_job)
            self.gif_job = None

        gif_path = os.path.join(AUDIT_GIF_DIR, f"{vid}.gif")
        loaded_frames = []
        try:
            # PENCEGAHAN CRASH PILLOW VERSI LAWAS
            try:
                resample_method = Image.Resampling.LANCZOS
            except AttributeError:
                resample_method = Image.LANCZOS
                
            gif_img = Image.open(gif_path)
            while True:
                frame = gif_img.copy().convert('RGB').resize((280, 280), resample_method)
                loaded_frames.append(ImageTk.PhotoImage(frame))
                gif_img.seek(len(loaded_frames))
        except EOFError:
            pass

        self.lbl_gif.config(text="", image='') 
        self._play_gif(loaded_frames, 0)

    def _play_gif(self, frames, ind):
        if not frames: return
        self.lbl_gif.configure(image=frames[ind])
        self.lbl_gif.image = frames[ind]
        ind = (ind + 1) % len(frames)
        self.gif_job = self.root.after(60, self._play_gif, frames, ind)

if __name__ == "__main__":
    root = tk.Tk()
    app = AuditUI(root)
    root.mainloop()
