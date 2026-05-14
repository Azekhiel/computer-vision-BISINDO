import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
import os
import json
import threading

import matplotlib
matplotlib.use('Agg') # Mencegah crash saat render grafik di thread terpisah
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import torch
import faiss

# Import modul arsitektur yang sudah dibuat
import faiss_manager as fm
import lstm_manager as lm
import transformer_manager as tm

DATABASE_DIR = 'dataset_parquets'
MODEL_DIR = 'models'

class EvaluatorBackend:
    """Mesin untuk memproses data test dan menjalankan inferensi pada model terpilih."""
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.test_data = [] # List of dict: {'label': str, 'sequence': np.array}
        self.classes = []
        self.results = {} 

    def load_test_data(self):
        if not os.path.exists(DATABASE_DIR):
            return False, "Folder database tidak ditemukan."
            
        self.test_data = []
        self.classes = []
        
        parquet_files = [f for f in os.listdir(DATABASE_DIR) if f.endswith('.parquet')]
        if not parquet_files:
            return False, "Database kosong."

        # Menelusuri semua file parquet untuk mencari data split 'test'
        for file in parquet_files:
            vocab = file.replace('.parquet', '')
            filepath = os.path.join(DATABASE_DIR, file)
            
            try:
                df = pd.read_parquet(filepath)
                test_df = df[df['split'] == 'test']
                
                if test_df.empty:
                    continue
                    
                if vocab not in self.classes:
                    self.classes.append(vocab)
                    
                grouped = test_df.groupby(['label', 'video_id'])
                for (label, vid), group in grouped:
                    group = group.sort_values('frame_num')
                    seq = np.array([list(map(float, f.split(','))) for f in group['features']], dtype=np.float32)
                    self.test_data.append({'label': label, 'sequence': seq})
            except Exception:
                continue
                
        self.classes = sorted(self.classes)
        
        if not self.test_data:
            return False, "Tidak ada data dengan label 'test' di database."
            
        return True, f"Berhasil memuat {len(self.test_data)} sampel uji dari {len(self.classes)} kelas."

    def run_evaluations(self, selected_models):
        """Menjalankan test HANYA untuk model yang dipilih."""
        self.results = {m: {'y_true': [], 'y_pred': []} for m in selected_models}
        
        # 1. EVALUASI FAISS
        if 'faiss' in selected_models:
            try:
                index = faiss.read_index(os.path.join(MODEL_DIR, 'sign_language.index'))
                faiss_labels = np.load(os.path.join(MODEL_DIR, 'label_map.npy'))
                
                for item in self.test_data:
                    seq = item['sequence']
                    true_label = item['label']
                    
                    # Interpolasi ke 30 frame sesuai standar faiss_manager terbaru
                    std_seq = fm.interpolate_sequence(seq, 30).astype('float32')
                    flat_vec = std_seq.flatten().reshape(1, -1)
                    faiss.normalize_L2(flat_vec)
                    
                    distances, indices = index.search(flat_vec, k=1)
                    pred_label = faiss_labels[indices[0][0]] if distances[0][0] < 1.5 else "unknown"
                    
                    self.results['faiss']['y_true'].append(true_label)
                    self.results['faiss']['y_pred'].append(pred_label)
            except Exception as e:
                print(f"FAISS Eval Error: {e}")

        # 2 & 3. EVALUASI PYTORCH (LSTM & TRANSFORMER)
        for model_type in ['lstm', 'transformer']:
            if model_type in selected_models:
                try:
                    weights_path = os.path.join(MODEL_DIR, f'{model_type}_weights.pth')
                    labels_path = os.path.join(MODEL_DIR, f'{model_type}_labels.json')
                    
                    with open(labels_path, 'r') as f:
                        label_map_str = json.load(f)
                        label_map = {int(k): v for k, v in label_map_str.items()}
                    
                    num_classes = len(label_map)
                    
                    if model_type == 'lstm':
                        model = lm.BiLSTMAttentionModel(144, 256, num_classes, 2)
                    else:
                        model = tm.TransformerSignModel(144, 256, 8, 3, 512, num_classes)
                        
                    model.load_state_dict(torch.load(weights_path, map_location=self.device))
                    model.to(self.device)
                    model.eval()
                    
                    with torch.no_grad():
                        for item in self.test_data:
                            seq = item['sequence']
                            true_label = item['label']
                            
                            tensor_seq = torch.tensor(seq).unsqueeze(0).to(self.device)
                            tensor_len = torch.tensor([len(seq)]).to(self.device)
                            
                            outputs = model(tensor_seq, tensor_len)
                            _, pred_idx = torch.max(outputs, 1)
                            pred_label = label_map[pred_idx.item()]
                            
                            self.results[model_type]['y_true'].append(true_label)
                            self.results[model_type]['y_pred'].append(pred_label)
                except Exception as e:
                    print(f"{model_type.upper()} Eval Error: {e}")

class EvalUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Model Benchmarking & Evaluation Dashboard")
        self.root.geometry("1000x750")
        
        self.backend = EvaluatorBackend()
        
        # --- TOP PANEL ---
        top_frame = tk.Frame(root, bg="#f8f9fa", pady=10)
        top_frame.pack(fill="x")
        tk.Label(top_frame, text="Sistem Evaluasi Data Test", font=("Arial", 16, "bold"), bg="#f8f9fa").pack()
        
        # --- CHECKBOX PILIHAN MODEL ---
        check_frame = tk.Frame(top_frame, bg="#f8f9fa")
        check_frame.pack(pady=5)
        
        self.var_faiss = tk.BooleanVar(value=True)
        self.var_lstm = tk.BooleanVar(value=True)
        self.var_trans = tk.BooleanVar(value=True)
        
        tk.Checkbutton(check_frame, text="FAISS (Baseline)", variable=self.var_faiss, bg="#f8f9fa", font=("Arial", 10)).pack(side="left", padx=10)
        tk.Checkbutton(check_frame, text="Bi-LSTM", variable=self.var_lstm, bg="#f8f9fa", font=("Arial", 10)).pack(side="left", padx=10)
        tk.Checkbutton(check_frame, text="Transformer", variable=self.var_trans, bg="#f8f9fa", font=("Arial", 10)).pack(side="left", padx=10)
        
        self.btn_run = tk.Button(top_frame, text="Mulai Evaluasi Terpilih", bg="#0d6efd", fg="white", font=("Arial", 11, "bold"), command=self.run_evaluation)
        self.btn_run.pack(pady=10)
        
        # --- TABS ---
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(expand=True, fill="both", padx=10, pady=10)
        
        self.tab_eda = tk.Frame(self.notebook, bg="white")
        self.notebook.add(self.tab_eda, text="1. EDA (Data Test)")
        
        self.tab_compare = tk.Frame(self.notebook, bg="white")
        self.notebook.add(self.tab_compare, text="2. Ringkasan Perbandingan")
        
        self.tab_detail = tk.Frame(self.notebook, bg="white")
        self.notebook.add(self.tab_detail, text="3. Detail Per Model")
        
        self.setup_detail_tab()

    def run_evaluation(self):
        selected_models = []
        if self.var_faiss.get(): selected_models.append('faiss')
        if self.var_lstm.get(): selected_models.append('lstm')
        if self.var_trans.get(): selected_models.append('transformer')
        
        if not selected_models:
            messagebox.showwarning("Peringatan", "Pilih minimal 1 model untuk dievaluasi!")
            return

        self.btn_run.config(state="disabled", text="Memuat Data & Menjalankan Inferensi...")
        
        def task():
            status, msg = self.backend.load_test_data()
            if not status:
                self.root.after(0, lambda: messagebox.showerror("Error", msg))
                self.root.after(0, lambda: self.btn_run.config(state="normal", text="Mulai Evaluasi Terpilih"))
                return
                
            self.backend.run_evaluations(selected_models)
            
            self.root.after(0, self.render_eda)
            self.root.after(0, self.render_comparison)
            
            # Update combobox dropdown berdasarkan model yang benar-benar dieksekusi
            self.root.after(0, lambda: self.combo_model.config(values=selected_models))
            self.root.after(0, lambda: self.combo_model.set(selected_models[0]))
            self.root.after(0, self.update_detail_view)
            
            self.root.after(0, lambda: self.btn_run.config(state="normal", text="Evaluasi Selesai (Update Ulang)"))
            
        threading.Thread(target=task, daemon=True).start()

    def clear_frame(self, frame):
        for widget in frame.winfo_children():
            widget.destroy()

    def render_eda(self):
        self.clear_frame(self.tab_eda)
        labels = [item['label'] for item in self.backend.test_data]
        df_counts = pd.Series(labels).value_counts().reset_index()
        df_counts.columns = ['Vocab', 'Jumlah Sampel Test']
        
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(data=df_counts, x='Vocab', y='Jumlah Sampel Test', ax=ax, palette="viridis")
        ax.set_title("Distribusi Kelas pada Data Test", fontweight="bold")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        plt.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, master=self.tab_eda)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def render_comparison(self):
        self.clear_frame(self.tab_compare)
        
        metrics = {'Model': [], 'Macro Precision': [], 'Macro Recall': [], 'Macro F1-Score': []}
        
        for model_name, data in self.backend.results.items():
            if not data['y_true']: continue
            
            p, r, f1, _ = precision_recall_fscore_support(data['y_true'], data['y_pred'], average='macro', zero_division=0)
            metrics['Model'].append(model_name.upper())
            metrics['Macro Precision'].append(p)
            metrics['Macro Recall'].append(r)
            metrics['Macro F1-Score'].append(f1)
            
        if not metrics['Model']: return # Cegah crash jika semua error
            
        df_metrics = pd.DataFrame(metrics)
        df_melted = df_metrics.melt(id_vars="Model", var_name="Metric", value_name="Score")
        
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(data=df_melted, x='Metric', y='Score', hue='Model', ax=ax, palette="Set2")
        ax.set_title("Perbandingan Performa Arsitektur (Macro Average)", fontweight="bold")
        ax.set_ylim(0, 1.05)
        for p in ax.patches:
            if p.get_height() > 0:
                ax.annotate(format(p.get_height(), '.2f'), 
                            (p.get_x() + p.get_width() / 2., p.get_height()), 
                            ha = 'center', va = 'center', xytext = (0, 9), textcoords = 'offset points')
                        
        plt.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=self.tab_compare)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def setup_detail_tab(self):
        control_frame = tk.Frame(self.tab_detail, bg="white")
        control_frame.pack(fill="x", pady=5)
        
        tk.Label(control_frame, text="Pilih Model:", bg="white", font=("Arial", 11, "bold")).pack(side="left", padx=10)
        self.combo_model = ttk.Combobox(control_frame, values=["faiss", "lstm", "transformer"], state="readonly")
        self.combo_model.set("faiss")
        self.combo_model.pack(side="left")
        self.combo_model.bind("<<ComboboxSelected>>", lambda e: self.update_detail_view())
        
        self.vis_frame = tk.Frame(self.tab_detail, bg="white")
        self.vis_frame.pack(fill="both", expand=True)
        
        self.cm_frame = tk.Frame(self.vis_frame, bg="white")
        self.cm_frame.pack(side="left", fill="both", expand=True)
        
        self.table_frame = tk.Frame(self.vis_frame, bg="white", width=300)
        self.table_frame.pack(side="right", fill="y", padx=10)
        
    def update_detail_view(self):
        selected_model = self.combo_model.get()
        res = self.backend.results.get(selected_model, {})
        
        if not res.get('y_true'):
            self.clear_frame(self.cm_frame)
            self.clear_frame(self.table_frame)
            return
            
        y_true = res['y_true']
        y_pred = res['y_pred']
        classes = sorted(list(set(y_true + y_pred)))
        
        # 1. Update Confusion Matrix
        self.clear_frame(self.cm_frame)
        cm = confusion_matrix(y_true, y_pred, labels=classes)
        
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes, ax=ax)
        ax.set_title(f"Confusion Matrix - {selected_model.upper()}", fontweight="bold")
        ax.set_ylabel("True Label")
        ax.set_xlabel("Predicted Label")
        plt.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, master=self.cm_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        
        # 2. Update Classification Report Table
        self.clear_frame(self.table_frame)
        report = classification_report(y_true, y_pred, labels=classes, output_dict=True, zero_division=0)
        
        tk.Label(self.table_frame, text="Skor Per Kelas", font=("Arial", 11, "bold"), bg="white").pack(pady=5)
        
        columns = ("Class", "Precision", "Recall", "F1-Score")
        tree = ttk.Treeview(self.table_frame, columns=columns, show="headings", height=15)
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=70, anchor="center")
        tree.column("Class", width=100, anchor="w")
        
        for cls in classes:
            if cls in report:
                row = (cls, f"{report[cls]['precision']:.2f}", f"{report[cls]['recall']:.2f}", f"{report[cls]['f1-score']:.2f}")
                tree.insert("", tk.END, values=row)
                
        tree.pack(fill="both", expand=True)

if __name__ == "__main__":
    root = tk.Tk()
    app = EvalUI(root)
    root.mainloop()