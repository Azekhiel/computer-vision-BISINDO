import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from tqdm import tqdm

# Import modul internal
import database_manager as dbm
import faiss_manager as fm

torch.backends.cudnn.enabled = False

# ==========================================
# KONFIGURASI DIREKTORI
# ==========================================
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATABASE_DIR = os.path.join(ROOT_DIR, 'dataset_parquets')
MODEL_DIR = os.path.join(ROOT_DIR, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

SEGMENTER_WEIGHTS = os.path.join(MODEL_DIR, 'segmenter_weights.pth')

# ==========================================
# ARSITEKTUR MODEL (SMART VAD)
# ==========================================
class VADSegmenterModel(nn.Module):
    # KUNCI PERBAIKAN: Ubah input_dim menjadi 179
    def __init__(self, input_dim=179, hidden_dim=64):
        super(VADSegmenterModel, self).__init__()
        # LSTM Ringan (1 layer, dimensi kecil) agar inference super cepat di background
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        # Output 1 neuron untuk Binary Classification (0 = Idle/Noise, 1 = Isyarat Valid)
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        # x shape: (batch_size, seq_length, input_dim)
        out, _ = self.lstm(x)
        # Ambil representasi matematis dari timestep terakhir saja
        out = out[:, -1, :]
        return self.fc(out)

# ==========================================
# DATASET & DATALOADER
# ==========================================
class SegmenterDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        seq = torch.tensor(self.sequences[idx], dtype=torch.float32)
        label = torch.tensor([self.labels[idx]], dtype=torch.float32)
        return seq, label

def parse_features(feature_str):
    return np.array(list(map(float, feature_str.split(','))))

# ==========================================
# FUNGSI TRAINING UTAMA
# ==========================================
def train_segmenter(epochs=15, batch_size=32):
    print("\n--- Memulai Persiapan Data Segmenter (Smart VAD) ---")
    vocabs = dbm.get_vocab_list()
    
    # Validasi Paling Penting: Pastikan kelas 'idle' sudah dibuat
    if 'idle' not in vocabs:
        return False, "ERROR: Kelas 'idle' belum ada! Buat kosakata bernama 'idle' di UI dan rekam data diam/noise terlebih dahulu."
        
    sequences = []
    labels = []
    
    print("Membaca dan melabeli data dari Parquet...")
    for vocab in vocabs:
        filepath = os.path.join(DATABASE_DIR, f"{vocab}.parquet")
        if not os.path.exists(filepath): continue
            
        df = pd.read_parquet(filepath)
        
        # Penentuan Label: 0 untuk idle (sampah/diam), 1 untuk isyarat valid (kelas apapun)
        current_label = 0 if vocab == 'idle' else 1
        
        for vid, group in df.groupby('video_id'):
            group = group.sort_values('frame_num')
            seq = np.array([parse_features(f) for f in group['features']])
            
            # Standarisasi ke 30 frame agar model tidak kaget saat training
            std_seq = fm.interpolate_sequence(seq, 30)
            
            sequences.append(std_seq)
            labels.append(current_label)
            
    if len(sequences) == 0:
        return False, "Data tidak ditemukan."
        
    # Kalkulasi rasio untuk menyeimbangkan bobot Loss
    num_idle = labels.count(0)
    num_sign = labels.count(1)
    print(f"Total Data: {len(sequences)} (Idle: {num_idle}, Isyarat: {num_sign})")
    
    if num_idle == 0:
        return False, "Data 'idle' kosong! Harus ada minimal 1 sampel idle asli dan augmentasinya."

    # Hitung bobot penyeimbang (Class Weights)
    weight_ratio = num_idle / num_sign if num_sign > 0 else 1.0
    pos_weight = torch.tensor([weight_ratio])
    
    # DataLoader
    dataset = SegmenterDataset(sequences, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Inisialisasi Model & Loss Function
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VADSegmenterModel(input_dim=179, hidden_dim=64).to(device)
    
    # BCEWithLogitsLoss sangat stabil untuk klasifikasi Binary karena sudah include Sigmoid
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print(f"\n[DEVICE] Training Segmenter di: {device.type.upper()}")
    
    # Training Loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        train_bar = tqdm(dataloader, desc=f"Epoch [{epoch+1}/{epochs}]")
        for batch_seqs, batch_labels in train_bar:
            batch_seqs = batch_seqs.to(device)
            batch_labels = batch_labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_seqs)
            
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            train_bar.set_postfix({'Loss': f"{loss.item():.4f}"})
            
    # Simpan bobot ke models/segmenter_weights.pth
    torch.save(model.state_dict(), SEGMENTER_WEIGHTS)
    return True, "Pelatihan Segmenter Berhasil! Bobot telah disimpan."

if __name__ == "__main__":
    status, msg = train_segmenter()
    print(msg)