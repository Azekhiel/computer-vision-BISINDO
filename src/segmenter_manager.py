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
    def __init__(self, input_dim=179, hidden_dim=64):
        super(VADSegmenterModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
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
    
    if 'idle' not in vocabs:
        return False, "ERROR: Kelas 'idle' belum ada! Buat kosakata bernama 'idle' di UI dan rekam data diam/noise terlebih dahulu."
        
    train_sequences, train_labels = [], []
    val_sequences, val_labels = [], []
    
    print("Membaca dan melabeli data dari Parquet...")
    for vocab in vocabs:
        filepath = os.path.join(DATABASE_DIR, f"{vocab}.parquet")
        if not os.path.exists(filepath): continue
            
        df = pd.read_parquet(filepath)
        current_label = 0 if vocab == 'idle' else 1
        
        # 1. Ekstrak Split Train
        train_df = df[df['split'] == 'train']
        for vid, group in train_df.groupby('video_id'):
            group = group.sort_values('frame_num')
            seq = np.array([parse_features(f) for f in group['features']])
            std_seq = fm.interpolate_sequence(seq, 30)
            train_sequences.append(std_seq)
            train_labels.append(current_label)
            
        # 2. Ekstrak Split Val
        val_df = df[df['split'] == 'val']
        for vid, group in val_df.groupby('video_id'):
            group = group.sort_values('frame_num')
            seq = np.array([parse_features(f) for f in group['features']])
            std_seq = fm.interpolate_sequence(seq, 30)
            val_sequences.append(std_seq)
            val_labels.append(current_label)
            
    if len(train_sequences) == 0:
        return False, "Data training tidak ditemukan."
        
    num_idle = train_labels.count(0)
    num_sign = train_labels.count(1)
    print(f"Data Train: {len(train_sequences)} (Idle: {num_idle}, Isyarat: {num_sign})")
    print(f"Data Val: {len(val_sequences)}")
    
    if num_idle == 0:
        return False, "Data 'idle' untuk training kosong! Harus ada minimal 1 sampel idle di partisi 'train'."

    # Hitung bobot penyeimbang untuk kelas minoritas
    weight_ratio = num_idle / num_sign if num_sign > 0 else 1.0
    pos_weight = torch.tensor([weight_ratio])
    
    # DataLoader
    train_dataset = SegmenterDataset(train_sequences, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_loader = None
    if len(val_sequences) > 0:
        val_dataset = SegmenterDataset(val_sequences, val_labels)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VADSegmenterModel(input_dim=179, hidden_dim=64).to(device)
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print(f"\n[DEVICE] Training Segmenter di: {device.type.upper()}")
    
    best_val_loss = float('inf')
    
    # Training Loop
    for epoch in range(epochs):
        # --- TRAINING PHASE ---
        model.train()
        running_loss = 0.0
        correct_train, total_train = 0, 0
        
        train_bar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}]")
        for batch_seqs, batch_labels in train_bar:
            batch_seqs = batch_seqs.to(device)
            batch_labels = batch_labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_seqs)
            
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # Hitung Akurasi Binary
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            total_train += batch_labels.size(0)
            correct_train += (predicted == batch_labels).sum().item()
            
            train_acc = 100 * correct_train / total_train
            train_bar.set_postfix({'Loss': f"{loss.item():.4f}", 'Acc': f"{train_acc:.1f}%"})
            
        # --- VALIDATION PHASE ---
        if val_loader is not None:
            model.eval()
            val_loss_total = 0.0
            correct_val, total_val = 0, 0
            
            with torch.no_grad():
                for batch_seqs, batch_labels in val_loader:
                    batch_seqs = batch_seqs.to(device)
                    batch_labels = batch_labels.to(device)
                    
                    outputs = model(batch_seqs)
                    v_loss = criterion(outputs, batch_labels)
                    val_loss_total += v_loss.item()
                    
                    predicted = (torch.sigmoid(outputs) > 0.5).float()
                    total_val += batch_labels.size(0)
                    correct_val += (predicted == batch_labels).sum().item()
                    
            avg_val_loss = val_loss_total / len(val_loader)
            val_acc = 100 * correct_val / total_val
            print(f"  -> [VAL] Loss: {avg_val_loss:.4f} | Acc: {val_acc:.2f}%")
            
            # Checkpoint
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), SEGMENTER_WEIGHTS)
        else:
            torch.save(model.state_dict(), SEGMENTER_WEIGHTS)
            
    return True, "Pelatihan Segmenter Berhasil! Bobot terbaik telah disimpan."

if __name__ == "__main__":
    status, msg = train_segmenter()
    print(msg)