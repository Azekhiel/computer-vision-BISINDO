import pandas as pd
import numpy as np
import os
import json
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import database_manager as dbm

# Konfigurasi Path
DATABASE_FILE = 'dataset_dynamic.csv'
MODEL_DIR = 'models'
TRANSFORMER_WEIGHTS = os.path.join(MODEL_DIR, 'transformer_weights.pth')
LABEL_ENCODER_FILE = os.path.join(MODEL_DIR, 'transformer_labels.json')

# Hyperparameters
INPUT_DIM = 144       # Koordinat spasial dari Mediapipe
D_MODEL = 256         # Dimensi representasi internal (harus bisa dibagi NHEAD)
NHEAD = 8             # Jumlah kepala Attention (Multi-Head Attention)
NUM_LAYERS = 3        # Jumlah tumpukan Encoder Transformer
DIM_FEEDFORWARD = 512 # Ukuran hidden layer di dalam feedforward Transformer
BATCH_SIZE = 32
LEARNING_RATE = 0.0005 # Biasanya Transformer butuh LR yang lebih kecil dari LSTM
EPOCHS = 50

# ==========================================
# 1. HARDWARE DETECTOR (CUDA/CPU)
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 2. DATASET & DATALOADER DENGAN PADDING
# ==========================================
def parse_features(feature_str):
    return np.array(list(map(float, feature_str.split(','))), dtype=np.float32)

class SignLanguageDataset(Dataset):
    def __init__(self, df, label_map):
        self.sequences = []
        self.labels = []
        
        grouped = df.groupby(['label', 'video_id'])
        for (label, video_id), group in grouped:
            group = group.sort_values('frame_num')
            seq = np.array([parse_features(f) for f in group['features']])
            
            self.sequences.append(torch.tensor(seq))
            self.labels.append(label_map[label])
            
    def __len__(self):
        return len(self.sequences)
        
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

def collate_fn(batch):
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(seq) for seq in sequences])
    
    # Pad sequence dengan 0.0
    padded_seqs = pad_sequence(sequences, batch_first=True, padding_value=0.0)
    labels = torch.tensor(labels, dtype=torch.long)
    
    return padded_seqs, labels, lengths

# ==========================================
# 3. ARSITEKTUR MODEL (TRANSFORMER SPATIO-TEMPORAL)
# ==========================================
class PositionalEncoding(nn.Module):
    """Menyuntikkan informasi urutan waktu ke dalam vektor karena Transformer tidak punya memori sekuensial."""
    def __init__(self, d_model, max_len=500):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0) # Bentuk: [1, max_len, d_model]

    def forward(self, x):
        # Menambahkan positional encoding ke input berdasarkan panjang sequence-nya
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :].to(x.device)

class TransformerSignModel(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, dim_feedforward, num_classes):
        super(TransformerSignModel, self).__init__()
        
        # Linear layer untuk memproyeksikan fitur 144-D menjadi d_model (256-D)
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer Encoder Layer
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=0.3, 
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # Layer klasifikasi akhir
        self.fc = nn.Linear(d_model, num_classes)
        
    def forward(self, src, lengths):
        # 1. Proyeksi fitur spasial & tambah informasi waktu (posisi)
        src = self.input_projection(src) # [batch_size, seq_len, d_model]
        src = self.pos_encoder(src)
        
        # 2. Buat Key Padding Mask. 
        # PyTorch Transformer butuh mask berisi 'True' untuk posisi padding (yang harus diabaikan)
        batch_size, max_seq_len, _ = src.size()
        # Buat matriks arange dan bandingkan dengan lengths untuk mendapat boolean mask
        mask = torch.arange(max_seq_len)[None, :] >= lengths[:, None]
        mask = mask.to(src.device) # [batch_size, seq_len]
        
        # 3. Masuk ke Transformer Encoder
        output = self.transformer_encoder(src, src_key_padding_mask=mask)
        
        # 4. Global Average Pooling (Mengambil intisari dari semua frame yang bukan padding)
        # Kita set output pada posisi padding menjadi 0
        output[mask] = 0.0
        # Jumlahkan nilai fiturnya, lalu bagi dengan panjang asli (bukan panjang hasil padding)
        summed = output.sum(dim=1)
        averaged = summed / lengths.unsqueeze(1).to(src.device).float()
        
        # 5. Klasifikasi Final
        logits = self.fc(averaged)
        return logits

# ==========================================
# 4. TRAINING ENGINE
# ==========================================
def train_transformer_model():
    print(f"\n--- Memulai Build & Train Transformer SOTA ---")
    print(f"[AKSELERASI] PyTorch menggunakan device: {device.type.upper()}")
    if device.type == 'cuda':
        print(f"GPU Terdeteksi: {torch.cuda.get_device_name(0)}")
        
    if not os.path.exists(DATABASE_FILE):
        return False, "Database belum ada."
        
    df = pd.read_csv(DATABASE_FILE)
    if df.empty: return False, "Database kosong."
    
    # Ambil data Train dan Val
    train_df = df[df['split'] == 'train']
    val_df = df[df['split'] == 'val']
    
    if train_df.empty:
        return False, "Tidak ada data 'train' untuk melatih model."
        
    # Buat pemetaan Label ke Integer (0, 1, 2, ...)
    unique_labels = sorted(df['label'].unique().tolist())
    label_map = {label: i for i, label in enumerate(unique_labels)}
    
    if not os.path.exists(MODEL_DIR): os.makedirs(MODEL_DIR)
    with open(LABEL_ENCODER_FILE, 'w') as f:
        json.dump({v: k for k, v in label_map.items()}, f)
        
    num_classes = len(unique_labels)
    
    # Persiapkan Dataset & DataLoader
    train_dataset = SignLanguageDataset(train_df, label_map)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    
    val_loader = None
    if not val_df.empty:
        val_dataset = SignLanguageDataset(val_df, label_map)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
        
    # Inisialisasi Model, Loss, dan Optimizer
    model = TransformerSignModel(INPUT_DIM, D_MODEL, NHEAD, NUM_LAYERS, DIM_FEEDFORWARD, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    # Optimizer AdamW biasanya lebih stabil untuk melatih Transformer
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    best_val_loss = float('inf')
    
    # Training Loop
    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0
        correct_train = 0
        total_train = 0
        
        for batch_seqs, batch_labels, batch_lengths in train_loader:
            batch_seqs = batch_seqs.to(device)
            batch_labels = batch_labels.to(device)
            batch_lengths = batch_lengths.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_seqs, batch_lengths)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            
            # Gradient Clipping: Penting untuk mencegah parameter Transformer "meledak"
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += batch_labels.size(0)
            correct_train += (predicted == batch_labels).sum().item()
            
        train_acc = 100 * correct_train / total_train
        
        # Validation Loop
        val_msg = ""
        if val_loader:
            model.eval()
            total_val_loss = 0
            correct_val = 0
            total_val = 0
            with torch.no_grad():
                for batch_seqs, batch_labels, batch_lengths in val_loader:
                    batch_seqs = batch_seqs.to(device)
                    batch_labels = batch_labels.to(device)
                    batch_lengths = batch_lengths.to(device)
                    
                    outputs = model(batch_seqs, batch_lengths)
                    loss = criterion(outputs, batch_labels)
                    
                    total_val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total_val += batch_labels.size(0)
                    correct_val += (predicted == batch_labels).sum().item()
                    
            val_loss = total_val_loss / len(val_loader)
            val_acc = 100 * correct_val / total_val
            val_msg = f" | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%"
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), TRANSFORMER_WEIGHTS)
        else:
            torch.save(model.state_dict(), TRANSFORMER_WEIGHTS)
            
        if (epoch + 1) % 5 == 0 or epoch == 0:
            avg_train_loss = total_train_loss / len(train_loader)
            print(f"Epoch [{epoch+1}/{EPOCHS}] Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}%{val_msg}")

    # Lapor ke database metadata
    dbm.update_metadata("transformer")
    
    return True, f"Training Transformer Selesai (100%). Model terbaik tersimpan."

def get_transformer_status():
    """Mengembalikan status model Transformer saat ini."""
    status_dict = dbm.check_model_status()
    return status_dict.get("transformer", "Unknown")

if __name__ == "__main__":
    status, msg = train_transformer_model()
    print(msg)