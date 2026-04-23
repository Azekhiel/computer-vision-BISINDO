import pandas as pd
import numpy as np
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

import database_manager as dbm

# Konfigurasi Path
DATABASE_FILE = 'dataset_dynamic.csv'
MODEL_DIR = 'models'
LSTM_WEIGHTS = os.path.join(MODEL_DIR, 'lstm_weights.pth')
LABEL_ENCODER_FILE = os.path.join(MODEL_DIR, 'lstm_labels.json')

# Hyperparameters
INPUT_DIM = 144  # Koordinat spasial dari Mediapipe
HIDDEN_DIM = 256
NUM_LAYERS = 2
BATCH_SIZE = 32
LEARNING_RATE = 0.001
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
            # Susun matriks [Seq_Len, 144]
            seq = np.array([parse_features(f) for f in group['features']])
            
            self.sequences.append(torch.tensor(seq))
            self.labels.append(label_map[label])
            
    def __len__(self):
        return len(self.sequences)
        
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

def collate_fn(batch):
    """
    Fungsi khusus DataLoader untuk menyatukan sequence beda durasi ke dalam 1 batch.
    Sequence terpendek akan diisi angka 0 (padding) sampai menyamai sequence terpanjang di batch itu.
    """
    sequences, labels = zip(*batch)
    
    # Simpan panjang asli masing-masing sequence sebelum di-pad
    lengths = torch.tensor([len(seq) for seq in sequences])
    
    # Pad sequence
    padded_seqs = pad_sequence(sequences, batch_first=True, padding_value=0.0)
    labels = torch.tensor(labels, dtype=torch.long)
    
    return padded_seqs, labels, lengths

# ==========================================
# 3. ARSITEKTUR MODEL (Bi-LSTM + ATTENTION)
# ==========================================
class TemporalAttention(nn.Module):
    def __init__(self, hidden_size):
        super(TemporalAttention, self).__init__()
        self.attention = nn.Linear(hidden_size, 1)
        
    def forward(self, lstm_output, lengths):
        # lstm_output shape: [batch_size, seq_len, hidden_size]
        attn_weights = self.attention(lstm_output).squeeze(2) # [batch_size, seq_len]
        
        # Buat mask agar frame hasil padding (angka 0) tidak ikut dihitung attention-nya
        mask = torch.arange(lstm_output.size(1))[None, :] < lengths[:, None]
        mask = mask.to(lstm_output.device)
        
        attn_weights[~mask] = float('-inf') # Beri bobot minus tak hingga untuk padding
        attn_weights = torch.softmax(attn_weights, dim=1)
        
        # Kalikan bobot dengan output LSTM untuk mendapatkan vektor konteks
        context = torch.bmm(attn_weights.unsqueeze(1), lstm_output).squeeze(1)
        return context

class BiLSTMAttentionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers):
        super(BiLSTMAttentionModel, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True # Membaca ke depan dan ke belakang
        )
        
        # hidden_dim * 2 karena model bi-directional
        self.attention = TemporalAttention(hidden_dim * 2)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        
    def forward(self, x, lengths):
        # Gunakan pack_padded_sequence agar LSTM tidak memproses angka 0 (padding)
        packed_input = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        
        # Terapkan Attention Mechanism
        context = self.attention(output, lengths)
        
        # Klasifikasi akhir
        logits = self.fc(context)
        return logits

# ==========================================
# 4. TRAINING ENGINE
# ==========================================
def train_lstm_model():
    print(f"\n--- Memulai Build & Train Bi-LSTM ---")
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
    
    # Simpan label encoder agar saat Inference sistem tahu Node 0 itu vocab apa
    if not os.path.exists(MODEL_DIR): os.makedirs(MODEL_DIR)
    with open(LABEL_ENCODER_FILE, 'w') as f:
        json.dump({v: k for k, v in label_map.items()}, f) # Simpan dalam format {0: 'halo', 1: 'terima_kasih'}
        
    num_classes = len(unique_labels)
    
    # Persiapkan Dataset & DataLoader
    train_dataset = SignLanguageDataset(train_df, label_map)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    
    val_loader = None
    if not val_df.empty:
        val_dataset = SignLanguageDataset(val_df, label_map)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
        
    # Inisialisasi Model, Loss, dan Optimizer
    model = BiLSTMAttentionModel(INPUT_DIM, HIDDEN_DIM, num_classes, NUM_LAYERS).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
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
            optimizer.step()
            
            total_train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += batch_labels.size(0)
            correct_train += (predicted == batch_labels).sum().item()
            
        train_acc = 100 * correct_train / total_train
        
        # Validation Loop (jika ada data val)
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
            
            # Simpan model terbaik berdasarkan Validation Loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), LSTM_WEIGHTS)
        else:
            # Jika tidak ada data val, simpan terus di setiap akhir epoch
            torch.save(model.state_dict(), LSTM_WEIGHTS)
            
        if (epoch + 1) % 5 == 0 or epoch == 0:
            avg_train_loss = total_train_loss / len(train_loader)
            print(f"Epoch [{epoch+1}/{EPOCHS}] Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}%{val_msg}")

    # Lapor ke database metadata
    dbm.update_metadata("lstm")
    
    return True, f"Training Bi-LSTM Selesai (100%). Model tersimpan dengan {num_classes} klasifikasi vocab."

def get_lstm_status():
    """Mengembalikan status model LSTM saat ini."""
    status_dict = dbm.check_model_status()
    return status_dict.get("lstm", "Unknown")

if __name__ == "__main__":
    status, msg = train_lstm_model()
    print(msg)