import pandas as pd
import numpy as np
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from tqdm import tqdm  # Library untuk progress bar yang sempurna

import database_manager as dbm

# ==========================================
# KONFIGURASI PATH (Tahan Banting & Partisi)
# ==========================================
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATABASE_DIR = os.path.join(ROOT_DIR, 'dataset_parquets')
MODEL_DIR = os.path.join(ROOT_DIR, 'models')

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
    """Mengubah string fitur menjadi array float32."""
    return np.array(list(map(float, feature_str.split(','))), dtype=np.float32)

class SignLanguageDataset(Dataset):
    def __init__(self, df, label_map):
        self.sequences = []
        self.labels = []
        
        if df.empty:
            return

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
    Menangani batching untuk sequence dengan durasi berbeda menggunakan padding.
    """
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(seq) for seq in sequences])
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
        attn_weights = self.attention(lstm_output).squeeze(2)
        mask = torch.arange(lstm_output.size(1))[None, :] < lengths[:, None]
        mask = mask.to(lstm_output.device)
        attn_weights[~mask] = float('-inf')
        attn_weights = torch.softmax(attn_weights, dim=1)
        context = torch.bmm(attn_weights.unsqueeze(1), lstm_output).squeeze(1)
        return context

class BiLSTMAttentionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers):
        super(BiLSTMAttentionModel, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, 
                            num_layers=num_layers, batch_first=True, bidirectional=True)
        self.attention = TemporalAttention(hidden_dim * 2)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        
    def forward(self, x, lengths):
        packed_input = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        context = self.attention(output, lengths)
        return self.fc(context)

# ==========================================
# 4. TRAINING ENGINE
# ==========================================
def train_lstm_model():
    print(f"\n--- Memulai Build & Train Bi-LSTM ---")
    print(f"[DEVICE] PyTorch menggunakan: {device.type.upper()}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        
    if not os.path.exists(DATABASE_DIR):
        return False, "Folder database tidak ditemukan."

    # --- A. MUAT DATA DARI PARTISI PARQUET ---
    print("Memuat data dari partisi folder...")
    train_dfs = []
    val_dfs = []
    all_vocabs = []

    for file in os.listdir(DATABASE_DIR):
        if file.endswith('.parquet'):
            filepath = os.path.join(DATABASE_DIR, file)
            df = pd.read_parquet(filepath)
            train_dfs.append(df[df['split'] == 'train'])
            val_dfs.append(df[df['split'] == 'val'])
            all_vocabs.append(file.replace('.parquet', ''))

    if not train_dfs:
        return False, "Tidak ada data training sama sekali."

    full_train_df = pd.concat(train_dfs, ignore_index=True)
    full_val_df = pd.concat(val_dfs, ignore_index=True) if val_dfs else pd.DataFrame()

    unique_labels = sorted(all_vocabs)
    label_map = {label: i for i, label in enumerate(unique_labels)}
    
    # Simpan label encoder untuk keperluan inferensi nanti
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(LABEL_ENCODER_FILE, 'w') as f:
        json.dump({i: label for label, i in label_map.items()}, f)
        
    num_classes = len(unique_labels)
    
    # Persiapkan Dataset & DataLoader
    print("Mempersiapkan Dataset & DataLoader...")
    train_dataset = SignLanguageDataset(full_train_df, label_map)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    
    val_loader = None
    if not full_val_df.empty:
        val_dataset = SignLanguageDataset(full_val_df, label_map)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
        
    # Inisialisasi Model, Loss, dan Optimizer
    model = BiLSTMAttentionModel(INPUT_DIM, HIDDEN_DIM, num_classes, NUM_LAYERS).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    best_val_loss = float('inf')
    
    print(f"Memulai Training sebanyak {EPOCHS} Epoch...")
    
    for epoch in range(EPOCHS):
        model.train()
        total_train_loss, correct_train, total_train = 0, 0, 0
        
        # Integrasi TQDM untuk progress bar per Batch
        train_bar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{EPOCHS}]", unit="batch")
        
        for batch_seqs, batch_labels, batch_lengths in train_bar:
            batch_seqs, batch_labels, batch_lengths = batch_seqs.to(device), batch_labels.to(device), batch_lengths.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_seqs, batch_lengths)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += batch_labels.size(0)
            correct_train += (predicted == batch_labels).sum().item()
            
            # Update informasi pada progress bar secara real-time
            train_bar.set_postfix(loss=loss.item(), acc=f"{100 * correct_train / total_train:.2f}%")
            
        train_acc = 100 * correct_train / total_train
        avg_train_loss = total_train_loss / len(train_loader)
        
        val_msg = ""
        if val_loader:
            model.eval()
            total_val_loss, correct_val, total_val = 0, 0, 0
            with torch.no_grad():
                for batch_seqs, batch_labels, batch_lengths in val_loader:
                    batch_seqs, batch_labels, batch_lengths = batch_seqs.to(device), batch_labels.to(device), batch_lengths.to(device)
                    outputs = model(batch_seqs, batch_lengths)
                    loss = criterion(outputs, batch_labels)
                    total_val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total_val += batch_labels.size(0)
                    correct_val += (predicted == batch_labels).sum().item()
            
            val_loss = total_val_loss / len(val_loader)
            val_acc = 100 * correct_val / total_val
            val_msg = f" | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%"
            
            # Simpan model terbaik berdasarkan Validation Loss (mencegah overfitting)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), LSTM_WEIGHTS)
        else:
            # Jika tidak ada data validasi, simpan model setiap akhir epoch
            torch.save(model.state_dict(), LSTM_WEIGHTS)
            
        # Log ringkasan per epoch
        print(f" -> Summary Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}%{val_msg}")

    # Perbarui metadata untuk menginformasikan bahwa model sudah up-to-date
    dbm.update_metadata("lstm")
    
    return True, f"Training Bi-LSTM Selesai. Model klasifikasi {num_classes} vocab tersimpan di {LSTM_WEIGHTS}."

def get_lstm_status():
    """Mengecek status model melalui database manager."""
    status_dict = dbm.check_model_status()
    return status_dict.get("lstm", "Unknown")

if __name__ == "__main__":
    status, msg = train_lstm_model()
    print(msg)