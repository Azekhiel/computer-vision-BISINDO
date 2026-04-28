import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from tqdm import tqdm

# Import modul internal
import database_manager as dbm

# ==========================================
# KONFIGURASI DIREKTORI
# ==========================================
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATABASE_DIR = os.path.join(ROOT_DIR, 'dataset_parquets')
MODEL_DIR = os.path.join(ROOT_DIR, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

LSTM_WEIGHTS = os.path.join(MODEL_DIR, 'lstm_weights.pth')
LSTM_LABELS = os.path.join(MODEL_DIR, 'lstm_labels.json')

# ==========================================
# ARSITEKTUR BI-LSTM DENGAN ATTENTION
# ==========================================
class BiLSTMAttentionModel(nn.Module):
    def __init__(self, input_dim=144, hidden_dim=256, num_classes=10, num_layers=2):
        super(BiLSTMAttentionModel, self).__init__()
        self.hidden_dim = hidden_dim
        
        # Bi-LSTM Layer
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, 
                            batch_first=True, bidirectional=True)
        
        # Attention Mechanism Layer
        self.attention = nn.Linear(hidden_dim * 2, 1)
        
        # Fully Connected (Classifier) Layer
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x, lengths):
        # Packing padding sequence agar LSTM tidak menghitung frame kosong (0)
        # lengths dikonversi ke CPU karena PyTorch pack_padded_sequence mewajibkannya
        packed_x = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed_x)
        
        # Unpack kembali menjadi tensor utuh
        out, _ = pad_packed_sequence(packed_out, batch_first=True)
        
        # --- Proses Temporal Attention ---
        # Menghitung bobot penting dari masing-masing frame (timestep)
        attn_weights = torch.softmax(self.attention(out), dim=1) # Shape: (batch, seq_len, 1)
        
        # Mengalikan bobot dengan output LSTM untuk mendapatkan vektor intisari (Context Vector)
        context_vector = torch.sum(attn_weights * out, dim=1) # Shape: (batch, hidden_dim*2)
        
        # Klasifikasi ke jumlah kosakata
        output = self.fc(context_vector)
        return output

# ==========================================
# PENGELOLA DATASET (PyTorch)
# ==========================================
def parse_features(feature_str):
    return np.array(list(map(float, feature_str.split(','))))

class SignDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        seq = torch.tensor(self.sequences[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        # Kembalikan urutan, label, dan panjang ASLI dari rekaman (sebelum di-padding)
        return seq, label, len(seq)

def collate_fn(batch):
    """
    Fungsi khusus untuk menggabungkan batch video dengan durasi berbeda-beda.
    Video yang lebih pendek akan ditambahkan angka 0 di belakangnya (Padding).
    """
    seqs, labels, lengths = zip(*batch)
    seqs_padded = torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True)
    labels = torch.stack(labels)
    lengths = torch.tensor(lengths)
    return seqs_padded, labels, lengths

# ==========================================
# FUNGSI TRAINING UTAMA
# ==========================================
def train_lstm_model(epochs=35, batch_size=32, lr=0.001):
    print("\n--- Memulai Persiapan Data Bi-LSTM ---")
    vocabs = dbm.get_vocab_list()

    sequences = []
    labels = []
    label_map = {}
    current_label_id = 0

    print("Membaca dan menyaring data dari Parquet...")
    for vocab in vocabs:
        # ==========================================
        # KUNCI TWO-STAGE PIPELINE: SKIP KELAS IDLE
        # ==========================================
        if vocab == 'idle':
            print("  [INFO] Melewati kelas 'idle'. Bi-LSTM hanya belajar isyarat bermakna.")
            continue

        filepath = os.path.join(DATABASE_DIR, f"{vocab}.parquet")
        if not os.path.exists(filepath): continue
            
        print(f"  Mengekstrak {vocab}...")
        label_map[current_label_id] = vocab
        
        df = pd.read_parquet(filepath)
        
        for vid, group in df.groupby('video_id'):
            group = group.sort_values('frame_num')
            seq = np.array([parse_features(f) for f in group['features']])
            sequences.append(seq)
            labels.append(current_label_id)
            
        current_label_id += 1

    if len(sequences) == 0:
        return False, "Data isyarat valid tidak ditemukan (pastikan sudah ada data selain 'idle')."

    # Simpan map label (ID -> Nama Vocab) ke JSON untuk dibaca oleh inference_engine.py
    with open(LSTM_LABELS, 'w') as f:
        json.dump(label_map, f)

    num_classes = len(label_map)
    print(f"\nTotal Data Isyarat: {len(sequences)}")
    print(f"Total Kelas (Vocab): {num_classes}")

    # Persiapan DataLoader dengan Custom Collate
    dataset = SignDataset(sequences, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    # Inisialisasi Model ke GPU/CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BiLSTMAttentionModel(input_dim=144, hidden_dim=256, num_classes=num_classes, num_layers=2).to(device)

    # Kriteria dan Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print(f"\n[DEVICE] Training Bi-LSTM di: {device.type.upper()}")
    
    # Training Loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        train_bar = tqdm(dataloader, desc=f"Epoch [{epoch+1}/{epochs}]")
        for batch_seqs, batch_labels, batch_lengths in train_bar:
            batch_seqs = batch_seqs.to(device)
            batch_labels = batch_labels.to(device)
            batch_lengths = batch_lengths.to(device)
            
            optimizer.zero_grad()
            
            # Maju (Forward pass)
            outputs = model(batch_seqs, batch_lengths)
            loss = criterion(outputs, batch_labels)
            
            # Mundur (Backward pass) dan optimasi
            loss.backward()
            optimizer.step()
            
            # Hitung statistik untuk ditambahkan ke progress bar
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()
            
            acc = 100 * correct / total
            train_bar.set_postfix({'Loss': f"{loss.item():.4f}", 'Acc': f"{acc:.1f}%"})

    # Simpan bobot final
    torch.save(model.state_dict(), LSTM_WEIGHTS)
    return True, f"Pelatihan Bi-LSTM Selesai! Bobot disimpan untuk {num_classes} kelas isyarat."

if __name__ == "__main__":
    status, msg = train_lstm_model()
    print(msg)