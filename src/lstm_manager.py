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
import feature_engine as fe

# ==========================================
# KONFIGURASI DIREKTORI
# ==========================================
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATABASE_DIR = os.path.join(ROOT_DIR, 'dataset_parquets')
MODEL_DIR = os.path.join(ROOT_DIR, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

LSTM_WEIGHTS = os.path.join(MODEL_DIR, 'lstm_weights.pth')
LSTM_LABELS = os.path.join(MODEL_DIR, 'lstm_labels.json')
LSTM_METADATA = os.path.join(MODEL_DIR, 'lstm_metadata.json')

INPUT_DIM = 179 # Spasial(144) + Angles(32) + Flags(3)

# ==========================================
# ARSITEKTUR BI-LSTM DENGAN ATTENTION
# ==========================================
class BiLSTMAttentionModel(nn.Module):
    def __init__(self, input_dim=179, hidden_dim=256, num_classes=10, num_layers=2):
        super(BiLSTMAttentionModel, self).__init__()
        self.hidden_dim = hidden_dim
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, 
                            batch_first=True, bidirectional=True)
        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x, lengths):
        lengths = lengths.to(device=x.device, dtype=torch.long).clamp(min=1, max=x.size(1))
        packed_x = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed_x)
        out, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=x.size(1))
        
        max_len = out.size(1)
        pad_mask = torch.arange(max_len, device=out.device)[None, :] >= lengths[:, None]
        attn_logits = self.attention(out).squeeze(-1).masked_fill(pad_mask, -1e9)
        attn_weights = torch.softmax(attn_logits, dim=1).unsqueeze(-1)
        context_vector = torch.sum(attn_weights * out, dim=1) 
        
        return self.fc(context_vector)

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
        return seq, label, len(seq)

def collate_fn(batch):
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

    # PENAMBAHAN: Pisahkan list untuk Train dan Val
    train_sequences, train_labels = [], []
    val_sequences, val_labels = [], []
    
    label_map = {}
    current_label_id = 0

    print("Membaca dan menyaring data dari Parquet...")
    for vocab in vocabs:
        if vocab == 'idle':
            print("  [INFO] Melewati kelas 'idle'. Bi-LSTM hanya belajar isyarat bermakna.")
            continue

        filepath = os.path.join(DATABASE_DIR, f"{vocab}.parquet")
        if not os.path.exists(filepath): continue
            
        df = pd.read_parquet(filepath)
        df = fe.filter_current_feature_rows(df)
        if df.empty:
            print(f"  [SKIP] {vocab}: tidak ada data V3.1.")
            continue

        print(f"  Mengekstrak {vocab}...")
        label_map[current_label_id] = vocab
        
        # 1. Ekstrak data TRAIN
        train_df = df[df['split'] == 'train']
        for vid, group in train_df.groupby('video_id'):
            group = group.sort_values('frame_num')
            seq = np.array([parse_features(f) for f in group['features']])
            train_sequences.append(seq)
            train_labels.append(current_label_id)
            
        # 2. Ekstrak data VALIDATION
        val_df = df[df['split'] == 'val']
        for vid, group in val_df.groupby('video_id'):
            group = group.sort_values('frame_num')
            seq = np.array([parse_features(f) for f in group['features']])
            val_sequences.append(seq)
            val_labels.append(current_label_id)
            
        current_label_id += 1

    if len(train_sequences) == 0:
        return False, f"Data isyarat valid V3.1 (train) tidak ditemukan. Re-import dataset agar feature_version={fe.FEATURE_SCHEMA}."

    with open(LSTM_LABELS, 'w') as f:
        json.dump(label_map, f)

    num_classes = len(label_map)
    print(f"\nTotal Data - Train: {len(train_sequences)} | Val: {len(val_sequences)}")
    print(f"Total Kelas (Vocab): {num_classes}")

    # PENAMBAHAN: Buat 2 DataLoader
    train_dataset = SignDataset(train_sequences, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    
    val_loader = None
    if len(val_sequences) > 0:
        val_dataset = SignDataset(val_sequences, val_labels)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BiLSTMAttentionModel(input_dim=INPUT_DIM, hidden_dim=256, num_classes=num_classes, num_layers=2).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print(f"\n[DEVICE] Training Bi-LSTM di: {device.type.upper()}")
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # --- TRAINING LOOP ---
        model.train()
        total_train_loss = 0.0
        correct_train = 0
        total_train = 0
        
        train_bar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}]")
        for batch_seqs, batch_labels, batch_lengths in train_bar:
            batch_seqs = batch_seqs.to(device)
            batch_labels = batch_labels.to(device)
            batch_lengths = batch_lengths.to(device)
            
            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch_seqs, batch_lengths)
            loss = criterion(outputs, batch_labels)
            
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += batch_labels.size(0)
            correct_train += (predicted == batch_labels).sum().item()
            
            acc = 100 * correct_train / total_train
            train_bar.set_postfix({'Loss': f"{loss.item():.4f}", 'Acc': f"{acc:.1f}%"})
            
        # --- VALIDATION LOOP ---
        if val_loader is not None:
            model.eval()
            total_val_loss = 0.0
            correct_val = 0
            total_val = 0
            
            with torch.inference_mode():
                for batch_seqs, batch_labels, batch_lengths in val_loader:
                    batch_seqs = batch_seqs.to(device)
                    batch_labels = batch_labels.to(device)
                    batch_lengths = batch_lengths.to(device)
                    
                    outputs = model(batch_seqs, batch_lengths)
                    v_loss = criterion(outputs, batch_labels)
                    
                    total_val_loss += v_loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total_val += batch_labels.size(0)
                    correct_val += (predicted == batch_labels).sum().item()
                    
            avg_val_loss = total_val_loss / len(val_loader)
            val_acc = 100 * correct_val / total_val
            
            print(f"  -> [VAL] Loss: {avg_val_loss:.4f} | Acc: {val_acc:.2f}%")
            
            # Checkpoint: Simpan hanya jika val_loss membaik
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), LSTM_WEIGHTS)
        else:
            # Jika tidak ada data val, simpan di setiap epoch akhir
            torch.save(model.state_dict(), LSTM_WEIGHTS)

    with open(LSTM_METADATA, 'w') as f:
        json.dump({"feature_schema": fe.FEATURE_SCHEMA, "input_dim": INPUT_DIM, "num_classes": num_classes}, f, indent=4)

    dbm.update_metadata("lstm")
    return True, f"Pelatihan Bi-LSTM Selesai! Bobot terbaik disimpan untuk {num_classes} kelas isyarat."

if __name__ == "__main__":
    status, msg = train_lstm_model()
    print(msg)
