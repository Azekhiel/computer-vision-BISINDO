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
from tqdm import tqdm

import database_manager as dbm

# Konfigurasi Path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATABASE_DIR = os.path.join(ROOT_DIR, 'dataset_parquets')
MODEL_DIR = os.path.join(ROOT_DIR, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

TRANSFORMER_WEIGHTS = os.path.join(MODEL_DIR, 'transformer_weights.pth')
LABEL_ENCODER_FILE = os.path.join(MODEL_DIR, 'transformer_labels.json')

# Hyperparameters
INPUT_DIM = 179       # Spasial + Angles + Flags
D_MODEL = 256         
NHEAD = 8             
NUM_LAYERS = 3        
DIM_FEEDFORWARD = 512 
BATCH_SIZE = 32
LEARNING_RATE = 0.0005 
EPOCHS = 15

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 2. DATASET & DATALOADER DENGAN PADDING
# ==========================================
def parse_features(feature_str):
    return np.array(list(map(float, feature_str.split(','))), dtype=np.float32)

class SignLanguageDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels
            
    def __len__(self):
        return len(self.sequences)
        
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

def collate_fn(batch):
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(seq) for seq in sequences])
    padded_seqs = pad_sequence(sequences, batch_first=True, padding_value=0.0)
    labels = torch.tensor(labels, dtype=torch.long)
    return padded_seqs, labels, lengths

# ==========================================
# 3. ARSITEKTUR MODEL
# ==========================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]

class TransformerSignModel(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, dim_feedforward, num_classes):
        super(TransformerSignModel, self).__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=0.3, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.fc = nn.Linear(d_model, num_classes)
        
    def forward(self, src, lengths):
        lengths = lengths.to(device=src.device, dtype=torch.long).clamp(min=1, max=src.size(1))
        src = self.input_projection(src)
        src = self.pos_encoder(src)
        
        batch_size, max_seq_len, _ = src.size()
        
        mask = torch.arange(max_seq_len, device=src.device)[None, :] >= lengths[:, None]
        
        output = self.transformer_encoder(src, src_key_padding_mask=mask)
        
        valid = (~mask).unsqueeze(-1).to(output.dtype)
        summed = (output * valid).sum(dim=1)
        averaged = summed / lengths.unsqueeze(1).to(output.dtype)
        
        logits = self.fc(averaged)
        return logits

# ==========================================
# 4. TRAINING ENGINE
# ==========================================
def train_transformer_model():
    print(f"\n--- Memulai Build & Train Transformer SOTA ---")
    print(f"[AKSELERASI] PyTorch menggunakan device: {device.type.upper()}")
    
    vocabs = dbm.get_vocab_list()
    
    # PENAMBAHAN: Pemisahan Data Train dan Val
    train_sequences, train_labels = [], []
    val_sequences, val_labels = [], []
    
    label_map = {}
    current_label_id = 0

    print("Membaca dan menyaring data dari Parquet...")
    for vocab in vocabs:
        if vocab == 'idle': continue

        filepath = os.path.join(DATABASE_DIR, f"{vocab}.parquet")
        if not os.path.exists(filepath): continue
            
        label_map[current_label_id] = vocab
        df = pd.read_parquet(filepath)
        
        # 1. TRAIN SPLIT
        train_df = df[df['split'] == 'train']
        for vid, group in train_df.groupby('video_id'):
            group = group.sort_values('frame_num')
            seq = np.array([parse_features(f) for f in group['features']], dtype=np.float32)
            train_sequences.append(torch.tensor(seq))
            train_labels.append(current_label_id)
            
        # 2. VAL SPLIT
        val_df = df[df['split'] == 'val']
        for vid, group in val_df.groupby('video_id'):
            group = group.sort_values('frame_num')
            seq = np.array([parse_features(f) for f in group['features']], dtype=np.float32)
            val_sequences.append(torch.tensor(seq))
            val_labels.append(current_label_id)
            
        current_label_id += 1

    if len(train_sequences) == 0:
        return False, "Data isyarat valid (train) tidak ditemukan."

    with open(LABEL_ENCODER_FILE, 'w') as f:
        json.dump(label_map, f)
        
    num_classes = len(label_map)
    print(f"\nTotal Data - Train: {len(train_sequences)} | Val: {len(val_sequences)}")
    
    # PENAMBAHAN: Loader untuk Train dan Val
    train_dataset = SignLanguageDataset(train_sequences, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    
    val_loader = None
    if len(val_sequences) > 0:
        val_dataset = SignLanguageDataset(val_sequences, val_labels)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
        
    model = TransformerSignModel(INPUT_DIM, D_MODEL, NHEAD, NUM_LAYERS, DIM_FEEDFORWARD, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    best_val_loss = float('inf')
    
    for epoch in range(EPOCHS):
        # --- TRAINING LOOP ---
        model.train()
        total_train_loss = 0.0
        correct_train = 0
        total_train = 0
        
        train_bar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{EPOCHS}]")
        for batch_seqs, batch_labels, batch_lengths in train_bar:
            batch_seqs = batch_seqs.to(device)
            batch_labels = batch_labels.to(device)
            batch_lengths = batch_lengths.to(device)
            
            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch_seqs, batch_lengths)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), TRANSFORMER_WEIGHTS)
        else:
            torch.save(model.state_dict(), TRANSFORMER_WEIGHTS)

    dbm.update_metadata("transformer")
    return True, f"Training Transformer Selesai (100%). Model terbaik tersimpan."

if __name__ == "__main__":
    status, msg = train_transformer_model()
    print(msg)
