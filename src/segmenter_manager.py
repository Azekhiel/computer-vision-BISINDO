import os
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import database_manager as dbm
import faiss_manager as fm
import feature_engine as fe

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATABASE_DIR = os.path.join(ROOT_DIR, "dataset_parquets")
MODEL_DIR = os.path.join(ROOT_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

SEGMENTER_WEIGHTS = os.path.join(MODEL_DIR, "segmenter_weights.pth")
SEGMENTER_METADATA = os.path.join(MODEL_DIR, "segmenter_metadata.json")
VAD_TARGET_FRAMES = 30
INPUT_DIM = 179


class VADSegmenterModel(nn.Module):
    """
    Binary sign/idle detector.

    The old implementation used out[:, -1, :] from a BiLSTM. For a
    bidirectional model that is a weak summary because the backward stream at
    the last output corresponds mostly to the sequence start. This version uses
    the final hidden states from both directions, with optional packing for
    variable lengths.
    """

    def __init__(self, input_dim=INPUT_DIM, hidden_dim=64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x, lengths=None):
        if lengths is not None:
            lengths = lengths.to(device=x.device, dtype=torch.long).clamp(min=1, max=x.size(1))
            packed_x = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            _, (h_n, _) = self.lstm(packed_x)
        else:
            _, (h_n, _) = self.lstm(x)

        h_n = h_n.view(1, 2, x.size(0), self.hidden_dim)[-1]
        context = torch.cat([h_n[0], h_n[1]], dim=1)
        return self.fc(context)


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
    return np.array(list(map(float, feature_str.split(","))), dtype=np.float32)


def _standardize_vad_sequence(seq: np.ndarray) -> np.ndarray:
    if seq.ndim != 2 or seq.shape[1] != INPUT_DIM:
        raise ValueError(f"Expected VAD sequence shape (T, {INPUT_DIM}), got {seq.shape}")
    return fm.interpolate_sequence(seq, VAD_TARGET_FRAMES).astype(np.float32)


def _load_split_sequences(vocab: str, split: str, label: int):
    filepath = os.path.join(DATABASE_DIR, f"{vocab}.parquet")
    if not os.path.exists(filepath):
        return [], []

    df = pd.read_parquet(filepath)
    df = fe.filter_current_feature_rows(df)
    if df.empty:
        return [], []
    split_df = df[df["split"] == split]
    sequences, labels = [], []
    for _, group in split_df.groupby("video_id"):
        group = group.sort_values("frame_num")
        seq = np.array([parse_features(f) for f in group["features"]], dtype=np.float32)
        if len(seq) == 0:
            continue
        sequences.append(_standardize_vad_sequence(seq))
        labels.append(label)
    return sequences, labels


def train_segmenter(epochs=15, batch_size=32):
    print("\n--- Memulai Persiapan Data Segmenter (Smart VAD) ---")
    vocabs = dbm.get_vocab_list()

    if "idle" not in vocabs:
        return False, "ERROR: Kelas 'idle' belum ada. Buat vocab 'idle' dan rekam data diam/noise terlebih dahulu."

    train_sequences, train_labels = [], []
    val_sequences, val_labels = [], []

    print("Membaca data Parquet dan membentuk window VAD 30-frame...")
    for vocab in vocabs:
        current_label = 0 if vocab == "idle" else 1

        seqs, labels = _load_split_sequences(vocab, "train", current_label)
        train_sequences.extend(seqs)
        train_labels.extend(labels)

        seqs, labels = _load_split_sequences(vocab, "val", current_label)
        val_sequences.extend(seqs)
        val_labels.extend(labels)

    if len(train_sequences) == 0:
        return False, f"Data training {fe.FEATURE_SCHEMA} tidak ditemukan. Re-import dataset."

    num_idle = train_labels.count(0)
    num_sign = train_labels.count(1)
    print(f"Data Train: {len(train_sequences)} (Idle: {num_idle}, Isyarat: {num_sign})")
    print(f"Data Val: {len(val_sequences)}")

    if num_idle == 0:
        return False, "Data 'idle' untuk training kosong. Harus ada minimal 1 sampel idle di split train."
    if num_sign == 0:
        return False, "Data isyarat untuk training kosong."

    pos_weight = torch.tensor([num_idle / max(num_sign, 1)], dtype=torch.float32)

    train_dataset = SegmenterDataset(train_sequences, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_loader = None
    if len(val_sequences) > 0:
        val_dataset = SegmenterDataset(val_sequences, val_labels)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VADSegmenterModel(input_dim=INPUT_DIM, hidden_dim=64).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print(f"\n[DEVICE] Training Segmenter di: {device.type.upper()}")
    best_val_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        correct_train, total_train = 0, 0

        train_bar = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{epochs}]")
        for batch_seqs, batch_labels in train_bar:
            batch_seqs = batch_seqs.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch_seqs)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            predicted = (torch.sigmoid(outputs) > 0.5).float()
            total_train += batch_labels.size(0)
            correct_train += (predicted == batch_labels).sum().item()
            train_acc = 100.0 * correct_train / max(total_train, 1)
            train_bar.set_postfix({"Loss": f"{loss.item():.4f}", "Acc": f"{train_acc:.1f}%"})

        if val_loader is not None:
            model.eval()
            val_loss_total = 0.0
            correct_val, total_val = 0, 0

            with torch.inference_mode():
                for batch_seqs, batch_labels in val_loader:
                    batch_seqs = batch_seqs.to(device)
                    batch_labels = batch_labels.to(device)
                    outputs = model(batch_seqs)
                    v_loss = criterion(outputs, batch_labels)
                    val_loss_total += v_loss.item()

                    predicted = (torch.sigmoid(outputs) > 0.5).float()
                    total_val += batch_labels.size(0)
                    correct_val += (predicted == batch_labels).sum().item()

            avg_val_loss = val_loss_total / max(len(val_loader), 1)
            val_acc = 100.0 * correct_val / max(total_val, 1)
            print(f"  -> [VAL] Loss: {avg_val_loss:.4f} | Acc: {val_acc:.2f}%")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), SEGMENTER_WEIGHTS)
        else:
            torch.save(model.state_dict(), SEGMENTER_WEIGHTS)

    with open(SEGMENTER_METADATA, "w") as f:
        json.dump({"feature_schema": fe.FEATURE_SCHEMA, "target_frames": VAD_TARGET_FRAMES, "input_dim": INPUT_DIM}, f, indent=4)

    return True, "Pelatihan Segmenter Berhasil. Bobot terbaik telah disimpan."


if __name__ == "__main__":
    status, msg = train_segmenter()
    print(msg)
