"""Bidirectional GRU with temporal attention pooling.

Compact sequence classifier (~0.2M params at 180-D input): LayerNorm front,
a single bidirectional cuDNN GRU, then attention pooling over time instead of
taking only the last hidden state — so frames that matter most for the sign
dominate the pooled representation.

Module contract (shared with gru_adi etc.): exposes TARGET_FRAMES,
DEFAULT_LR/BATCH_SIZE/EPOCHS/PATIENCE and ``build_model(input_dim, num_classes)``.
"""

from __future__ import annotations

import torch
from torch import nn


TARGET_FRAMES = 60
DEFAULT_LR = 1e-3
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 300
DEFAULT_DROPOUT = 0.30
DEFAULT_PATIENCE = 40

HIDDEN_DIM = 96


class BiGRUAttentionModel(nn.Module):
    """LayerNorm -> BiGRU -> additive attention pooling -> FC classifier."""

    target_frames = TARGET_FRAMES

    def __init__(
        self,
        input_dim: int = 180,
        num_classes: int = 10,
        hidden_dim: int = HIDDEN_DIM,
        dropout: float = DEFAULT_DROPOUT,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_classes = int(num_classes)
        self.hidden_dim = int(hidden_dim)

        self.input_norm = nn.LayerNorm(self.input_dim)
        self.gru = nn.GRU(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        feat_dim = self.hidden_dim * 2
        self.attn = nn.Linear(feat_dim, 1)
        self.dropout = nn.Dropout(float(dropout))
        self.classifier = nn.Linear(feat_dim, self.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_norm(x)
        seq, _ = self.gru(x)                       # (B, T, 2H)
        scores = self.attn(seq)                    # (B, T, 1)
        weights = torch.softmax(scores, dim=1)     # attention over time
        pooled = torch.sum(weights * seq, dim=1)   # (B, 2H)
        pooled = self.dropout(pooled)
        return self.classifier(pooled)


def build_model(input_dim: int, num_classes: int) -> BiGRUAttentionModel:
    return BiGRUAttentionModel(input_dim=input_dim, num_classes=num_classes)
