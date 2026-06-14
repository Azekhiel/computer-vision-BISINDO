"""Compact Transformer encoder for BISINDO sign classification.

Input projection to d=128 + learned positional embeddings, two pre-norm
TransformerEncoder layers (4 heads, FF 256), then mean pooling over time
(~0.30M params at 180-D). Self-attention captures long-range frame relations
that a single GRU pass can miss.

Module contract (shared with gru_adi etc.): exposes TARGET_FRAMES,
DEFAULT_LR/BATCH_SIZE/EPOCHS/PATIENCE and ``build_model(input_dim, num_classes)``.
"""

from __future__ import annotations

import torch
from torch import nn


TARGET_FRAMES = 60
DEFAULT_LR = 5e-4
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 300
DEFAULT_DROPOUT = 0.20
DEFAULT_PATIENCE = 40

D_MODEL = 128
N_HEAD = 4
FF_DIM = 256
N_LAYERS = 2


class TransformerSignModel(nn.Module):
    """Linear proj + learned pos-emb -> TransformerEncoder x2 -> mean pool -> FC."""

    target_frames = TARGET_FRAMES

    def __init__(
        self,
        input_dim: int = 180,
        num_classes: int = 10,
        d_model: int = D_MODEL,
        n_head: int = N_HEAD,
        ff_dim: int = FF_DIM,
        n_layers: int = N_LAYERS,
        max_len: int = TARGET_FRAMES,
        dropout: float = DEFAULT_DROPOUT,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_classes = int(num_classes)
        self.max_len = int(max_len)

        self.input_proj = nn.Linear(self.input_dim, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.max_len, d_model))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=ff_dim,
            dropout=float(dropout),
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(float(dropout))
        self.classifier = nn.Linear(d_model, self.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        steps = x.shape[1]
        x = self.input_proj(x)
        x = x + self.pos_embedding[:, :steps, :]
        x = self.encoder(x)
        x = self.norm(x)
        pooled = torch.mean(x, dim=1)     # mean pool over time
        pooled = self.dropout(pooled)
        return self.classifier(pooled)


def build_model(input_dim: int, num_classes: int) -> TransformerSignModel:
    return TransformerSignModel(input_dim=input_dim, num_classes=num_classes)
