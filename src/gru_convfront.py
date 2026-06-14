"""Conv1d front-end + GRU sequence classifier.

Two temporal Conv1d layers extract local motion patterns and halve the time axis
(60 -> 30) before a cuDNN GRU summarises the sequence (~0.27M params at 180-D).
The conv front-end denoises frame-level jitter and lightens the recurrent load.

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

CONV_CHANNELS = 128
GRU_HIDDEN = 128


class ConvFrontGRUModel(nn.Module):
    """Conv1d(k5) -> Conv1d(k3) -> MaxPool(2) -> GRU -> FC classifier."""

    target_frames = TARGET_FRAMES

    def __init__(
        self,
        input_dim: int = 180,
        num_classes: int = 10,
        conv_channels: int = CONV_CHANNELS,
        gru_hidden: int = GRU_HIDDEN,
        dropout: float = DEFAULT_DROPOUT,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_classes = int(num_classes)

        self.conv1 = nn.Conv1d(self.input_dim, conv_channels, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(conv_channels, conv_channels, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.act = nn.ReLU()
        self.conv_dropout = nn.Dropout(float(dropout))

        self.gru = nn.GRU(
            input_size=conv_channels,
            hidden_size=gru_hidden,
            num_layers=1,
            batch_first=True,
        )
        self.dropout = nn.Dropout(float(dropout))
        self.classifier = nn.Linear(gru_hidden, self.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, T, D) -> (B, D, T) for Conv1d over the time axis.
        x = x.transpose(1, 2)
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.pool(x)                  # (B, C, T/2)
        x = self.conv_dropout(x)
        x = x.transpose(1, 2)             # (B, T/2, C)
        _, h = self.gru(x)                # h: (1, B, H)
        feat = self.dropout(h[-1])
        return self.classifier(feat)


def build_model(input_dim: int, num_classes: int) -> ConvFrontGRUModel:
    return ConvFrontGRUModel(input_dim=input_dim, num_classes=num_classes)
