"""Khukuh-style GRU architecture for Smart V8 BISINDO features."""

from __future__ import annotations

import torch
from torch import nn


TARGET_FRAMES = 30
DEFAULT_LR = 1e-4
DEFAULT_BATCH_SIZE = 64
DEFAULT_EPOCHS = 100
DEFAULT_DROPOUT_INPUT = 0.10
DEFAULT_DROPOUT_BLOCK = 0.20


class KhukuhGRUModel(nn.Module):
    """Three-stage GRU network adapted from Khukuh Prihatmikho's paper.

    The paper uses 30 frames and 1629 MediaPipe Holistic values per frame. This
    version preserves the temporal layout and layer sizes, while accepting the
    local Smart V8 180-D feature contract used by the current dataset.
    """

    target_frames = TARGET_FRAMES

    def __init__(
        self,
        input_dim: int = 180,
        num_classes: int = 10,
        dropout_input: float = DEFAULT_DROPOUT_INPUT,
        dropout_block: float = DEFAULT_DROPOUT_BLOCK,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_classes = int(num_classes)

        self.input_dropout = nn.Dropout(float(dropout_input))

        self.gru1 = nn.GRU(self.input_dim, 128, batch_first=True)
        self.dense1 = nn.Linear(128, 64)
        self.dropout1 = nn.Dropout(float(dropout_block))

        self.gru2 = nn.GRU(64, 64, batch_first=True)
        self.dense2 = nn.Linear(64, 64)
        self.dropout2 = nn.Dropout(float(dropout_block))

        self.gru3 = nn.GRU(64, 32, batch_first=True)
        self.dense3 = nn.Linear(32, 64)
        self.dropout3 = nn.Dropout(float(dropout_block))

        self.batch_norm = nn.BatchNorm1d(64)
        self.classifier = nn.Linear(64, self.num_classes)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_dropout(x)

        x, _ = self.gru1(x)
        x = self.activation(self.dense1(x))
        x = self.dropout1(x)

        x, _ = self.gru2(x)
        x = self.activation(self.dense2(x))
        x = self.dropout2(x)

        x, _ = self.gru3(x)
        x = x[:, -1, :]

        x = self.activation(self.dense3(x))
        x = self.dropout3(x)
        x = self.batch_norm(x)
        return self.classifier(x)


def build_model(input_dim: int, num_classes: int) -> KhukuhGRUModel:
    return KhukuhGRUModel(input_dim=input_dim, num_classes=num_classes)
