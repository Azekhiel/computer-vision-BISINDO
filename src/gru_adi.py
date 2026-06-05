"""Adi-style GRU architecture with ReLU candidate activation."""

from __future__ import annotations

import math

import torch
from torch import nn


TARGET_FRAMES = 60
DEFAULT_LR = 1e-3
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 300
DEFAULT_DROPOUT = 0.30
DEFAULT_PATIENCE = 40


class ReLUGRUCell(nn.Module):
    """GRU cell matching Keras GRU's configurable candidate activation."""

    def __init__(self, input_dim: int, hidden_dim: int, activation: str = "relu") -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.activation_name = activation
        self.weight_ih = nn.Parameter(torch.empty(3 * hidden_dim, input_dim))
        self.weight_hh = nn.Parameter(torch.empty(3 * hidden_dim, hidden_dim))
        self.bias_ih = nn.Parameter(torch.empty(3 * hidden_dim))
        self.bias_hh = nn.Parameter(torch.empty(3 * hidden_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.weight_ih)
        for chunk in self.weight_hh.chunk(3, dim=0):
            nn.init.orthogonal_(chunk)
        fan_in = self.input_dim + self.hidden_dim
        bound = 1.0 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias_ih, -bound, bound)
        nn.init.uniform_(self.bias_hh, -bound, bound)

    def _candidate_activation(self, value: torch.Tensor) -> torch.Tensor:
        if self.activation_name == "relu":
            return torch.relu(value)
        if self.activation_name == "tanh":
            return torch.tanh(value)
        raise ValueError(f"Unsupported GRU activation: {self.activation_name}")

    def forward(self, x: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        gi = torch.matmul(x, self.weight_ih.t()) + self.bias_ih
        gh = torch.matmul(h_prev, self.weight_hh.t()) + self.bias_hh
        i_z, i_r, i_n = gi.chunk(3, dim=-1)
        h_z, h_r, h_n = gh.chunk(3, dim=-1)

        z = torch.sigmoid(i_z + h_z)
        r = torch.sigmoid(i_r + h_r)
        n = self._candidate_activation(i_n + r * h_n)
        return z * h_prev + (1.0 - z) * n


class ReLUGRULayer(nn.Module):
    """Single-direction batch-first GRU layer with ReLU candidate activation."""

    def __init__(self, input_dim: int, hidden_dim: int, return_sequences: bool) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.return_sequences = bool(return_sequences)
        self.cell = ReLUGRUCell(input_dim=input_dim, hidden_dim=hidden_dim, activation="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, steps, _ = x.shape
        h = x.new_zeros(batch_size, self.hidden_dim)
        outputs = []
        for step in range(steps):
            h = self.cell(x[:, step, :], h)
            if self.return_sequences:
                outputs.append(h)
        if self.return_sequences:
            return torch.stack(outputs, dim=1)
        return h


def _batch_norm_time(batch_norm: nn.BatchNorm1d, x: torch.Tensor) -> torch.Tensor:
    return batch_norm(x.transpose(1, 2)).transpose(1, 2)


class AdiGRUModel(nn.Module):
    """Two-layer ReLU-GRU network adapted from Adi et al.'s paper."""

    target_frames = TARGET_FRAMES

    def __init__(
        self,
        input_dim: int = 180,
        num_classes: int = 10,
        dropout: float = DEFAULT_DROPOUT,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_classes = int(num_classes)

        self.gru1 = ReLUGRULayer(self.input_dim, 64, return_sequences=True)
        self.bn1 = nn.BatchNorm1d(64)
        self.dropout1 = nn.Dropout(float(dropout))

        self.gru2 = ReLUGRULayer(64, 128, return_sequences=False)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(float(dropout))

        self.dense = nn.Linear(128, 64)
        self.dropout3 = nn.Dropout(float(dropout))
        self.classifier = nn.Linear(64, self.num_classes)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.gru1(x)
        x = _batch_norm_time(self.bn1, x)
        x = self.dropout1(x)

        x = self.gru2(x)
        x = self.bn2(x)
        x = self.dropout2(x)

        x = self.activation(self.dense(x))
        x = self.dropout3(x)
        return self.classifier(x)


def build_model(input_dim: int, num_classes: int) -> AdiGRUModel:
    return AdiGRUModel(input_dim=input_dim, num_classes=num_classes)
