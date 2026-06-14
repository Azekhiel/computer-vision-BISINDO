"""Temporal Convolutional Network (TCN) for BISINDO sign classification.

Four residual dilated-causal Conv1d blocks (dilations 1,2,4,8, kernel 3) grow the
receptive field to cover the whole 60-frame window without recurrence, then a
global average pool feeds the classifier (~0.43M params at 180-D). Fully
convolutional, so it parallelises well and is friendly to the Jetson GPU.

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
DEFAULT_DROPOUT = 0.20
DEFAULT_PATIENCE = 40

CHANNELS = 128
KERNEL_SIZE = 3
DILATIONS = (1, 2, 4, 8)


class _Chomp1d(nn.Module):
    """Trim the right padding so each block stays causal."""

    def __init__(self, chomp_size: int) -> None:
        super().__init__()
        self.chomp_size = int(chomp_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size == 0:
            return x
        return x[:, :, : -self.chomp_size].contiguous()


class _TemporalBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        self.chomp1 = _Chomp1d(pad)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        self.chomp2 = _Chomp1d(pad)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(float(dropout))
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.dropout(self.act(self.chomp1(self.conv1(x))))
        out = self.dropout(self.act(self.chomp2(self.conv2(out))))
        res = x if self.downsample is None else self.downsample(x)
        return self.act(out + res)


class TCNSignModel(nn.Module):
    """Stacked dilated TCN blocks -> global average pool -> FC classifier."""

    target_frames = TARGET_FRAMES

    def __init__(
        self,
        input_dim: int = 180,
        num_classes: int = 10,
        channels: int = CHANNELS,
        kernel_size: int = KERNEL_SIZE,
        dilations: tuple[int, ...] = DILATIONS,
        dropout: float = DEFAULT_DROPOUT,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_classes = int(num_classes)

        blocks: list[nn.Module] = []
        in_ch = self.input_dim
        for dilation in dilations:
            blocks.append(_TemporalBlock(in_ch, channels, kernel_size, dilation, dropout))
            in_ch = channels
        self.network = nn.Sequential(*blocks)
        self.dropout = nn.Dropout(float(dropout))
        self.classifier = nn.Linear(channels, self.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, T, D) -> (B, D, T) for Conv1d over the time axis.
        x = x.transpose(1, 2)
        x = self.network(x)
        pooled = torch.mean(x, dim=2)     # global average pool over time
        pooled = self.dropout(pooled)
        return self.classifier(pooled)


def build_model(input_dim: int, num_classes: int) -> TCNSignModel:
    return TCNSignModel(input_dim=input_dim, num_classes=num_classes)
