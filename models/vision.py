"""A small CNN that turns a 64x64 image into a 128-d vector.

Speaker and listener each build their *own* VisionEncoder. They start with
different random weights and never share parameters, so whatever they agree
on has to travel through the message, not through a shared brain.
"""

from __future__ import annotations

import torch
from torch import nn


class VisionEncoder(nn.Module):
    """4 strided conv layers (64 -> 32 -> 16 -> 8 -> 4 pixels) then a linear layer to `out_dim`."""

    def __init__(self, out_dim: int = 128, channels: int = 32) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_ch = 3
        for _ in range(4):
            # Stride 2 halves the resolution each time, which keeps the network cheap on a CPU.
            layers += [nn.Conv2d(in_ch, channels, kernel_size=3, stride=2, padding=1), nn.ReLU()]
            in_ch = channels
        self.conv = nn.Sequential(*layers)
        self.fc = nn.Linear(channels * 4 * 4, out_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """images: (..., 3, 64, 64) floats in [0, 1] -> (..., out_dim)."""
        lead = images.shape[:-3]
        x = self.conv(images.reshape(-1, *images.shape[-3:]))
        x = self.fc(x.flatten(start_dim=1))
        return x.reshape(*lead, -1)
