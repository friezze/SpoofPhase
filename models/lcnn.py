"""LCNN — Light CNN baseline with Max-Feature-Map activation.

Reference
---------
Lavrentyeva G., Novoselov S., Malykh E., Kozlov A., Kudashev O., Shchemelinin V.
*Audio replay attack detection with deep learning frameworks*. Interspeech 2017.

This implementation matches the canonical 9-layer LCNN variant adapted to a
two-channel input. It is intentionally compact (~340 K params) and serves as a
non-attention baseline against which the SE-ResNet and PhaseAware variants are
compared.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MFM(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        if channels % 2 != 0:
            raise ValueError("MFM requires an even number of input channels.")
        self.half = channels // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = x[:, : self.half], x[:, self.half:]
        return torch.maximum(a, b)

def _conv_mfm(in_c: int, out_c: int, k: int, s: int = 1, p: int = 0) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_c, 2 * out_c, kernel_size=k, stride=s, padding=p),
        MFM(2 * out_c),
    )

class LCNN(nn.Module):
    def __init__(self, in_channels: int = 2, num_classes: int = 2) -> None:
        super().__init__()
        self.block1 = _conv_mfm(in_channels, 32, k=5, s=1, p=2)
        self.pool1 = nn.MaxPool2d(2)
        self.block2 = _conv_mfm(32, 32, k=1)
        self.block3 = _conv_mfm(32, 48, k=3, p=1)
        self.pool2 = nn.MaxPool2d(2)
        self.block4 = _conv_mfm(48, 48, k=1)
        self.block5 = _conv_mfm(48, 64, k=3, p=1)
        self.pool3 = nn.MaxPool2d(2)
        self.block6 = _conv_mfm(64, 64, k=1)
        self.block7 = _conv_mfm(64, 32, k=3, p=1)
        self.pool4 = nn.MaxPool2d(2)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(self.block1(x))
        x = self.pool2(self.block3(self.block2(x)))
        x = self.pool3(self.block5(self.block4(x)))
        x = self.pool4(self.block7(self.block6(x)))
        return self.classifier(x)

    @property
    def gradcam_target(self) -> nn.Module:
        return self.block7[0]