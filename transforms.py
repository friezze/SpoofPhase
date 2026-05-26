"""On-the-fly augmentations applied to ``(2, n_mels, T)`` feature tensors.

Two augmentations are implemented:

* ``SpecAugment`` — frequency- and time-axis masking (Park et al., 2019).
* ``ChannelMix`` — random per-channel scaling within ±10 %, simulating mild
  amplitude rebalancing between the mel and group-delay streams.

Codec-based augmentation (Opus/AMR/radio) is performed offline via
``scripts/apply_compression.py`` because the FFmpeg round-trip is too slow
for in-loop application on every batch.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

import torch


@dataclass
class SpecAugment:
    """Time + frequency masking. See Park et al., Interspeech 2019."""

    time_mask_width: int = 30
    freq_mask_width: int = 12
    num_masks: int = 2

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        out = x.clone()
        _C, F, T = out.shape
        for _ in range(self.num_masks):
            if self.freq_mask_width > 0 and F > self.freq_mask_width:
                w = random.randint(0, self.freq_mask_width)
                f0 = random.randint(0, F - w)
                out[:, f0:f0 + w, :] = 0.0
            if self.time_mask_width > 0 and T > self.time_mask_width:
                w = random.randint(0, self.time_mask_width)
                t0 = random.randint(0, T - w)
                out[:, :, t0:t0 + w] = 0.0
        return out


@dataclass
class ChannelMix:
    """Per-channel multiplicative jitter (limited)."""

    delta: float = 0.10

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        gains = 1.0 + (torch.rand(x.size(0), 1, 1) * 2 - 1) * self.delta
        return x * gains.to(x.dtype)


class Compose:
    """Trivial transform composer (avoids torchvision dependency)."""

    def __init__(self, *transforms) -> None:
        self.transforms = transforms

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        for tfm in self.transforms:
            x = tfm(x)
        return x
