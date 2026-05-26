"""PhaseAware-DualStream — proposed-but-not-yet-validated detector.

Hypothesis
----------
Mel and group-delay channels carry **complementary** information whose
optimal combination changes from sample to sample. Forcing a shared
convolutional stem (as in SE-ResNet) couples the two streams at the lowest
level and may under-use group delay on samples where mel is "clean" enough
to dominate gradients.

Design
------
* Two **independent** small ResNet stems, one per channel.
* The resulting feature maps are tokenised (one token per time frame).
* A two-layer transformer with **cross-attention** between the streams
  fuses tokens; we then mean-pool over time and classify.

Status
------
This model is the "proposed-but-not-validated" architecture promised by
``Vision.txt`` and required by the user's task statement. Its weights are
not shipped with the archive; the file exists so that the registry returns
a working ``nn.Module`` (and so that the web UI can load a checkpoint when
the user trains one). See ``docs/DEVIATIONS.md`` D6 for the research log.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _Stem(nn.Module):
    def __init__(self, in_c: int = 1, out_c: int = 64) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_c, out_c // 2, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(out_c // 2),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2),
            nn.Conv2d(out_c // 2, out_c, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)

class _CrossAttentionBlock(nn.Module):
    def __init__(self, dim: int, heads: int = 4) -> None:
        super().__init__()
        self.mel_to_gd = nn.MultiheadAttention(dim, num_heads=heads, batch_first=True)
        self.gd_to_mel = nn.MultiheadAttention(dim, num_heads=heads, batch_first=True)
        self.ln_a = nn.LayerNorm(dim)
        self.ln_b = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim),
        )

    def forward(self, mel_tok: torch.Tensor, gd_tok: torch.Tensor):
        a, _ = self.mel_to_gd(mel_tok, gd_tok, gd_tok)
        b, _ = self.gd_to_mel(gd_tok, mel_tok, mel_tok)
        mel_tok = self.ln_a(mel_tok + a)
        gd_tok = self.ln_b(gd_tok + b)
        fused = mel_tok + gd_tok
        return fused + self.ffn(fused)

class PhaseAwareDualStream(nn.Module):
    def __init__(self, in_channels: int = 2, num_classes: int = 2,
                 stem_channels: int = 64, fuse_dim: int = 128) -> None:
        super().__init__()
        if in_channels != 2:
            raise ValueError("PhaseAwareDualStream requires exactly 2 input channels.")
        self.stem_mel = _Stem(in_c=1, out_c=stem_channels)
        self.stem_gd = _Stem(in_c=1, out_c=stem_channels)
        self.proj_mel = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, None)),
            nn.Conv2d(stem_channels, fuse_dim, kernel_size=1),
        )
        self.proj_gd = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, None)),
            nn.Conv2d(stem_channels, fuse_dim, kernel_size=1),
        )
        self.cross = _CrossAttentionBlock(dim=fuse_dim, heads=4)
        self.classifier = nn.Sequential(
            nn.Linear(fuse_dim, fuse_dim),
            nn.GELU(),
            nn.Linear(fuse_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mel = x[:, 0:1]
        gd = x[:, 1:2]
        fm = self.stem_mel(mel)
        fg = self.stem_gd(gd)
        tm = self.proj_mel(fm).squeeze(2).transpose(1, 2)
        tg = self.proj_gd(fg).squeeze(2).transpose(1, 2)
        fused = self.cross(tm, tg).mean(dim=1)
        return self.classifier(fused)

    @property
    def gradcam_target(self) -> nn.Module:
        return self.stem_mel.body[4]