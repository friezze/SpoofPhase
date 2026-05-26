"""AASIST-lite — a lightweight spectro-temporal graph attention detector.

Inspired by Tak H., Patino J., Todisco M., Nautsch A., Evans N., Larcher A.
*AASIST: Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention
Networks*. ICASSP 2022, pp. 6367-6371.

This is a simplified reimplementation suitable for cross-comparison on the same
2-channel mel + group-delay input. We replace the full HS-GAL stack with a
single graph-attention layer over spectral and temporal token pools, keeping
the central idea (graph attention over spectro-temporal patches) while
reducing parameters to ~700 K. Use the official AASIST checkpoint if you need
the original metric.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ConvBN(nn.Sequential):
    def __init__(self, in_c: int, out_c: int, k: int = 3, s: int = 1, p: int = 1) -> None:
        super().__init__(
            nn.Conv2d(in_c, out_c, kernel_size=k, stride=s, padding=p, bias=False),
            nn.BatchNorm2d(out_c),
            nn.SELU(inplace=False),
        )

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim, bias=False)
        self.attn = nn.Linear(2 * out_dim, 1, bias=False)
        self.act = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.proj(x)
        B, N, D = h.shape
        a = h.unsqueeze(2).expand(B, N, N, D)
        b = h.unsqueeze(1).expand(B, N, N, D)
        e = self.act(self.attn(torch.cat([a, b], dim=-1))).squeeze(-1)
        alpha = F.softmax(e, dim=-1)
        return torch.bmm(alpha, h)

class AASISTLite(nn.Module):
    def __init__(self, in_channels: int = 2, num_classes: int = 2,
                 base_channels: int = 32, token_dim: int = 64) -> None:
        super().__init__()
        c = base_channels
        self.encoder = nn.Sequential(
            _ConvBN(in_channels, c),
            nn.MaxPool2d(2),
            _ConvBN(c, 2 * c),
            nn.MaxPool2d(2),
            _ConvBN(2 * c, 4 * c),
            nn.MaxPool2d(2),
        )
        self.spectral_pool = nn.AdaptiveAvgPool2d((16, 1))
        self.temporal_pool = nn.AdaptiveAvgPool2d((1, 32))

        self.spec_gat = GraphAttentionLayer(4 * c, token_dim)
        self.temp_gat = GraphAttentionLayer(4 * c, token_dim)

        self.classifier = nn.Sequential(
            nn.Linear(2 * token_dim, token_dim),
            nn.SELU(inplace=False),
            nn.Linear(token_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        spec = self.spectral_pool(h).squeeze(-1)
        temp = self.temporal_pool(h).squeeze(-2)
        spec = spec.transpose(1, 2)
        temp = temp.transpose(1, 2)
        spec = self.spec_gat(spec).mean(dim=1)
        temp = self.temp_gat(temp).mean(dim=1)
        return self.classifier(torch.cat([spec, temp], dim=-1))

    @property
    def gradcam_target(self) -> nn.Module:
        return self.encoder[4][0]
