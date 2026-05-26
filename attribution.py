"""Channel-attribution helper for the dual-input detector.

By zeroing out one of the two input channels (mel or group delay) in turn we
get a quick "leave-one-channel-out" sensitivity estimate: how much the spoof
probability changes when each modality is ablated. This is *not* a substitute
for SHAP or LIME — it is a cheap sanity check that lets the operator decide
whether the model is making its decision on mel, group delay, or both.
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


@torch.no_grad()
def channel_ablation(model: torch.nn.Module,
                     x: torch.Tensor,
                     target_class: int = 1) -> Dict[str, float]:
    """Return ``{full, ablate_mel, ablate_gd, mel_importance, gd_importance}``.

    All probabilities are for ``target_class`` (default 1 = spoof). The two
    "*_importance" entries are the difference vs. the full-input score —
    larger drop = more important channel.
    """
    if x.dim() != 4 or x.size(0) != 1 or x.size(1) != 2:
        raise ValueError("Expected a (1, 2, F, T) tensor.")
    model.eval()
    full_p = float(F.softmax(model(x), dim=-1)[0, target_class])

    x_no_mel = x.clone()
    x_no_mel[:, 0] = 0.0
    p_no_mel = float(F.softmax(model(x_no_mel), dim=-1)[0, target_class])

    x_no_gd = x.clone()
    x_no_gd[:, 1] = 0.0
    p_no_gd = float(F.softmax(model(x_no_gd), dim=-1)[0, target_class])

    return {
        "full": full_p,
        "ablate_mel": p_no_mel,
        "ablate_gd": p_no_gd,
        "mel_importance": full_p - p_no_mel,
        "gd_importance": full_p - p_no_gd,
    }
