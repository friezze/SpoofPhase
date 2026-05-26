"""Detector backbones registry — локальні імпорти без пакету spoof_phase."""
from typing import Dict
import torch.nn as nn

from models.aasist_lite import AASISTLite
from models.lcnn import LCNN
from models.phase_aware import PhaseAwareDualStream
from models.se_resnet import SEResNet18

_REGISTRY: Dict[str, type] = {
    "lcnn":         LCNN,
    "se_resnet18":  SEResNet18,
    "aasist_lite":  AASISTLite,
    "phase_aware":  PhaseAwareDualStream,
}

def build_model(name: str, in_channels: int = 2, num_classes: int = 2, **kw) -> nn.Module:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[name](in_channels=in_channels, num_classes=num_classes, **kw)

def available_models() -> list[str]:
    return sorted(_REGISTRY)

__all__ = ["build_model", "available_models",
           "LCNN", "SEResNet18", "AASISTLite", "PhaseAwareDualStream"]
