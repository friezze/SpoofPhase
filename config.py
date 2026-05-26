"""YAML configuration loaders.

Configs are intentionally plain dictionaries — no pydantic / dataclass coupling —
so that callers can override fields ad-hoc from CLI or tests without paying for
a heavyweight schema layer. Where typing safety matters (e.g. ``FeatureConfig``
in ``features.py``), a dataclass is provided locally to that module.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def load_yaml(path: str | Path) -> Dict[str, Any]:
    """Load a YAML file into a plain ``dict``."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Top-level YAML in {path} must be a mapping, got {type(data).__name__}")
    return data


def load_train_config(path: str | Path) -> Dict[str, Any]:
    """Load and lightly validate a training config."""
    cfg = load_yaml(path)
    required = {"dataset", "feature", "model", "optim"}
    missing = required.difference(cfg)
    if missing:
        raise KeyError(f"train config {path} missing required keys: {sorted(missing)}")
    return cfg


def load_infer_config(path: str | Path) -> Dict[str, Any]:
    """Load and lightly validate an inference / web config."""
    cfg = load_yaml(path)
    required = {"models", "device"}
    missing = required.difference(cfg)
    if missing:
        raise KeyError(f"infer config {path} missing required keys: {sorted(missing)}")
    return cfg
