"""Evaluation entry point. Runs a trained model over a manifest and reports metrics."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets import NpzFeatureDataset, collate_pad
from metrics import det_curve, evaluate_classifier
from models import build_model


def _resolve_device(name: str) -> torch.device:
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _load_checkpoint(path: str | Path):
    return torch.load(str(path), map_location="cpu")


def evaluate(checkpoint_path: str | Path,
             cfg: Dict[str, Any],
             manifest_csv: str | Path | None = None,
             out_json: str | Path | None = None) -> Dict[str, Any]:
    """Compute EER / AUC / accuracy / DET curve on the given manifest."""
    ckpt = _load_checkpoint(checkpoint_path)
    model_name = ckpt.get("model_name") or cfg["model"]["name"]
    device = _resolve_device(cfg.get("device", "cuda"))

    model = build_model(model_name,
                        in_channels=cfg["model"].get("in_channels", 2),
                        num_classes=cfg["model"].get("num_classes", 2))
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()

    manifest = Path(manifest_csv or cfg["dataset"]["manifest_eval"])
    ds = NpzFeatureDataset(manifest, cfg["dataset"]["features_root"],
                           frame_target=cfg["dataset"]["frame_target"],
                           train=False)
    loader = DataLoader(ds, batch_size=cfg["optim"]["batch_size"],
                        shuffle=False, num_workers=2, collate_fn=collate_pad)

    scores = []
    labels = []
    per_ds: Dict[str, list] = {}
    with torch.no_grad():
        for feats, ys, tags in loader:
            feats = feats.to(device, non_blocking=True)
            logits = model(feats)
            probs = F.softmax(logits, dim=-1)[:, 1].cpu().numpy().tolist()
            scores.extend(probs)
            labels.extend(ys.numpy().tolist())
            for s, y, tag in zip(probs, ys.numpy().tolist(), tags):
                per_ds.setdefault(tag, []).append((float(s), int(y)))

    report = evaluate_classifier(scores, labels, threshold=ckpt.get("threshold"))
    far, frr = det_curve(scores, labels)

    out: Dict[str, Any] = {
        "checkpoint": str(checkpoint_path),
        "manifest": str(manifest),
        "model_name": model_name,
        "overall": report.as_dict(),
        "det_curve": {"far": far.tolist(), "frr": frr.tolist()},
        "by_dataset": {},
    }
    for tag, items in per_ds.items():
        s = [it[0] for it in items]
        y = [it[1] for it in items]
        out["by_dataset"][tag] = evaluate_classifier(s, y).as_dict()

    if out_json is not None:
        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(out_json).write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"[eval] {model_name} on {manifest}: "
          f"EER={report.eer*100:.2f}% AUC={report.auc:.4f} ACC={report.accuracy*100:.2f}%")
    return out
