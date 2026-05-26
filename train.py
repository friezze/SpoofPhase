"""Training loop з підтримкою resume з будь-якого чекпоінту.

Якщо в out_dir є last.pt — продовжує з нього автоматично.
Кожна епоха зберігає last.pt (перезаписується) і best.pt (якщо EER покращився).
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets import NpzFeatureDataset, collate_pad
from transforms import ChannelMix, Compose, SpecAugment
from metrics import equal_error_rate
from models import build_model


def _seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _make_loaders(cfg: Dict[str, Any]):
    ds_cfg = cfg["dataset"]
    aug_cfg = cfg.get("augment", {})
    transform = Compose(
        SpecAugment(time_mask_width=aug_cfg.get("spec_mask_time", 30),
                    freq_mask_width=aug_cfg.get("spec_mask_freq", 12),
                    num_masks=aug_cfg.get("spec_mask_num", 2)),
        ChannelMix(delta=0.10),
    )
    train_ds = NpzFeatureDataset(
        ds_cfg["manifest_train"], ds_cfg["features_root"],
        frame_target=ds_cfg["frame_target"], train=True, transform=transform,
    )
    val_ds = NpzFeatureDataset(
        ds_cfg["manifest_val"],
        ds_cfg.get("features_root_val", ds_cfg["features_root"]),
        frame_target=ds_cfg["frame_target"], train=False,
    )
    bs = cfg["optim"]["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                              num_workers=4, pin_memory=True,
                              collate_fn=collate_pad, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False,
                            num_workers=2, pin_memory=True,
                            collate_fn=collate_pad)
    return train_loader, val_loader


def _build_scheduler(opt, cfg, steps_per_epoch: int, last_step: int = -1):
    sched = cfg["optim"].get("scheduler", "cosine")
    warmup = cfg["optim"].get("warmup_epochs", 0) * steps_per_epoch
    total = cfg["optim"]["epochs"] * steps_per_epoch
    if sched == "cosine":
        def lr_lambda(step: int) -> float:
            if step < warmup:
                return float(step) / max(1, warmup)
            progress = (step - warmup) / max(1, total - warmup)
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda,
                                                  last_epoch=last_step)
    if sched == "step":
        return torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.5,
                                               last_epoch=last_step)
    return None


def _resolve_device(name: str) -> torch.device:
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def _validate(model, loader, device):
    model.eval()
    scores, labels = [], []
    for feats, ys, _ in loader:
        feats = feats.to(device, non_blocking=True)
        logits = model(feats)
        probs = F.softmax(logits, dim=-1)
        scores.extend(probs[:, 1].detach().cpu().numpy().tolist())
        labels.extend(ys.numpy().tolist())
    eer, th = equal_error_rate(scores, labels)
    return eer, th


def train(cfg: Dict[str, Any]) -> Path:
    """Run training з auto-resume. Повертає шлях best checkpoint."""
    _seed(cfg.get("seed", 42))
    device = _resolve_device(cfg.get("device", "cuda"))
    out_dir = Path(cfg["logging"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader = _make_loaders(cfg)
    steps_per_epoch = len(train_loader)

    model = build_model(cfg["model"]["name"],
                        in_channels=cfg["model"].get("in_channels", 2),
                        num_classes=cfg["model"].get("num_classes", 2)).to(device)

    opt = torch.optim.AdamW(model.parameters(),
                            lr=cfg["optim"]["lr"],
                            weight_decay=cfg["optim"]["weight_decay"])

    # ── resume ───────────────────────────────────────────────────────────────
    last_path = out_dir / "last.pt"
    best_path = out_dir / "best.pt"
    history: list[dict] = []
    start_epoch = 0
    best_eer = float("inf")
    epochs_no_improve = 0
    last_step = -1

    if last_path.exists():
        print(f"[resume] завантажую {last_path}")
        ckpt = torch.load(str(last_path), map_location="cpu")
        model.load_state_dict(ckpt["model_state"])
        opt.load_state_dict(ckpt["opt_state"])
        start_epoch = ckpt["epoch"] + 1
        best_eer = ckpt.get("best_eer", float("inf"))
        epochs_no_improve = ckpt.get("epochs_no_improve", 0)
        history = ckpt.get("history", [])
        last_step = ckpt.get("global_step", -1)
        print(f"[resume] продовжую з епохи {start_epoch}, best EER={best_eer*100:.2f}%")
    else:
        print(f"[train] старт з нуля: {cfg['model']['name']}")

    sched = _build_scheduler(opt, cfg, steps_per_epoch, last_step=last_step)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    smoothing = cfg["optim"].get("label_smoothing", 0.0)
    patience = cfg["optim"].get("early_stopping_patience", 3)
    total_epochs = cfg["optim"]["epochs"]

    # якщо вже натреновано повністю — виходимо
    if start_epoch >= total_epochs:
        print(f"[skip] {cfg['model']['name']} вже натреновано ({total_epochs} епох)")
        return best_path

    # ── training loop ────────────────────────────────────────────────────────
    global_step = last_step + 1

    for epoch in range(start_epoch, total_epochs):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for step, (feats, ys, _tags) in enumerate(train_loader):
            feats = feats.to(device, non_blocking=True)
            ys = ys.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                logits = model(feats)
                loss = F.cross_entropy(logits, ys, label_smoothing=smoothing)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            if sched is not None:
                sched.step()
            epoch_loss += loss.item()
            global_step += 1
            if step % cfg["logging"].get("log_every", 50) == 0:
                cur_lr = opt.param_groups[0]["lr"]
                print(f"[ep {epoch:03d} st {step:04d}] "
                      f"loss={loss.item():.4f} lr={cur_lr:.2e}")

        train_loss = epoch_loss / max(1, len(train_loader))
        eer, th = _validate(model, val_loader, device)
        epoch_t = time.time() - t0
        print(f"[ep {epoch:03d}] train_loss={train_loss:.4f} "
              f"val_eer={eer*100:.2f}%  ({epoch_t:.1f}s)")

        history.append({"epoch": epoch, "train_loss": train_loss,
                        "val_eer": eer, "threshold": th, "time_s": epoch_t})

        # best checkpoint
        if eer < best_eer:
            best_eer = eer
            epochs_no_improve = 0
            torch.save({"model_state": model.state_dict(),
                        "model_name": cfg["model"]["name"],
                        "config": cfg,
                        "val_eer": eer,
                        "threshold": th}, best_path)
            print(f"  [best] EER={eer*100:.2f}% -> {best_path}")
        else:
            epochs_no_improve += 1

        # last checkpoint — завжди після кожної епохи
        torch.save({"model_state": model.state_dict(),
                    "model_name": cfg["model"]["name"],
                    "opt_state": opt.state_dict(),
                    "config": cfg,
                    "epoch": epoch,
                    "best_eer": best_eer,
                    "epochs_no_improve": epochs_no_improve,
                    "global_step": global_step,
                    "history": history}, last_path)

        # periodical checkpoint
        if (epoch + 1) % cfg["logging"].get("save_every", 5) == 0:
            torch.save({"model_state": model.state_dict(),
                        "model_name": cfg["model"]["name"],
                        "config": cfg,
                        "epoch": epoch},
                       out_dir / f"epoch_{epoch:03d}.pt")

        # early stopping
        if epochs_no_improve >= patience:
            print(f"[early stop] {patience} епох без покращення -> стоп")
            break

    (out_dir / "history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8")
    print(f"[done] best val EER = {best_eer*100:.2f}%  -> {best_path}")
    return best_path