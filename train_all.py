"""Тренує всі 4 моделі послідовно і будує графіки навчання + якості.

Запуск:
    python train_all.py

Результати:
    checkpoints/<model>/best.pt
    checkpoints/<model>/history.json
    checkpoints/<model>/plots/  <- графіки
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# локальні імпорти
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import load_train_config
from train import train

# ── конфіги і порядок тренування ─────────────────────────────────────────────
CONFIGS = [
    # Path("configs/train_lcnn.yaml"),
    # Path("configs/train_se_resnet18.yaml"),
    # Path("configs/train_aasist_lite.yaml"),
    Path("configs/train_phase_aware.yaml"),
]


# ── графіки для однієї моделі ─────────────────────────────────────────────────

def plot_training(history: list[dict], model_name: str, out_dir: Path) -> None:
    """Loss + EER по епохах."""
    out_dir.mkdir(parents=True, exist_ok=True)
    epochs     = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_eer    = [h["val_eer"] * 100 for h in history]
    best_ep    = int(np.argmin(val_eer))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"{model_name} — навчання", fontsize=13)

    ax1.plot(epochs, train_loss, "b-o", markersize=3, label="train loss")
    ax1.set_xlabel("Епоха"); ax1.set_ylabel("Loss")
    ax1.set_title("Train Loss"); ax1.grid(alpha=0.3); ax1.legend()

    ax2.plot(epochs, val_eer, "r-o", markersize=3, label="val EER %")
    ax2.axvline(epochs[best_ep], color="green", linestyle="--",
                label=f"best ep {epochs[best_ep]} ({val_eer[best_ep]:.2f}%)")
    ax2.set_xlabel("Епоха"); ax2.set_ylabel("EER %")
    ax2.set_title("Validation EER"); ax2.grid(alpha=0.3); ax2.legend()

    plt.tight_layout()
    path = out_dir / "training_curves.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [plot] {path}")


def plot_comparison(all_history: dict[str, list[dict]], out_dir: Path) -> None:
    """Порівняння всіх моделей на одному графіку."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Порівняння моделей", fontsize=13)

    colors = ["steelblue", "tomato", "seagreen", "darkorange"]
    for (name, history), color in zip(all_history.items(), colors):
        epochs     = [h["epoch"] for h in history]
        train_loss = [h["train_loss"] for h in history]
        val_eer    = [h["val_eer"] * 100 for h in history]
        ax1.plot(epochs, train_loss, "-o", markersize=2,
                 color=color, label=name)
        ax2.plot(epochs, val_eer, "-o", markersize=2,
                 color=color, label=name)

    ax1.set_xlabel("Епоха"); ax1.set_ylabel("Loss")
    ax1.set_title("Train Loss"); ax1.grid(alpha=0.3); ax1.legend()

    ax2.set_xlabel("Епоха"); ax2.set_ylabel("EER %")
    ax2.set_title("Validation EER"); ax2.grid(alpha=0.3); ax2.legend()

    plt.tight_layout()
    path = out_dir / "comparison.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [plot] {path}")


def plot_final_bar(all_history: dict[str, list[dict]], out_dir: Path) -> None:
    """Фінальний bar chart: best EER і best train loss по моделях."""
    out_dir.mkdir(parents=True, exist_ok=True)
    names     = list(all_history.keys())
    best_eers = [min(h["val_eer"] for h in hist) * 100
                 for hist in all_history.values()]
    best_loss = [min(h["train_loss"] for h in hist)
                 for hist in all_history.values()]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Фінальні метрики по моделях", fontsize=13)

    colors = ["steelblue", "tomato", "seagreen", "darkorange"]
    bars1 = ax1.bar(names, best_eers, color=colors, edgecolor="white")
    ax1.set_ylabel("Best EER %"); ax1.set_title("Best Validation EER (↓ краще)")
    ax1.bar_label(bars1, fmt="%.2f%%", padding=3); ax1.grid(axis="y", alpha=0.3)

    bars2 = ax2.bar(names, best_loss, color=colors, edgecolor="white")
    ax2.set_ylabel("Loss"); ax2.set_title("Best Train Loss (↓ краще)")
    ax2.bar_label(bars2, fmt="%.4f", padding=3); ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = out_dir / "final_metrics.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [plot] {path}")


# ── головний цикл ─────────────────────────────────────────────────────────────

def main() -> None:
    all_history: dict[str, list[dict]] = {}
    plots_root = Path("checkpoints/plots")

    for cfg_path in CONFIGS:
        if not cfg_path.exists():
            print(f"[skip] конфіг не знайдено: {cfg_path}")
            continue

        cfg = load_train_config(cfg_path)
        model_name = cfg["model"]["name"]
        print(f"\n{'='*60}")
        print(f"  Тренування: {model_name}")
        print(f"{'='*60}")

        try:
            train(cfg)
        except Exception as e:
            print(f"[ERROR] {model_name}: {e}", file=sys.stderr)
            continue

        # читаємо history.json який train() зберігає сам
        history_path = Path(cfg["logging"]["out_dir"]) / "history.json"
        if history_path.exists():
            history = json.loads(history_path.read_text())
            all_history[model_name] = history
            # графіки для цієї моделі
            model_plots = Path(cfg["logging"]["out_dir"]) / "plots"
            plot_training(history, model_name, model_plots)
        else:
            print(f"  [warn] history.json не знайдено для {model_name}")

    # порівняльні графіки по всіх моделях
    if len(all_history) > 1:
        print(f"\n{'='*60}")
        print("  Будую порівняльні графіки...")
        plot_comparison(all_history, plots_root)
        plot_final_bar(all_history, plots_root)

    print(f"\n[done] натреновано {len(all_history)} моделей")
    print(f"  checkpoints/  <- ваги і history.json")
    print(f"  {plots_root}/  <- порівняльні графіки")


if __name__ == "__main__":
    main()
