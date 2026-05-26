"""Dataset-level visualisations (class balance, source mix, durations).

These are usually called once during EDA / report writing — they read a
manifest CSV and emit static PNGs / SVGs that you can paste straight into
the experimental chapter of the course report.
"""
from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _read_manifest(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def plot_class_distribution(manifest_csv: str | Path,
                            save_path: Optional[str | Path] = None,
                            title: Optional[str] = None):
    """Bar chart of ``bona-fide`` vs ``spoof`` counts."""
    rows = _read_manifest(manifest_csv)
    counts = Counter(r["label"] for r in rows)
    labels = ["bona-fide", "spoof"]
    values = [counts.get(lbl, 0) for lbl in labels]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, values, color=["#2ca02c", "#d62728"])
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v, str(v),
                ha="center", va="bottom")
    ax.set_ylabel("samples")
    ax.set_title(title or f"Class distribution — {Path(manifest_csv).name}")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return None
    return fig


def plot_dataset_distribution(manifest_csv: str | Path,
                              save_path: Optional[str | Path] = None,
                              title: Optional[str] = None):
    """Stacked bar of ``bona-fide`` / ``spoof`` per dataset tag."""
    rows = _read_manifest(manifest_csv)
    grid: dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        grid[r.get("dataset", "")][r["label"]] += 1
    datasets = sorted(grid.keys())
    bona = [grid[d].get("bona-fide", 0) for d in datasets]
    spoof = [grid[d].get("spoof", 0) for d in datasets]
    x = np.arange(len(datasets))
    fig, ax = plt.subplots(figsize=(max(6, len(datasets) * 1.5), 4))
    ax.bar(x, bona, label="bona-fide", color="#2ca02c")
    ax.bar(x, spoof, bottom=bona, label="spoof", color="#d62728")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=30, ha="right")
    ax.set_ylabel("samples")
    ax.set_title(title or f"Per-dataset balance — {Path(manifest_csv).name}")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return None
    return fig


def plot_duration_histogram(durations_s: np.ndarray,
                            bins: int = 50,
                            save_path: Optional[str | Path] = None,
                            title: str = "Per-sample duration"):
    """Histogram of sample durations (seconds)."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(durations_s, bins=bins, color="#1f77b4", alpha=0.85)
    ax.set_xlabel("duration, s")
    ax.set_ylabel("count")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return None
    return fig
