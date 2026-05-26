"""Per-sample visualisation plots: waveform, mel-spectrogram, group-delay,
Grad-CAM overlay, DET curve, and a 4-panel summary.

All functions accept an optional ``ax`` so they can be embedded in a multi-panel
figure built by the web app or a Jupyter notebook.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Dict

import matplotlib
matplotlib.use("Agg")          # safe for headless / web contexts
import matplotlib.pyplot as plt
import numpy as np



def _save_or_return(fig, save_path: Optional[str | Path]):
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return None
    return fig


def plot_waveform(wav: np.ndarray, sr: int = 16000,
                  ax: Optional[plt.Axes] = None,
                  save_path: Optional[str | Path] = None):
    """Time-domain waveform plot."""
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(8, 2))
    t = np.arange(len(wav)) / float(sr)
    ax.plot(t, wav, linewidth=0.6, color="#1f77b4")
    ax.set_xlabel("time, s")
    ax.set_ylabel("amplitude")
    ax.set_title("Waveform")
    ax.grid(True, alpha=0.3)
    if own_fig:
        return _save_or_return(fig, save_path)
    return None


def plot_mel_spectrogram(mel_db: np.ndarray, sr: int = 16000,
                         hop: int = 256,
                         ax: Optional[plt.Axes] = None,
                         save_path: Optional[str | Path] = None):
    """Log-mel spectrogram (dB) with proper time/frequency axes."""
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(8, 3))
    n_mels, frames = mel_db.shape
    extent = [0.0, frames * hop / sr, 0.0, n_mels]
    im = ax.imshow(mel_db, origin="lower", aspect="auto", cmap="magma", extent=extent)
    ax.set_xlabel("time, s")
    ax.set_ylabel("mel bin")
    ax.set_title("Log-mel spectrogram (dB)")
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="dB")
    if own_fig:
        return _save_or_return(fig, save_path)
    return None


def plot_group_delay(gd: np.ndarray, sr: int = 16000, hop: int = 256,
                     ax: Optional[plt.Axes] = None,
                     save_path: Optional[str | Path] = None):
    """Normalised group-delay map."""
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(8, 3))
    n_mels, frames = gd.shape
    extent = [0.0, frames * hop / sr, 0.0, n_mels]
    im = ax.imshow(gd, origin="lower", aspect="auto", cmap="viridis",
                   extent=extent)
    ax.set_xlabel("time, s")
    ax.set_ylabel("mel bin")
    ax.set_title("Group delay")
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    if own_fig:
        return _save_or_return(fig, save_path)
    return None




def plot_gradcam_overlay(base: np.ndarray, cam: np.ndarray,
                         title: str = "Grad-CAM overlay",
                         sr: int = 16000, hop: int = 256,
                         ax: Optional[plt.Axes] = None,
                         save_path: Optional[str | Path] = None):
    """Translucent jet overlay of ``cam`` over a normalised feature image ``base``."""
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(8, 3))
    if base.shape != cam.shape:
        raise ValueError(f"shape mismatch: base {base.shape}, cam {cam.shape}")
    base_norm = (base - base.min()) / (base.max() - base.min() + 1e-9)
    n_mels, frames = base.shape
    extent = [0.0, frames * hop / sr, 0.0, n_mels]
    ax.imshow(base_norm, origin="lower", aspect="auto", cmap="gray", extent=extent)
    ax.imshow(cam, origin="lower", aspect="auto", cmap="jet",
              alpha=0.5, extent=extent, vmin=0.0, vmax=1.0)
    ax.set_xlabel("time, s")
    ax.set_ylabel("mel bin")
    ax.set_title(title)
    if own_fig:
        return _save_or_return(fig, save_path)
    return None


def plot_features_panel(wav: np.ndarray, mel_db: np.ndarray, gd: np.ndarray,
                        cams: Optional[Dict[str, np.ndarray]] = None,
                        sr: int = 16000, hop: int = 256,
                        save_path: Optional[str | Path] = None,
                        title: Optional[str] = None):
    """Multi-panel summary plot dynamically sizing based on available Grad-CAMs."""
    n_cams = len(cams) if cams else 0
    n_panels = 3 + (n_cams if n_cams > 0 else 1)
    
    fig, axes = plt.subplots(n_panels, 1, figsize=(9, 2.5 * n_panels))
    
    plot_waveform(wav, sr=sr, ax=axes[0])
    plot_mel_spectrogram(mel_db, sr=sr, hop=hop, ax=axes[1])
    plot_group_delay(gd, sr=sr, hop=hop, ax=axes[2])
    
    if cams:
        for i, (model_name, cam) in enumerate(cams.items()):
            ax_idx = 3 + i
            plot_gradcam_overlay(mel_db, cam, title=f"Grad-CAM (over mel) - {model_name}",
                                 sr=sr, hop=hop, ax=axes[ax_idx])
    else:
        plot_group_delay(gd, sr=sr, hop=hop, ax=axes[3])
        axes[3].set_title("Group delay (duplicate; no Grad-CAM available)")
        
    if title:
        fig.suptitle(title, y=1.02)
    fig.tight_layout()
    return _save_or_return(fig, save_path)


def plot_det_curve(fars: Sequence[np.ndarray], frrs: Sequence[np.ndarray],
                   labels: Sequence[str],
                   save_path: Optional[str | Path] = None):
    """DET curve in log-deviate axes (canonical for ASV antispoofing)."""
    fig, ax = plt.subplots(figsize=(6, 6))
    for far, frr, lbl in zip(fars, frrs, labels):
        ax.plot(far * 100, frr * 100, label=lbl)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("False Acceptance Rate (%)")
    ax.set_ylabel("False Rejection Rate (%)")
    ax.set_title("DET curves")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    return _save_or_return(fig, save_path)
