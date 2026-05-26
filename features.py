"""Two-channel feature extraction: log-mel spectrogram (dB) + group delay (normalised).

Both channels share the same ``(n_mels, T)`` geometry so they can be stacked
into the network's 2-channel input tensor without any cross-shape projection.

Mathematical reference
----------------------
The group-delay map is computed via the product-spectrum form
(Yegnanarayana & Murthy, 1992)::

    τ(m, k) = Re[X*(m,k) · Y(m,k)] / (|X(m,k)|² + ε),   Y = STFT(n · x[n]).

This avoids the numerically unstable ``-dφ/dω`` formulation in low-energy
frequency bins.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import librosa
import numpy as np


TARGET_SR = 16_000


@dataclass
class FeatureConfig:
    """Parameters controlling all STFT-based extraction. Stable across the project."""

    sample_rate: int = TARGET_SR
    n_fft: int = 1024
    hop_length: int = 256
    win_length: int = 1024
    n_mels: int = 128
    fmin: float = 0.0
    fmax: float = 8000.0
    eps: float = 1e-10
    gd_clip: float = 200.0


DEFAULT = FeatureConfig()


def _load_mono(path: str | Path, sr: int) -> np.ndarray:
    y, _ = librosa.load(str(path), sr=sr, mono=True)
    return y.astype(np.float32)


def mel_db(y: np.ndarray, cfg: FeatureConfig = DEFAULT) -> np.ndarray:
    """Log-power mel-spectrogram in decibel scale, shape ``(n_mels, T)``."""
    spec = librosa.feature.melspectrogram(
        y=y, sr=cfg.sample_rate, n_fft=cfg.n_fft, hop_length=cfg.hop_length,
        win_length=cfg.win_length, n_mels=cfg.n_mels, fmin=cfg.fmin, fmax=cfg.fmax,
        power=2.0,
    )
    return librosa.power_to_db(spec + cfg.eps).astype(np.float32)


def group_delay(y: np.ndarray, cfg: FeatureConfig = DEFAULT) -> np.ndarray:
    """Stable group-delay map on linear-frequency grid (n_fft/2 + 1 bins)."""
    x = librosa.stft(y, n_fft=cfg.n_fft, hop_length=cfg.hop_length,
                     win_length=cfg.win_length)
    n_axis = np.arange(len(y), dtype=np.float32)
    yw = librosa.stft(n_axis * y, n_fft=cfg.n_fft, hop_length=cfg.hop_length,
                      win_length=cfg.win_length)
    num = np.real(np.conj(x) * yw)
    den = (np.abs(x) ** 2) + cfg.eps
    gd = num / den
    return np.clip(gd, -cfg.gd_clip, cfg.gd_clip).astype(np.float32)


def reduce_to_mel_grid(gd: np.ndarray, cfg: FeatureConfig = DEFAULT) -> np.ndarray:
    """Aggregate linear-frequency group delay into ``cfg.n_mels`` bands.

    Phase decorrelates rapidly with frequency, so we keep only the dominant
    component per mel band (``argmax |τ|`` preserving sign).
    """
    mel_basis = librosa.filters.mel(
        sr=cfg.sample_rate, n_fft=cfg.n_fft, n_mels=cfg.n_mels,
        fmin=cfg.fmin, fmax=cfg.fmax,
    )
    band_masks = mel_basis > 0
    out = np.zeros((cfg.n_mels, gd.shape[1]), dtype=np.float32)
    cols = np.arange(gd.shape[1])
    for b, mask in enumerate(band_masks):
        if not mask.any():
            continue
        band = gd[mask, :]
        idx = np.argmax(np.abs(band), axis=0)
        out[b, :] = band[idx, cols]
    return out


def normalise_gd(gd: np.ndarray, cfg: FeatureConfig = DEFAULT) -> np.ndarray:
    """Map clipped group-delay values from ``[-gd_clip, +gd_clip]`` to ``[0, 1]``."""
    return ((gd + cfg.gd_clip) / (2 * cfg.gd_clip)).astype(np.float32)


def get_audio_features(path: str | Path,
                       cfg: FeatureConfig = DEFAULT
                       ) -> Tuple[np.ndarray, np.ndarray, int, np.ndarray]:
    """Return ``(mel_db, group_delay_norm, sample_rate, waveform)``."""
    y = _load_mono(path, cfg.sample_rate)
    mel = mel_db(y, cfg)
    gd = group_delay(y, cfg)
    gd = reduce_to_mel_grid(gd, cfg)
    gd = normalise_gd(gd, cfg)
    return mel, gd, cfg.sample_rate, y


def stack_features(mel: np.ndarray, gd: np.ndarray) -> np.ndarray:
    """Return the ``(2, n_mels, T)`` tensor expected by the detector models."""
    return np.stack([mel, gd], axis=0).astype(np.float32)


def features_from_waveform(y: np.ndarray,
                           cfg: FeatureConfig = DEFAULT
                           ) -> np.ndarray:
    """Compute stacked features from an already-loaded waveform (used by the web app)."""
    mel = mel_db(y, cfg)
    gd = group_delay(y, cfg)
    gd = reduce_to_mel_grid(gd, cfg)
    gd = normalise_gd(gd, cfg)
    return stack_features(mel, gd)
