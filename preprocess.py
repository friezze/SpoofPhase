"""Preprocessing pipeline: resample → (optional) denoise → VAD-based segmentation.

The DeepFilterNet and Silero-VAD models are optional — if the corresponding
packages are unavailable, the pipeline falls back to identity denoising and a
trivial energy-based VAD respectively. This keeps the feature pipeline runnable
on minimal CPU-only setups (useful for the web demo).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torchaudio


TARGET_SR = 16_000


def _try_import_df():
    try:
        from df.enhance import enhance, init_df, load_audio, save_audio  # noqa: F401
        return enhance, init_df, load_audio, save_audio
    except Exception:  # pylint: disable=broad-except
        return None


def _try_import_silero():
    try:
        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            trust_repo=True,
        )
        return model, utils
    except Exception:  # pylint: disable=broad-except
        return None


@dataclass
class Cleaners:
    """Bundle of pretrained helpers. ``None`` entries mean fallback paths are used."""

    df_pack: Optional[Tuple] = None
    df_state: Optional[object] = None
    vad_model: Optional[object] = None
    vad_utils: Optional[Tuple] = None
    use_denoise: bool = True
    use_vad: bool = True


def init_cleaners(use_denoise: bool = True, use_vad: bool = True) -> Cleaners:
    """Lazily load DeepFilterNet and Silero VAD. Silent fallback if not installed."""
    df_pack = None
    df_state = None
    if use_denoise:
        df_mod = _try_import_df()
        if df_mod is not None:
            enhance, init_df, load_audio, save_audio = df_mod
            df_model, df_state, _ = init_df()
            df_pack = (enhance, init_df, load_audio, save_audio, df_model)

    vad_model = None
    vad_utils = None
    if use_vad:
        loaded = _try_import_silero()
        if loaded is not None:
            vad_model, vad_utils = loaded

    return Cleaners(
        df_pack=df_pack, df_state=df_state,
        vad_model=vad_model, vad_utils=vad_utils,
        use_denoise=use_denoise and df_pack is not None,
        use_vad=use_vad and vad_model is not None,
    )


def _energy_vad(wav: np.ndarray, sr: int,
                frame_ms: float = 30.0,
                threshold_db: float = -35.0) -> List[Tuple[int, int]]:
    """Fallback VAD: drop frames quieter than ``threshold_db`` dBFS."""
    frame = int(sr * frame_ms / 1000.0)
    if frame <= 0:
        return [(0, len(wav))]
    n = len(wav) // frame
    if n == 0:
        return [(0, len(wav))]
    frames = wav[: n * frame].reshape(n, frame)
    rms = np.sqrt((frames ** 2).mean(axis=1) + 1e-12)
    db = 20.0 * np.log10(rms + 1e-12)
    active = db > threshold_db
    spans: List[Tuple[int, int]] = []
    i = 0
    while i < n:
        if active[i]:
            j = i
            while j < n and active[j]:
                j += 1
            spans.append((i * frame, j * frame))
            i = j
        else:
            i += 1
    return spans or [(0, len(wav))]


def enhance_audio(input_path: str | Path,
                  output_path: str | Path,
                  cleaners: Cleaners) -> Path:
    """Run DeepFilterNet denoising (or copy if unavailable)."""
    input_path, output_path = Path(input_path), Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if cleaners.use_denoise and cleaners.df_pack is not None:
        enhance, _, load_audio, save_audio, df_model = cleaners.df_pack
        audio, _ = load_audio(str(input_path), sr=cleaners.df_state.sr())
        enhanced = enhance(df_model, cleaners.df_state, audio)
        save_audio(str(output_path), enhanced, cleaners.df_state.sr())
    else:
        wav, sr = torchaudio.load(str(input_path))
        torchaudio.save(str(output_path), wav, sr)
    return output_path


def _to_mono(wav: torch.Tensor) -> torch.Tensor:
    if wav.dim() == 2 and wav.size(0) > 1:
        return wav.mean(dim=0, keepdim=True)
    return wav if wav.dim() == 2 else wav.unsqueeze(0)


def resample_to_target(wav: torch.Tensor, sr: int) -> torch.Tensor:
    """Resample to ``TARGET_SR`` if needed."""
    if sr == TARGET_SR:
        return wav
    return torchaudio.functional.resample(wav, sr, TARGET_SR)


def split_by_vad(audio_path: str | Path,
                 output_dir: str | Path,
                 cleaners: Cleaners,
                 label: Optional[str] = None,
                 min_chunk_samples: int = TARGET_SR // 2) -> List[Path]:
    """Segment a canonical-rate WAV by voice activity and write chunk WAVs."""
    audio_path = Path(audio_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if cleaners.use_vad and cleaners.vad_model is not None and cleaners.vad_utils is not None:
        get_speech_timestamps, _, read_audio, _, _ = cleaners.vad_utils
        wav = read_audio(str(audio_path), sampling_rate=TARGET_SR)
        timestamps = get_speech_timestamps(wav, cleaners.vad_model,
                                           sampling_rate=TARGET_SR)
        spans = [(int(t["start"]), int(t["end"])) for t in timestamps]
    else:
        wav_t, _ = torchaudio.load(str(audio_path))
        wav_np = wav_t.mean(0).numpy()
        wav = torch.from_numpy(wav_np)
        spans = _energy_vad(wav_np, TARGET_SR)

    produced: List[Path] = []
    for i, (start, end) in enumerate(spans):
        if end - start < min_chunk_samples:
            continue
        chunk = wav[start:end].unsqueeze(0)
        out = output_dir / f"{audio_path.stem}_chunk_{i:04d}.wav"
        torchaudio.save(str(out), chunk, TARGET_SR)
        (out.with_suffix(".json")).write_text(
            json.dumps({
                "source": str(audio_path),
                "start_sample": start,
                "end_sample": end,
                "sample_rate": TARGET_SR,
                "label": label,
            }, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        produced.append(out)
    return produced


def preprocess_one(audio_path: str | Path,
                   work_dir: str | Path,
                   cleaners: Cleaners,
                   label: Optional[str] = None) -> List[Path]:
    """Full single-file pipeline: denoise → canonical 16 kHz mono → VAD chunks."""
    audio_path = Path(audio_path)
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    denoised = work_dir / f"{audio_path.stem}_denoised.wav"
    enhance_audio(audio_path, denoised, cleaners)

    wav, sr = torchaudio.load(str(denoised))
    wav = _to_mono(wav)
    wav = resample_to_target(wav, sr).squeeze(0)
    canonical = work_dir / f"{audio_path.stem}_16k.wav"
    torchaudio.save(str(canonical), wav.unsqueeze(0), TARGET_SR)

    chunks_dir = work_dir / f"{audio_path.stem}_chunks"
    return split_by_vad(canonical, chunks_dir, cleaners, label=label)


def preprocess_many(audio_paths: Iterable[str | Path],
                    work_dir: str | Path,
                    cleaners: Optional[Cleaners] = None,
                    label: Optional[str] = None) -> List[Path]:
    """Batch driver. Reuses a single Cleaners across files."""
    cleaners = cleaners or init_cleaners()
    out: List[Path] = []
    for path in audio_paths:
        out.extend(preprocess_one(path, work_dir, cleaners, label=label))
    return out


def load_canonical(path: str | Path) -> np.ndarray:
    """Load an audio file into a 16 kHz mono float32 numpy array (for the web app)."""
    wav, sr = torchaudio.load(str(path))
    wav = _to_mono(wav)
    wav = resample_to_target(wav, sr).squeeze(0)
    return wav.numpy().astype(np.float32)
