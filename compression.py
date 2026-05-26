"""Channel-degradation simulation: Opus, AMR-WB, narrow-band radio mask.

Used in two roles:
    * scripts/apply_compression.py — build offline stress-test partitions
    * spoof_phase.data.transforms — on-the-fly augmentation during training

All paths are passed as argument vectors (no shell), and each run uses its own
``TemporaryDirectory`` so parallel callers never collide on intermediate files.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List


_FFMPEG = shutil.which("ffmpeg") or "ffmpeg"


def _run(cmd: List[str]) -> None:
    subprocess.run(cmd, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _two_pass(input_path: str | Path,
              output_path: str | Path,
              encode_args: List[str],
              codec_ext: str) -> Path:
    input_path, output_path = Path(input_path), Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_codec = Path(tmp) / f"chunk{codec_ext}"
        _run([_FFMPEG, "-y", "-i", str(input_path),
              *encode_args, str(tmp_codec)])
        _run([_FFMPEG, "-y", "-i", str(tmp_codec),
              "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(output_path)])
    return output_path


def apply_opus(input_path: str | Path, output_path: str | Path,
               bitrate: str = "16k") -> Path:
    """Opus encode-decode round-trip (Telegram/WhatsApp profile)."""
    return _two_pass(input_path, output_path,
                     ["-c:a", "libopus", "-b:a", bitrate], ".opus")


def apply_amr(input_path: str | Path, output_path: str | Path,
              bitrate: str = "12.65k") -> Path:
    """AMR-WB cellular encode-decode round-trip."""
    return _two_pass(input_path, output_path,
                     ["-ar", "16000", "-c:a", "amr_wb", "-b:a", bitrate],
                     ".amr")


def apply_radio(input_path: str | Path, output_path: str | Path) -> Path:
    """Tactical-radio mask: 300–3000 Hz band-pass at 8 kHz then re-up to 16 kHz."""
    input_path, output_path = Path(input_path), Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_wav = Path(tmp) / "radio.wav"
        _run([_FFMPEG, "-y", "-i", str(input_path),
              "-af", "highpass=f=300,lowpass=f=3000,aformat=sample_rates=8000",
              "-c:a", "pcm_s16le", str(tmp_wav)])
        _run([_FFMPEG, "-y", "-i", str(tmp_wav),
              "-ar", "16000", "-c:a", "pcm_s16le", str(output_path)])
    return output_path


# Convenience dispatcher for use in augmentation transforms.
CODECS = {"opus": apply_opus, "amr": apply_amr, "radio": apply_radio}


def degrade(input_path: str | Path, output_path: str | Path,
            codec: str, **kw) -> Path:
    """Dispatch by codec name; raises ``KeyError`` for unknown codecs."""
    return CODECS[codec](input_path, output_path, **kw)
