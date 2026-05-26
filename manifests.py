"""Manifest builders for the four supported datasets.

Each builder writes a CSV with three columns: ``path,label,dataset`` where
``label`` ∈ {``bona-fide``, ``spoof``} and ``path`` is **relative** to the
dataset root passed in. Downstream stages append the absolute root themselves;
this keeps manifests portable across machines.

Tested against the directory layout described in ``file_str.txt``:

    /data/
        asvspoof_2019/LA/...
        wavefake/generated_audio/<vocoder>/*.wav
        inthewild/release_in_the_wild/{*.wav, meta.csv}
        ADD/Track1.2/{train,dev}/...
"""
from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Iterable, List, Sequence


# --- ASVspoof 2019 LA ------------------------------------------------------

_ASV_CM_FILES = {
    "train": "LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt",
    "dev":   "LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt",
    "eval":  "LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt",
}

_ASV_AUDIO_DIRS = {
    "train": "LA/ASVspoof2019_LA_train/flac",
    "dev":   "LA/ASVspoof2019_LA_dev/flac",
    "eval":  "LA/ASVspoof2019_LA_eval/flac",
}


def build_asvspoof_manifest(root: str | Path, split: str,
                            out_csv: str | Path) -> int:
    """Parse the official ASVspoof2019 LA CM protocol into a CSV manifest.

    Protocol line format:  ``SPEAKER_ID FILE_ID SYSTEM_ID KEY LABEL``
    where ``LABEL`` is ``bonafide`` or ``spoof``.
    """
    root = Path(root)
    protocol = root / _ASV_CM_FILES[split]
    audio_dir = _ASV_AUDIO_DIRS[split]

    if not protocol.exists():
        raise FileNotFoundError(f"ASVspoof protocol not found: {protocol}")

    rows: List[List[str]] = []
    with protocol.open("r", encoding="utf-8") as fh:
        for raw in fh:
            parts = raw.strip().split()
            if len(parts) < 5:
                continue
            file_id, label = parts[1], parts[-1]
            rel = f"{audio_dir}/{file_id}.flac"
            tag = "bona-fide" if label.lower().startswith("bona") else "spoof"
            rows.append([rel, tag, "asvspoof2019_la"])
    _write_csv(out_csv, rows)
    return len(rows)


# --- WaveFake --------------------------------------------------------------

def build_wavefake_manifest(root: str | Path,
                            ljspeech_dir: str | Path | None,
                            jsut_dir: str | Path | None,
                            out_csv: str | Path) -> int:
    """Build a WaveFake manifest.

    WaveFake itself contains only **spoof** samples produced by six vocoders;
    the bona-fide counterparts must be sourced separately from the LJSpeech
    and JSUT corpora. ``ljspeech_dir`` / ``jsut_dir`` may be ``None`` if the
    corresponding bona-fide corpus is unavailable — only the spoof entries
    are then emitted.
    """
    root = Path(root)
    rows: List[List[str]] = []

    spoof_root = root / "generated_audio"
    if spoof_root.exists():
        for sub in sorted(spoof_root.iterdir()):
            if not sub.is_dir():
                continue
            for wav in sorted(sub.glob("*.wav")):
                rows.append([str(wav.relative_to(root)), "spoof", "wavefake"])

    if ljspeech_dir is not None:
        ljs = Path(ljspeech_dir)
        for wav in sorted(ljs.glob("**/*.wav")):
            rows.append([str(wav.resolve()), "bona-fide", "ljspeech"])

    if jsut_dir is not None:
        jsut = Path(jsut_dir)
        for wav in sorted(jsut.glob("**/*.wav")):
            rows.append([str(wav.resolve()), "bona-fide", "jsut"])

    _write_csv(out_csv, rows)
    return len(rows)


# --- In-the-Wild -----------------------------------------------------------

def build_inthewild_manifest(root: str | Path,
                             out_csv: str | Path) -> int:
    """Parse ``meta.csv`` of the In-the-Wild release."""
    root = Path(root)
    meta = root / "release_in_the_wild" / "meta.csv"
    if not meta.exists():
        raise FileNotFoundError(f"In-the-Wild meta.csv not found: {meta}")
    rows: List[List[str]] = []
    with meta.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for line in reader:
            # column names vary by release: try common spellings.
            file_col = next((k for k in ("file", "filename", "path") if k in line), None)
            label_col = next((k for k in ("label", "class", "target") if k in line), None)
            if not file_col or not label_col:
                continue
            label = "bona-fide" if line[label_col].strip().lower().startswith("bona") else "spoof"
            rel = f"release_in_the_wild/{line[file_col]}"
            rows.append([rel, label, "inthewild"])
    _write_csv(out_csv, rows)
    return len(rows)


# --- ADD 2023 Track 1.2 ----------------------------------------------------

def build_add_manifest(root: str | Path, split: str,
                       out_csv: str | Path) -> int:
    """ADD 2023 Track 1.2 manifest. Reads the per-split ``label.txt``.

    Each line is ``FILE_ID LABEL`` where LABEL ∈ {``genuine``, ``fake``}.
    """
    root = Path(root)
    track = root / "Track1.2" / split
    label_file = track / "label.txt"
    if not label_file.exists():
        # Some packs ship as ._label.txt (macOS dotfile) — try that
        label_file = track / "._label.txt"
    if not label_file.exists():
        raise FileNotFoundError(f"ADD label file not found under {track}")

    rows: List[List[str]] = []
    with label_file.open("r", encoding="utf-8", errors="ignore") as fh:
        for raw in fh:
            parts = raw.strip().split()
            if len(parts) < 2:
                continue
            file_id, lbl = parts[0], parts[1]
            label = "bona-fide" if lbl.lower().startswith(("gen", "real", "bona")) else "spoof"
            # ADD audio is usually .wav under split/wav/ — adjust if needed
            rel = f"Track1.2/{split}/wav/{file_id}.wav"
            rows.append([rel, label, f"add_{split}"])
    _write_csv(out_csv, rows)
    return len(rows)


# --- helpers ---------------------------------------------------------------

def _write_csv(path: str | Path, rows: Iterable[Sequence[str]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["path", "label", "dataset"])
        for row in rows:
            w.writerow(row)


def merge_manifests(inputs: Sequence[str | Path], out_csv: str | Path) -> int:
    """Concatenate several manifests, keeping the header once."""
    rows: List[List[str]] = []
    for src in inputs:
        with Path(src).open("r", encoding="utf-8") as fh:
            r = csv.reader(fh)
            header = next(r, None)
            for row in r:
                rows.append(row)
    _write_csv(out_csv, rows)
    return len(rows)
