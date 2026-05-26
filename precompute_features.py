"""Parallel pre-computation of mel + group-delay tensors from a manifest.

The output mirrors the manifest path structure (``manifest path`` →
``features_root/<path>.npz``) and an aggregated ``manifest.csv`` is written
alongside, ready to be consumed by ``NpzFeatureDataset``.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "."))

from features import get_audio_features, stack_features  # noqa: E402


def _read_manifest(path: Path) -> list[tuple[str, str, str]]:
    out: list[tuple[str, str, str]] = []
    with path.open("r", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            out.append((row["path"], row["label"], row.get("dataset", "")))
    return out


def _process(args):
    rel, audio_root, feature_root, label, dataset = args
    audio_path = Path(audio_root) / rel
    feat_path = Path(feature_root) / Path(rel).with_suffix(".npz")
    try:
        mel, gd, _sr, _wav = get_audio_features(audio_path)
        feat = stack_features(mel, gd).astype(np.float16)
        feat_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(feat_path, features=feat)
        return (str(Path(rel).with_suffix(".npz")), label, dataset,
                int(feat.shape[1]), int(feat.shape[2]), None)
    except Exception as exc:  # pylint: disable=broad-except
        return (rel, label, dataset, 0, 0, repr(exc))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("manifests", type=Path, nargs="+",
                   help="one or more manifest CSVs to process")
    p.add_argument("--audio-root", type=Path, required=True,
                   help="directory the manifest paths are relative to")
    p.add_argument("--features-root", type=Path, required=True,
                   help="output directory for NPZ tensors")
    p.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    args = p.parse_args(argv)

    items: list[tuple[str, str, str, str, str]] = []
    for mf in args.manifests:
        for rel, label, dataset in _read_manifest(mf):
            items.append((rel, str(args.audio_root),
                          str(args.features_root), label, dataset))

    args.features_root.mkdir(parents=True, exist_ok=True)
    out_manifest = args.features_root / "manifest.csv"
    errors: list[str] = []

    with out_manifest.open("w", newline="", encoding="utf-8") as fh, \
         ProcessPoolExecutor(max_workers=args.workers) as pool:
        writer = csv.writer(fh)
        writer.writerow(["path", "label", "dataset", "n_mels", "frames"])
        futures = [pool.submit(_process, it) for it in items]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="extract"):
            rel, label, dataset, n_mels, frames, err = fut.result()
            if err:
                errors.append(f"{rel}\t{err}")
                continue
            writer.writerow([rel, label, dataset, n_mels, frames])

    if errors:
        (args.features_root / "errors.log").write_text(
            "\n".join(errors), encoding="utf-8")
        print(f"[precompute] {len(errors)} failures -> errors.log",
              file=sys.stderr)
    print(f"[precompute] wrote {out_manifest} ({len(items) - len(errors)} entries)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
