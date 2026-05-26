"""Build dataset manifests for the four supported corpora and MLAAD."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Iterable, List, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "."))

from manifests import (
    build_add_manifest,
    build_asvspoof_manifest,
    build_inthewild_manifest,
    build_wavefake_manifest,
    merge_manifests,
)

def _write_csv(path: str | Path, rows: Iterable[Sequence[str]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["path", "label", "dataset"])
        for row in rows:
            w.writerow(row)

def build_mlaad_manifest(root: str | Path, out_csv: str | Path) -> int:
    root = Path(root)
    rows: List[List[str]] = []
    for wav in sorted(root.glob("**/*.wav")):
        label = "spoof" if "spoof" in str(wav).lower() else "bona-fide"
        rows.append([str(wav.relative_to(root)), label, "mlaad"])
    _write_csv(out_csv, rows)
    return len(rows)

def _try(builder, label: str, *args):
    try:
        n = builder(*args)
        print(f"  + {label}: {n} entries")
        return True
    except FileNotFoundError as exc:
        print(f"  ! {label}: skipped ({exc})", file=sys.stderr)
        return False

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--asvspoof-root", type=Path, default=None)
    p.add_argument("--wavefake-root", type=Path, default=None)
    p.add_argument("--ljspeech-root", type=Path, default=None)
    p.add_argument("--jsut-root", type=Path, default=None)
    p.add_argument("--inthewild-root", type=Path, default=None)
    p.add_argument("--add-root", type=Path, default=None)
    p.add_argument("--mlaad-root", type=Path, default=None)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--merge-train", action="store_true")
    args = p.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    produced: list[Path] = []

    if args.asvspoof_root is not None:
        for split in ("train", "dev", "eval"):
            out = args.out / f"asvspoof_{split}.csv"
            if _try(build_asvspoof_manifest, f"asvspoof_{split}", args.asvspoof_root, split, out):
                produced.append(out)

    if args.wavefake_root is not None:
        out = args.out / "wavefake.csv"
        if _try(build_wavefake_manifest, "wavefake", args.wavefake_root, args.ljspeech_root, args.jsut_root, out):
            produced.append(out)

    if args.inthewild_root is not None:
        out = args.out / "inthewild.csv"
        if _try(build_inthewild_manifest, "inthewild", args.inthewild_root, out):
            produced.append(out)

    if args.add_root is not None:
        for split in ("train", "dev"):
            out = args.out / f"add_{split}.csv"
            if _try(build_add_manifest, f"add_{split}", args.add_root, split, out):
                produced.append(out)

    if args.mlaad_root is not None:
        out = args.out / "mlaad.csv"
        if _try(build_mlaad_manifest, "mlaad", args.mlaad_root, out):
            produced.append(out)

    if args.merge_train:
        srcs: list[Path] = []
        for cand in (args.out / "asvspoof_train.csv", args.out / "wavefake.csv", args.out / "mlaad.csv"):
            if cand.exists():
                srcs.append(cand)
        if srcs:
            merged = args.out / "train.csv"
            n = merge_manifests(srcs, merged)
            print(f"  + merged train.csv: {n} entries")
            produced.append(merged)

    print(f"[done] wrote {len(produced)} manifests under {args.out}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())