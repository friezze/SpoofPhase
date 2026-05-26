"""Thin wrapper around ``spoof_phase.evaluate.evaluate``."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "."))

from config import load_train_config  # noqa: E402
from evaluate import evaluate  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("config", type=Path)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--manifest", type=Path, default=None)
    p.add_argument("--out-json", type=Path, default=None)
    args = p.parse_args(argv)
    cfg = load_train_config(args.config)
    evaluate(args.checkpoint, cfg,
             manifest_csv=args.manifest, out_json=args.out_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
