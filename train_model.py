"""Thin wrapper around ``spoof_phase.train.train`` — for use without pip-install."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "."))

from config import load_train_config  # noqa: E402
from train import train  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("config", type=Path)
    args = p.parse_args(argv)
    train(load_train_config(args.config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
