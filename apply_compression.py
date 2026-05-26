"""Apply Opus / AMR / radio degradation to a directory of WAVs.

Used to build the stress-test partitions referenced in section 4.6 of the
course report.
"""
from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "."))

from compression import CODECS  # noqa: E402


def _one(args):
    path, raw_dir, out_dir, codec, bitrate = args
    func = CODECS[codec]
    rel = Path(path).relative_to(raw_dir)
    target = Path(out_dir) / rel.with_suffix(".wav")
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        if codec == "radio":
            func(path, target)
        else:
            func(path, target, bitrate=bitrate)
        return None
    except Exception as exc:  # pylint: disable=broad-except
        return f"{path}: {exc!r}"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("raw_dir", type=Path)
    p.add_argument("out_dir", type=Path)
    p.add_argument("--codec", choices=CODECS.keys(), required=True)
    p.add_argument("--bitrate", default="16k")
    p.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    args = p.parse_args(argv)

    files = [Path(r) / n for r, _d, ns in os.walk(args.raw_dir)
             for n in ns if n.lower().endswith((".wav", ".flac"))]
    tasks = [(f, args.raw_dir, args.out_dir, args.codec, args.bitrate) for f in files]

    errors: list[str] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for err in tqdm(pool.map(_one, tasks), total=len(tasks),
                        desc=f"compress[{args.codec}]"):
            if err is not None:
                errors.append(err)
    if errors:
        (args.out_dir / "errors.log").write_text("\n".join(errors), encoding="utf-8")
        print(f"[compress] {len(errors)} failures -> errors.log", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
