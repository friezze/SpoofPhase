"""Підготовка даних: маніфести -> фічі -> (опційно) компресія.

Файл лежить поруч з папкою data/ і рештою скриптів:
    code/
    ├── prepare_data.py   <- цей файл
    ├── compression.py
    ├── features.py
    ├── manifests.py
    └── data/
        ├── mlaad/        fake/<lang>/<model>/*.wav
        ├── mailabs/      en_US/.../wavs/*.wav
        ├── replaydf/     wav/<UID>/spoof|benign/<lang>/*.wav
        └── manifests/
"""
from __future__ import annotations

import csv
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from compression import CODECS
from features import get_audio_features, stack_features
from manifests import merge_manifests


# =====================================================================
# КРОК 1 -- МАНІФЕСТИ
# =====================================================================

def _write_csv(path: Path, rows: Iterable[Sequence[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["path", "label", "dataset"])
        for row in rows:
            w.writerow(row)


def build_mlaad_manifest(mlaad_root: Path, mailabs_root: Path, out_csv: Path) -> int:
    """
    MLAAD (spoof) + M-AILABS (bona-fide) в один маніфест.
    Шляхи мають префікс mlaad/ або mailabs/ -- audio_root при
    step_features має бути ./data щоб резолвились правильно.
    """
    rows: List[List[str]] = []

    fake_dir = mlaad_root / "fake"
    src = fake_dir if fake_dir.exists() else mlaad_root
    for wav in sorted(src.glob("**/*.wav")):
        rel = str(wav.relative_to(mlaad_root))
        rows.append([f"mlaad/{rel}", "spoof", "mlaad"])

    if mailabs_root is not None and mailabs_root.exists():
        for wav in sorted(mailabs_root.glob("**/*.wav")):
            rel = str(wav.relative_to(mailabs_root))
            rows.append([f"mailabs/{rel}", "bona-fide", "mailabs"])

    _write_csv(out_csv, rows)
    return len(rows)


def build_inthewild_manifest(inthewild_root: Path, out_csv: Path) -> int:
    """
    In-the-Wild -- всі файли spoof (тільки синтетичні голоси).
    Структура пласка: inthewild/*.wav
    Використовується як eval/test set + для стрес-тесту компресією.
    """
    rows: List[List[str]] = []
    for wav in sorted(inthewild_root.glob("*.wav")):
        rows.append([f"inthewild/{wav.name}", "spoof", "inthewild"])
    if not rows:
        raise FileNotFoundError(f"WAV файлів не знайдено в {inthewild_root}")
    _write_csv(out_csv, rows)
    return len(rows)


def build_replaydf_manifest(replaydf_root: Path, out_csv: Path,
                             target_langs: set | None = None) -> int:
    rows: List[List[str]] = []
    wav_dir = replaydf_root / "wav"
    if not wav_dir.exists():
        raise FileNotFoundError(f"ReplayDF wav/ не знайдено: {wav_dir}")

    for uid_dir in sorted(wav_dir.iterdir()):
        if not uid_dir.is_dir():
            continue
        meta_path = uid_dir / "meta.csv"
        if not meta_path.exists():
            continue
        with meta_path.open(encoding="utf-8") as f:
            # роздільник — pipe
            reader = csv.DictReader(f, delimiter="|")
            for row in reader:
                lang = row.get("language", "").strip().lower()
                if target_langs and lang not in target_langs:
                    continue
                recorded = row.get("recorded_file", "").strip()
                label = row.get("label", "").strip()
                if not recorded or label not in ("spoof", "bona-fide"):
                    continue
                wav_path = replaydf_root / recorded
                if not wav_path.exists():
                    continue
                rows.append([recorded, label, f"replaydf_{lang}"])

    _write_csv(out_csv, rows)
    return len(rows)

def step_manifests(
    mlaad_root: Path,
    mailabs_root: Path,
    inthewild_root: Path,
    replaydf_root: Path,
    out_dir: Path,
    replaydf_langs: set | None = None,
) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    produced: List[Path] = []

    def _try(fn, label, *args):
        try:
            n = fn(*args)
            print(f"  + {label}: {n} записів")
            return True
        except FileNotFoundError as e:
            print(f"  ! {label}: пропущено ({e})", file=sys.stderr)
            return False

    out = out_dir / "mlaad.csv"
    if _try(build_mlaad_manifest, "mlaad+mailabs", mlaad_root, mailabs_root, out):
        produced.append(out)

    out = out_dir / "inthewild.csv"
    if _try(build_inthewild_manifest, "inthewild", inthewild_root, out):
        produced.append(out)

    out = out_dir / "replaydf.csv"
    if _try(build_replaydf_manifest, "replaydf", replaydf_root, out, replaydf_langs):
        produced.append(out)

    # train.csv = тільки mlaad+mailabs; replaydf -- тільки eval
    train_src = [p for p in produced if "mlaad" in p.name]
    if train_src:
        merged = out_dir / "train.csv"
        n = merge_manifests(train_src, merged)
        print(f"  + train.csv: {n} записів")
        produced.append(merged)

    print(f"[manifests] готово, {len(produced)} файлів у {out_dir}")
    return produced


# =====================================================================
# КРОК 2 -- ФІЧІ (mel + group delay -> NPZ)
# =====================================================================

def _read_manifest(path: Path) -> list[tuple[str, str, str]]:
    with path.open("r", encoding="utf-8") as fh:
        return [(r["path"], r["label"], r.get("dataset", ""))
                for r in csv.DictReader(fh)]


def _process_one(args):
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
    except Exception as exc:
        return (rel, label, dataset, 0, 0, repr(exc))


def step_features(
    manifests: List[Path],
    audio_root: Path,
    features_root: Path,
    workers: int = os.cpu_count() or 4,
) -> None:
    items = []
    for mf in manifests:
        for rel, label, dataset in _read_manifest(mf):
            items.append((rel, str(audio_root), str(features_root), label, dataset))

    features_root.mkdir(parents=True, exist_ok=True)
    out_manifest = features_root / "manifest.csv"
    errors: list[str] = []

    with out_manifest.open("w", newline="", encoding="utf-8") as fh, \
         ProcessPoolExecutor(max_workers=workers) as pool:
        writer = csv.writer(fh)
        writer.writerow(["path", "label", "dataset", "n_mels", "frames"])
        futures = [pool.submit(_process_one, it) for it in items]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="features"):
            rel, label, dataset, n_mels, frames, err = fut.result()
            if err:
                errors.append(f"{rel}\t{err}")
                continue
            writer.writerow([rel, label, dataset, n_mels, frames])

    if errors:
        (features_root / "errors.log").write_text("\n".join(errors), encoding="utf-8")
        print(f"[features] {len(errors)} помилок -> errors.log", file=sys.stderr)
    print(f"[features] {len(items) - len(errors)} NPZ -> {out_manifest}")


# =====================================================================
# КРОК 3 -- КОМПРЕСІЯ (стрес-тест, опційно)
# =====================================================================

def _compress_one(args):
    path, raw_dir, out_dir, codec, bitrate = args
    func = CODECS[codec]
    rel = Path(path).relative_to(raw_dir)
    target = Path(out_dir) / rel.with_suffix(".wav")
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        func(path, target) if codec == "radio" else func(path, target, bitrate=bitrate)
        return None
    except Exception as exc:
        return f"{path}: {exc!r}"


def step_compress(
    raw_dir: Path,
    out_dir: Path,
    codec: str = "opus",    # "opus" | "amr" | "radio"
    bitrate: str = "16k",
    workers: int = os.cpu_count() or 4,
) -> None:
    files = [Path(r) / n for r, _d, ns in os.walk(raw_dir)
             for n in ns if n.lower().endswith((".wav", ".flac"))]
    tasks = [(f, raw_dir, out_dir, codec, bitrate) for f in files]
    errors: list[str] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for err in tqdm(pool.map(_compress_one, tasks), total=len(tasks),
                        desc=f"compress[{codec}]"):
            if err:
                errors.append(err)
    if errors:
        (out_dir / "errors.log").write_text("\n".join(errors), encoding="utf-8")
        print(f"[compress] {len(errors)} помилок -> errors.log", file=sys.stderr)
    print(f"[compress] готово -> {out_dir}")


# =====================================================================
# ТОЧКА ВХОДУ
# =====================================================================

if __name__ == "__main__":

    # -- PATHS ---------------------------------------------------------------
    MLAAD_ROOT    = Path("./data/mlaad")       # fake/<lang>/<model>/*.wav
    MAILABS_ROOT  = Path("./data/mailabs")     # en_US/.../wavs/*.wav
    INTHEWILD_ROOT = Path("./data/inthewild")  # пласка папка з WAV (всі spoof)
    REPLAYDF_ROOT = Path("./data/replaydf")    # wav/<UID>/spoof|benign/<lang>/*.wav

    MANIFESTS_DIR = Path("./data/manifests")   # куди писати CSV маніфести
    FEATURES_DIR  = Path("./data/features")    # куди писати NPZ фічі

    # шляхи в mlaad.csv мають префікс mlaad/ або mailabs/
    # тому audio_root = ./data щоб ./data/mlaad/... резолвився
    AUDIO_ROOT    = Path("./data")

    COMPRESS_IN   = Path("./data/mlaad")       # вхід для стрес-тесту
    COMPRESS_OUT  = Path("./data/mlaad_opus")  # вихід стрес-тесту
    # ------------------------------------------------------------------------

    # КРОК 1 -- побудувати CSV маніфести
    manifests = step_manifests(
        mlaad_root=MLAAD_ROOT,
        mailabs_root=MAILABS_ROOT,
        inthewild_root=INTHEWILD_ROOT,
        replaydf_root=REPLAYDF_ROOT,
        out_dir=MANIFESTS_DIR,
        replaydf_langs={"en", "pl"},  # uk/ru в ReplayDF нема
    )

    # КРОК 2 -- витягти mel+GD фічі і зберегти як NPZ
    # (багато CPU і місця; запускай після кроку 1)
    # step_features(
    #     manifests=[MANIFESTS_DIR / "train.csv"],
    #     audio_root=AUDIO_ROOT,
    #     features_root=FEATURES_DIR / "train",
    # )

    # replaydf eval
    step_features(
        manifests=[MANIFESTS_DIR / "replaydf.csv"],
        audio_root=REPLAYDF_ROOT,
        features_root=FEATURES_DIR / "replaydf",
    )

    # inthewild eval
    # step_features(
    #     manifests=[MANIFESTS_DIR / "inthewild.csv"],
    #     audio_root=INTHEWILD_ROOT.parent,  # бо шляхи в csv мають префікс inthewild/
    #     features_root=FEATURES_DIR / "inthewild",
    # )

    # КРОК 3 -- стрес-тест компресією (опційно)
    # step_compress(
    #     raw_dir=COMPRESS_IN,
    #     out_dir=COMPRESS_OUT,
    #     codec="opus",   # або "amr", "radio"
    # )