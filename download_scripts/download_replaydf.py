"""
ReplayDF downloader — завантажує датасет і фільтрує по мовах через meta.csv.

ReplayDF структура:
    wav/<UID>/spoof/  — re-recorded synthetic speech (label: spoof)
    wav/<UID>/benign/ — re-recorded bona-fide speech (label: bona-fide)
    wav/<UID>/meta.csv — метадані запису (мова, модель, setup, etc.)

Оскільки датасет відносно невеликий (~кілька GB), є два варіанти:
    1. --meta-only  : завантажити тільки meta.csv по всіх UID, подивитись мови
    2. --download   : завантажити все + опційно відфільтрувати по мовах

Залежності:
    pip install huggingface_hub pandas tqdm

Ліцензія: CC-BY-NC 4.0 (тільки некомерційне використання)
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from tqdm import tqdm

REPO_ID = "mueller91/ReplayDF"
REPO_TYPE = "dataset"

# Мови які нас цікавлять (як вони записані в meta.csv ReplayDF)
# ReplayDF базується на MLAAD v5 — ті ж назви мов
TARGET_LANGS = {"en", "ru", "uk", "pl",
                "english", "russian", "ukrainian", "polish"}


# ── крок 1: тільки метадані ───────────────────────────────────────────────────

def fetch_meta_only(out_dir: Path, token: str | None) -> Path:
    """Завантажити всі meta.csv без аудіо, показати статистику по мовах."""
    api = HfApi()
    all_files = list(api.list_repo_files(REPO_ID, repo_type=REPO_TYPE, token=token))

    meta_files = [f for f in all_files if f.endswith("meta.csv")]
    print(f"[meta] знайдено {len(meta_files)} meta.csv (по кожному UID)")

    out_dir.mkdir(parents=True, exist_ok=True)
    frames = []

    for remote_path in tqdm(meta_files, desc="завантаження meta.csv"):
        local = hf_hub_download(
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            filename=remote_path,
            local_dir=str(out_dir / "_meta_raw"),
            token=token,
        )
        try:
            df = pd.read_csv(local)
            df["_uid_path"] = remote_path          # wav/<UID>/meta.csv
            df["_uid"] = Path(remote_path).parent.name
            frames.append(df)
        except Exception as e:
            print(f"  ! {remote_path}: {e}", file=sys.stderr)

    if not frames:
        print("[meta] нічого не завантажено — перевір токен/доступ")
        return out_dir

    merged = pd.concat(frames, ignore_index=True)
    summary_path = out_dir / "replaydf_meta_all.csv"
    merged.to_csv(summary_path, index=False)
    print(f"\n[meta] {len(merged)} рядків -> {summary_path}")
    print(f"\n── колонки: {list(merged.columns)} ──")

    # Статистика по мовах (назва колонки може відрізнятись)
    lang_col = next((c for c in merged.columns
                     if "lang" in c.lower()), None)
    if lang_col:
        print(f"\n── мови (колонка '{lang_col}') ──")
        print(merged[lang_col].value_counts().to_string())
    else:
        print("\n[!] колонка мови не знайдена — перевір вручну replaydf_meta_all.csv")

    return summary_path


# ── крок 2a: snapshot всього датасету ────────────────────────────────────────

def download_all(out_dir: Path, token: str | None) -> None:
    """Завантажити весь ReplayDF (рекомендовано — датасет невеликий)."""
    print(f"[download] snapshot ReplayDF -> {out_dir}")
    snapshot_download(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        local_dir=str(out_dir),
        token=token,
    )
    print(f"[download] готово -> {out_dir}")


# ── крок 2b: завантажити і залишити тільки потрібні мови ─────────────────────

def download_filtered(
    out_dir: Path,
    langs: list[str],
    token: str | None,
    meta_csv: Path | None = None,
) -> None:
    """
    Варіант для економії місця:
    1. Завантажує все
    2. Читає meta.csv кожного UID
    3. Видаляє WAV файли UID що не містять потрібних мов
    """
    langs_set = {l.lower() for l in langs}

    # Спочатку тягнемо всі meta.csv
    if meta_csv is None or not meta_csv.exists():
        meta_csv = fetch_meta_only(out_dir / "_meta", token)

    df = pd.read_csv(meta_csv)
    lang_col = next((c for c in df.columns if "lang" in c.lower()), None)

    if lang_col is None:
        print("[filter] не знайдено колонку мови — завантажую все без фільтрації")
        download_all(out_dir, token)
        return

    # Знайти UID які містять потрібні мови
    mask = df[lang_col].str.lower().isin(langs_set)
    good_uids = set(df[mask]["_uid"].unique()) if "_uid" in df.columns else None

    print(f"[filter] мови {langs} -> {len(good_uids) if good_uids else '?'} UID")

    # Качаємо з allow_patterns тільки потрібні UID
    if good_uids:
        patterns = []
        for uid in good_uids:
            patterns.append(f"wav/{uid}/**")
        patterns.append("**/meta.csv")
        patterns.append("mos/**")

        print(f"[download] завантажую {len(good_uids)} UID ...")
        snapshot_download(
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            local_dir=str(out_dir),
            allow_patterns=patterns,
            token=token,
        )
    else:
        download_all(out_dir, token)

    print(f"[download] готово -> {out_dir}")


# ── крок 3: побудувати маніфест ──────────────────────────────────────────────

def build_manifest(data_dir: Path, out_csv: Path, langs: list[str] | None = None) -> int:
    """
    Обходить wav/<UID>/spoof/ і wav/<UID>/benign/,
    читає meta.csv, будує manifest.csv сумісний з твоїм NpzFeatureDataset.
    """
    import csv

    wav_dir = data_dir / "wav"
    if not wav_dir.exists():
        print(f"[manifest] не знайдено {wav_dir}", file=sys.stderr)
        return 0

    langs_set = {l.lower() for l in langs} if langs else None
    rows = []

    for uid_dir in sorted(wav_dir.iterdir()):
        if not uid_dir.is_dir():
            continue

        # Читаємо meta.csv UID щоб знати мову
        meta_path = uid_dir / "meta.csv"
        uid_lang = None
        if meta_path.exists():
            try:
                m = pd.read_csv(meta_path)
                lang_col = next((c for c in m.columns if "lang" in c.lower()), None)
                if lang_col and not m.empty:
                    uid_lang = str(m[lang_col].iloc[0]).lower()
            except Exception:
                pass

        if langs_set and uid_lang and uid_lang not in langs_set:
            continue

        for label, subdir in [("spoof", "spoof"), ("bona-fide", "benign")]:
            sub = uid_dir / subdir
            if not sub.exists():
                continue
            for wav in sorted(sub.glob("*.wav")):
                rel = wav.relative_to(data_dir)
                rows.append([str(rel), label, f"replaydf_{uid_lang or uid_dir.name}"])

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["path", "label", "dataset"])
        for row in rows:
            w.writerow(row)

    print(f"[manifest] {len(rows)} записів -> {out_csv}")
    return len(rows)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Завантажити ReplayDF вибірково по мовах",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Приклади:
  # 1. Подивитись метадані (без аудіо)
  python download_replaydf.py --out ./replaydf meta

  # 2. Скачати весь датасет (він невеликий)
  python download_replaydf.py --out ./replaydf download

  # 3. Скачати тільки потрібні мови
  python download_replaydf.py --out ./replaydf --langs en ru uk pl download-filtered

  # 4. Побудувати маніфест після скачування
  python download_replaydf.py --out ./replaydf --langs en ru uk pl manifest
        """,
    )
    p.add_argument("--out", type=Path, default=Path("./replaydf_data"))
    p.add_argument("--token", default=os.environ.get("HF_TOKEN"),
                   help="HuggingFace токен (або HF_TOKEN env)")
    p.add_argument("--langs", nargs="+", default=["en", "ru", "uk", "pl"],
                   help="коди мов для фільтрації")
    p.add_argument("--meta-csv", type=Path, default=None,
                   help="вже завантажений replaydf_meta_all.csv (опційно)")

    sub = p.add_subparsers(dest="cmd")
    sub.add_parser("meta",            help="тільки meta.csv, без аудіо")
    sub.add_parser("download",        help="скачати весь датасет")
    sub.add_parser("download-filtered", help="скачати тільки потрібні мови")
    sub.add_parser("manifest",        help="побудувати manifest.csv з вже скачаних файлів")

    args = p.parse_args(argv)

    # токен None — huggingface_hub читає з ~/.cache/huggingface/token автоматично

    if args.cmd == "meta":
        fetch_meta_only(args.out, args.token)

    elif args.cmd == "download":
        download_all(args.out, args.token)

    elif args.cmd == "download-filtered":
        download_filtered(args.out, args.langs, args.token, args.meta_csv)

    elif args.cmd == "manifest":
        out_csv = args.out / "replaydf_manifest.csv"
        build_manifest(args.out, out_csv, args.langs)

    else:
        p.print_help()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())