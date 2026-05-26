"""
MLAAD selective downloader — два режими:

  1. --meta-only   : тягне лише meta.csv по всіх мовах, будує зведену таблицю
  2. --audio       : після перегляду мета тягне WAV тільки для потрібних мов

Потрібен токен HuggingFace (датасет потребує реєстрації):
    huggingface-cli login
або передати через HF_TOKEN env-змінну.

Залежності:
    pip install huggingface_hub pandas tqdm
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from tqdm import tqdm

REPO_ID = "mueller91/MLAAD"
REPO_TYPE = "dataset"

# Назви папок мов у репозиторії (як вони є на HF)
TARGET_LANGS = {
    "en": "english",
    "ru": "russian",
    "uk": "ukrainian",
    "pl": "polish",
}


# ── крок 1: завантажити тільки meta.csv ──────────────────────────────────────

def fetch_all_meta(out_dir: Path, token: str | None) -> Path:
    """
    Тягне всі meta.csv файли без аудіо.
    Повертає шлях до зведеного CSV.
    """
    api = HfApi()
    all_files = api.list_repo_files(REPO_ID, repo_type=REPO_TYPE, token=token)

    meta_files = [f for f in all_files if f.endswith("meta.csv")]
    print(f"[meta] знайдено {len(meta_files)} meta.csv файлів")

    out_dir.mkdir(parents=True, exist_ok=True)
    frames = []

    for remote_path in tqdm(meta_files, desc="завантаження meta.csv"):
        local = hf_hub_download(
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            filename=remote_path,
            local_dir=str(out_dir / "meta_raw"),
            token=token,
        )
        try:
            df = pd.read_csv(local, sep="|")
            df["_meta_source"] = remote_path
            frames.append(df)
        except Exception as e:
            print(f"  ! не вдалося прочитати {remote_path}: {e}", file=sys.stderr)

    if not frames:
        print("[meta] жодного файлу не завантажено — перевір токен/доступ")
        return out_dir

    merged = pd.concat(frames, ignore_index=True)
    summary_path = out_dir / "mlaad_meta_all.csv"
    merged.to_csv(summary_path, index=False)
    print(f"\n[meta] збережено {len(merged)} рядків -> {summary_path}")

    # Показати статистику по мовах
    if "language" in merged.columns:
        print("\n── мови в датасеті ──")
        lang_stats = (
            merged.groupby("language")
            .agg(files=("path", "count"), hours=("duration", "sum"))
            .sort_values("files", ascending=False)
        )
        lang_stats["hours"] = (lang_stats["hours"] / 3600).round(1)
        print(lang_stats.to_string())

    return summary_path


# ── крок 2: завантажити аудіо тільки для потрібних мов ───────────────────────

def download_audio_for_langs(
    meta_csv: Path,
    out_dir: Path,
    langs: list[str],
    token: str | None,
) -> None:
    """
    Читає зведений meta.csv, фільтрує потрібні мови,
    завантажує тільки ці WAV файли з HF.
    """
    df = pd.read_csv(meta_csv)

    if "language" not in df.columns:
        print("[audio] колонка 'language' не знайдена в meta.csv", file=sys.stderr)
        return

    filtered = df[df["language"].isin(langs)]
    print(f"[audio] {len(filtered)} файлів для мов {langs}")

    if filtered.empty:
        print("[audio] нічого не знайдено — перевір назви мов у --langs")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    errors = []

    for _, row in tqdm(filtered.iterrows(), total=len(filtered), desc="download audio"):
        remote_path = row["path"]          # напр. fake/en/vits/audio_0001.wav
        target = out_dir / remote_path
        if target.exists():
            continue
        try:
            hf_hub_download(
                repo_id=REPO_ID,
                repo_type=REPO_TYPE,
                filename=remote_path,
                local_dir=str(out_dir),
                token=token,
            )
        except Exception as e:
            errors.append(f"{remote_path}: {e}")

    if errors:
        err_log = out_dir / "audio_errors.log"
        err_log.write_text("\n".join(errors), encoding="utf-8")
        print(f"[audio] {len(errors)} помилок -> {err_log}")
    else:
        print(f"[audio] готово, файли в {out_dir}")


# ── альтернатива: snapshot лише потрібних папок ──────────────────────────────

def snapshot_langs(out_dir: Path, langs: list[str], token: str | None) -> None:
    """
    Швидший варіант через snapshot_download з allow_patterns.
    Скачує ТІЛЬКИ папки потрібних мов (fake/en/**, fake/ru/**, etc.)
    + всі meta.csv.
    """
    patterns = []
    for lang in langs:
        patterns.append(f"fake/{lang}/**")   # всі файли мови
    patterns.append("**/meta.csv")           # всі метадані

    print(f"[snapshot] скачую: {patterns}")
    snapshot_download(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        local_dir=str(out_dir),
        allow_patterns=patterns,
        token=token,
    )
    print(f"[snapshot] готово -> {out_dir}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Завантажити MLAAD вибірково по мовах",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--out", type=Path, default=Path("./mlaad_data"),
                   help="директорія для збереження")
    p.add_argument("--token", default=os.environ.get("HF_TOKEN"),
                   help="HuggingFace токен (або встанови HF_TOKEN)")
    p.add_argument("--langs", nargs="+",
                   default=["en", "ru", "uk", "pl"],
                   help="коди мов (en ru uk pl ...)")

    sub = p.add_subparsers(dest="cmd")

    # підкоманда 1: тільки мета
    sub.add_parser("meta", help="завантажити тільки meta.csv (без аудіо)")

    # підкоманда 2: аудіо по meta.csv
    audio_p = sub.add_parser("audio", help="завантажити аудіо після перегляду мета")
    audio_p.add_argument("--meta-csv", type=Path, required=True,
                         help="шлях до зведеного mlaad_meta_all.csv")

    # підкоманда 3: snapshot (найшвидший варіант одразу з аудіо)
    sub.add_parser("snapshot", help="snapshot_download тільки потрібних мов")

    args = p.parse_args(argv)

    if args.token is None:
        print(
            "УВАГА: токен не знайдено. Датасет потребує реєстрації на HF.\n"
            "  huggingface-cli login\n"
            "або --token <твій_токен>",
            file=sys.stderr,
        )

    if args.cmd == "meta":
        fetch_all_meta(args.out, args.token)

    elif args.cmd == "audio":
        download_audio_for_langs(args.meta_csv, args.out, args.langs, args.token)

    elif args.cmd == "snapshot":
        snapshot_langs(args.out, args.langs, args.token)

    else:
        p.print_help()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())