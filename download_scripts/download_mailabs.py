"""
M-AILABS selective downloader — качає тільки потрібні мови.

M-AILABS розповсюджується як окремі zip-архіви по мовах з GitHub releases.
Цей скрипт тягне тільки en, ru, uk, pl архіви.

Залежності:
    pip install requests tqdm

Використання:
    python download_mailabs.py --out ./mailabs_data --langs en ru uk pl
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

# Прямі посилання на zip-архіви з офіційного репо
# https://github.com/imdatceleste/m-ailabs-dataset
MAILABS_URLS: dict[str, str] = {
    "en_US": "https://ics.tau-ceti.space/data/Training/stt_tts/en_US.tgz",
    "en_UK": "https://ics.tau-ceti.space/data/Training/stt_tts/en_UK.tgz",
    "ru_RU": "https://ics.tau-ceti.space/data/Training/stt_tts/ru_RU.tgz",
    "uk_UK": "https://ics.tau-ceti.space/data/Training/stt_tts/uk_UK.tgz",
    "pl_PL": "https://ics.tau-ceti.space/data/Training/stt_tts/pl_PL.tgz",
}

# Відповідність коротких кодів до ключів архівів
LANG_MAP: dict[str, list[str]] = {
    "en": ["en_US", "en_UK"],
    "ru": ["ru_RU"],
    "uk": ["uk_UK"],
    "pl": ["pl_PL"],
}

# Приблизні розміри (щоб заздалегідь знати)
APPROX_SIZE_GB: dict[str, float] = {
    "en_US": 8.5,
    "en_UK": 3.0,
    "ru_RU": 6.0,
    "uk_UK": 1.2,
    "pl_PL": 2.0,
}


def _download_file(url: str, dest: Path, resume: bool = True) -> Path:
    """Завантажити файл з підтримкою resume (якщо перервалось)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    headers = {}
    mode = "wb"
    downloaded = 0

    if resume and dest.exists():
        downloaded = dest.stat().st_size
        headers["Range"] = f"bytes={downloaded}-"
        mode = "ab"

    r = requests.get(url, headers=headers, stream=True, timeout=30)

    if r.status_code == 416:  # Range Not Satisfiable — файл вже повний
        print(f"  [skip] {dest.name} вже завантажено")
        return dest

    if r.status_code not in (200, 206):
        raise RuntimeError(f"HTTP {r.status_code} для {url}")

    total = int(r.headers.get("content-length", 0)) + downloaded
    with open(dest, mode) as fh, tqdm(
        total=total,
        initial=downloaded,
        unit="B",
        unit_scale=True,
        desc=dest.name,
    ) as bar:
        for chunk in r.iter_content(chunk_size=1 << 20):  # 1 MB chunks
            fh.write(chunk)
            bar.update(len(chunk))
    return dest


def _extract(archive: Path, out_dir: Path) -> None:
    """Розпакувати tgz або zip архів."""
    import tarfile
    out_dir.mkdir(parents=True, exist_ok=True)
    if archive.suffix in (".tgz", ".gz") or archive.name.endswith(".tar.gz"):
        with tarfile.open(archive, "r:gz") as tf:
            tf.extractall(out_dir)
    elif archive.suffix == ".zip":
        with zipfile.ZipFile(archive, "r") as zf:
            zf.extractall(out_dir)
    else:
        raise ValueError(f"Невідомий формат архіву: {archive}")


def estimate_size(langs: list[str]) -> float:
    total = 0.0
    for lang in langs:
        for key in LANG_MAP.get(lang, []):
            total += APPROX_SIZE_GB.get(key, 0)
    return total


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Завантажити M-AILABS вибірково по мовах",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Приклади:
  # Тільки оцінити розмір без скачування
  python download_mailabs.py --dry-run --langs en ru uk pl

  # Скачати і розпакувати
  python download_mailabs.py --out ./mailabs_data --langs en ru uk pl

  # Скачати без розпакування (зберегти архіви)
  python download_mailabs.py --out ./mailabs_data --langs uk --no-extract
        """,
    )
    p.add_argument("--out", type=Path, default=Path("./mailabs_data"))
    p.add_argument("--langs", nargs="+", default=["en", "ru", "uk", "pl"],
                   choices=list(LANG_MAP.keys()),
                   help="які мови качати")
    p.add_argument("--no-extract", action="store_true",
                   help="не розпаковувати архіви")
    p.add_argument("--dry-run", action="store_true",
                   help="тільки показати що буде скачано і скільки GB")
    args = p.parse_args(argv)

    # Зібрати список архівів
    to_download: list[tuple[str, str]] = []  # (key, url)
    for lang in args.langs:
        for key in LANG_MAP.get(lang, []):
            if key in MAILABS_URLS:
                to_download.append((key, MAILABS_URLS[key]))
            else:
                print(f"  ! немає URL для {key}", file=sys.stderr)

    total_gb = estimate_size(args.langs)
    print(f"\nПлан завантаження ({len(to_download)} архівів, ~{total_gb:.1f} GB):")
    for key, url in to_download:
        gb = APPROX_SIZE_GB.get(key, 0)
        print(f"  {key:12s}  ~{gb:.1f} GB  {url}")

    if args.dry_run:
        print("\n[dry-run] реального завантаження не було")
        return 0

    confirm = input(f"\nПродовжити? [y/N] ").strip().lower()
    if confirm != "y":
        print("Скасовано.")
        return 0

    archives_dir = args.out / "_archives"
    archives_dir.mkdir(parents=True, exist_ok=True)

    for key, url in to_download:
        ext = ".tgz"
        archive_path = archives_dir / f"{key}{ext}"
        print(f"\n── {key} ──")
        try:
            _download_file(url, archive_path)
        except Exception as e:
            print(f"  ! помилка завантаження {key}: {e}", file=sys.stderr)
            continue

        if not args.no_extract:
            print(f"  розпакування {archive_path.name} ...")
            try:
                _extract(archive_path, args.out)
                print(f"  -> {args.out}/{key}/")
            except Exception as e:
                print(f"  ! помилка розпакування {key}: {e}", file=sys.stderr)

    print(f"\n[done] M-AILABS файли в {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())