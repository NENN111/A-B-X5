"""Скачать публичный датасет RetailHero с проверкой целостности файлов."""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class DatasetFile:
    name: str
    size: int

    @property
    def url(self) -> str:
        base = "https://huggingface.co/datasets/pytorch-lifestream/retailhero-uplift"
        return f"{base}/resolve/main/data/{self.name}?download=true"


FILES = (
    DatasetFile("clients.csv.gz", 7_634_884),
    DatasetFile("products.csv.gz", 1_054_470),
    DatasetFile("purchases.csv.gz", 608_830_890),
    DatasetFile("uplift_train.csv.gz", 1_182_429),
)


def _download(item: DatasetFile, output_dir: Path, *, force: bool = False) -> Path:
    destination = output_dir / item.name
    if destination.exists() and destination.stat().st_size == item.size and not force:
        print(f"[готово] {item.name}")
        return destination

    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(item.url, headers={"User-Agent": "retail-ab-dashboard/1.0"})
    digest = hashlib.sha256()
    downloaded = 0
    try:
        with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as target:
            while block := response.read(1024 * 1024):
                target.write(block)
                digest.update(block)
                downloaded += len(block)
                if downloaded % (50 * 1024 * 1024) < len(block):
                    print(f"  {item.name}: {downloaded / 1024**2:.0f} MiB")
        if downloaded != item.size:
            raise RuntimeError(
                f"Размер {item.name} отличается от ожидаемого: {downloaded} != {item.size}"
            )
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"[скачан] {item.name}: {downloaded / 1024**2:.1f} MiB, sha256={digest.hexdigest()[:12]}")
    return destination


def download_dataset(output_dir: Path, *, force: bool = False) -> tuple[Path, ...]:
    """Скачать все обязательные файлы, безопасно продолжая повторный запуск."""
    return tuple(_download(item, output_dir, force=force) for item in FILES)


def main() -> None:
    parser = argparse.ArgumentParser(description="Скачать RetailHero Uplift.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    try:
        download_dataset(args.output_dir, force=args.force)
    except (OSError, RuntimeError) as error:
        print(f"Ошибка загрузки: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
