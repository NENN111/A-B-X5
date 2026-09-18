"""Поиск и загрузка исходных файлов RetailHero с экономным использованием памяти."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Final

import polars as pl

from src.config.settings import get_settings


LOGGER = logging.getLogger(__name__)
SUPPORTED_SUFFIXES: Final[tuple[str, ...]] = (".csv", ".gz", ".parquet")
EXPECTED_ENTITIES: Final[tuple[str, ...]] = (
    "clients",
    "products",
    "purchases",
    "uplift_train",
)


class MissingRawDataError(FileNotFoundError):
    """Ошибка отсутствующего или неполного набора исходных данных."""


class UnsupportedDataFormatError(ValueError):
    """Ошибка формата источника, для которого нет lazy-загрузчика."""


class RawDataLoader:
    """Найти исходные файлы и представить их как Polars LazyFrame.

    Файлы сопоставляются по именам сущностей, а схемы всегда читаются из
    фактических файлов. Названия колонок источника заранее не предполагаются.
    """

    def __init__(self, raw_dir: Path) -> None:
        self.raw_dir = raw_dir

    def discover(self) -> dict[str, Path]:
        """Сопоставить обязательные сущности с уникальными файлами raw-каталога."""
        if not self.raw_dir.exists():
            raise MissingRawDataError(self._missing_message(EXPECTED_ENTITIES))

        files = [path for path in self.raw_dir.iterdir() if path.is_file()]
        discovered: dict[str, Path] = {}
        ambiguous: dict[str, list[Path]] = {}

        for entity in EXPECTED_ENTITIES:
            matches = [
                path
                for path in files
                if self._is_supported(path) and entity in path.name.lower()
            ]
            if len(matches) == 1:
                discovered[entity] = matches[0]
            elif len(matches) > 1:
                ambiguous[entity] = matches

        if ambiguous:
            details = "; ".join(
                f"{entity}: {', '.join(path.name for path in paths)}"
                for entity, paths in ambiguous.items()
            )
            raise MissingRawDataError(
                "Найдено несколько файлов-кандидатов. "
                "Оставьте один файл для каждой сущности. "
                f"{details}"
            )

        missing = tuple(entity for entity in EXPECTED_ENTITIES if entity not in discovered)
        if missing:
            raise MissingRawDataError(self._missing_message(missing))
        return discovered

    def scan(self, path: Path) -> pl.LazyFrame:
        """Создать lazy scan без материализации всего набора данных."""
        suffixes = [suffix.lower() for suffix in path.suffixes]
        if suffixes[-1:] == [".parquet"]:
            return pl.scan_parquet(path)
        if suffixes[-1:] == [".csv"] or suffixes[-2:] == [".csv", ".gz"]:
            return pl.scan_csv(path, try_parse_dates=True, infer_schema_length=10_000)
        raise UnsupportedDataFormatError(
            f"Неподдерживаемый формат источника для {path.name}. "
            "Используйте CSV, CSV.GZ или Parquet."
        )

    def scan_all(self) -> dict[str, pl.LazyFrame]:
        """Найти и лениво открыть все обязательные сущности источника."""
        paths = self.discover()
        return {entity: self.scan(path) for entity, path in paths.items()}

    def inspect_schemas(self) -> dict[str, dict[str, str]]:
        """Прочитать фактические схемы без загрузки таблиц целиком в память."""
        schemas: dict[str, dict[str, str]] = {}
        for entity, frame in self.scan_all().items():
            schema = frame.collect_schema()
            schemas[entity] = {name: str(dtype) for name, dtype in schema.items()}
            LOGGER.info("Проверена сущность %s: %d колонок", entity, len(schema))
        return schemas

    @staticmethod
    def _is_supported(path: Path) -> bool:
        name = path.name.lower()
        return name.endswith((".csv", ".csv.gz", ".parquet"))

    def _missing_message(self, missing: tuple[str, ...]) -> str:
        expected = ", ".join(f"{name}.csv[.gz] или {name}.parquet" for name in missing)
        return (
            f"Исходные данные RetailHero отсутствуют в '{self.raw_dir}'. "
            f"Ожидаются файлы: {expected}. "
            "Инструкции по получению и размещению находятся в data/raw/README.md."
        )


def main() -> None:
    """Проверить схемы источников из командной строки."""
    parser = argparse.ArgumentParser(description="Проверить схемы исходных данных RetailHero.")
    parser.add_argument("--raw-dir", type=Path, help="Переопределить настроенный raw-каталог.")
    args = parser.parse_args()
    settings = get_settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    loader = RawDataLoader(args.raw_dir or settings.raw_data_dir)
    for entity, schema in loader.inspect_schemas().items():
        print(f"\n[{entity}]")
        for column, dtype in schema.items():
            print(f"{column}: {dtype}")


if __name__ == "__main__":
    main()
