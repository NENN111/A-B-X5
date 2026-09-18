"""Консервативная очистка и профилирование исходных данных RetailHero."""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from uuid import uuid4

import polars as pl

from src.config.settings import Settings, get_settings
from src.data.loader import MissingRawDataError, RawDataLoader
from src.data.quality import ProfileArtifacts, profile_dataset

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class CleaningConfig:
    """Безопасные настройки очистки, не зависящие от бизнес-схемы."""

    nan_to_null: bool = True
    drop_full_duplicates: bool = True


@dataclass(frozen=True, slots=True)
class CleaningAudit:
    """Аудит изменений одной сущности."""

    entity: str
    input_rows: int
    output_rows: int
    rows_removed: int
    nan_values_replaced: int
    output_path: str


@dataclass(frozen=True, slots=True)
class Stage2Result:
    """Артефакты полного запуска Stage 2."""

    audits: tuple[CleaningAudit, ...]
    raw_profile: ProfileArtifacts
    cleaned_profile: ProfileArtifacts
    audit_path: Path


def _collect(frame: pl.LazyFrame) -> pl.DataFrame:
    """Выполнить lazy-план в streaming-режиме."""
    return frame.collect(engine="streaming")


def clean_frame(frame: pl.LazyFrame, config: CleaningConfig) -> pl.LazyFrame:
    """Применить только универсальные преобразования без догадок о колонках."""
    cleaned = frame
    if config.nan_to_null:
        float_columns = [
            column
            for column, dtype in cleaned.collect_schema().items()
            if dtype.is_float()
        ]
        if float_columns:
            cleaned = cleaned.with_columns(
                pl.col(column).fill_nan(None) for column in float_columns
            )
    if config.drop_full_duplicates:
        cleaned = cleaned.unique(maintain_order=False)
    return cleaned


def _input_statistics(frame: pl.LazyFrame) -> tuple[int, int]:
    """Посчитать число строк и NaN до очистки одним агрегирующим запросом."""
    schema = frame.collect_schema()
    nan_expressions = [
        pl.col(column).is_nan().sum()
        for column, dtype in schema.items()
        if dtype.is_float()
    ]
    total_nan = (
        pl.sum_horizontal(nan_expressions).alias("nan_count")
        if nan_expressions
        else pl.lit(0).alias("nan_count")
    )
    result = _collect(frame.select(pl.len().alias("row_count"), total_nan)).row(
        0, named=True
    )
    return int(result["row_count"]), int(result["nan_count"] or 0)


def clean_entity(
    entity: str,
    frame: pl.LazyFrame,
    output_path: Path,
    config: CleaningConfig,
) -> CleaningAudit:
    """Очистить сущность, атомарно записать Parquet и вернуть аудит."""
    input_rows, input_nans = _input_statistics(frame)
    cleaned = clean_frame(frame, config)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(f".{uuid4().hex}.tmp.parquet")
    try:
        cleaned.sink_parquet(temporary_path, mkdir=True)
        output_rows = int(
            _collect(
                pl.scan_parquet(temporary_path).select(pl.len().alias("row_count"))
            ).item()
        )
        temporary_path.replace(output_path)
    finally:
        temporary_path.unlink(missing_ok=True)

    audit = CleaningAudit(
        entity=entity,
        input_rows=input_rows,
        output_rows=output_rows,
        rows_removed=input_rows - output_rows,
        nan_values_replaced=input_nans if config.nan_to_null else 0,
        output_path=str(output_path),
    )
    LOGGER.info(
        "Очищена сущность %s: %d -> %d строк",
        entity,
        input_rows,
        output_rows,
    )
    return audit


def clean_dataset(
    frames: Mapping[str, pl.LazyFrame],
    output_dir: Path,
    config: CleaningConfig,
) -> tuple[tuple[CleaningAudit, ...], dict[str, pl.LazyFrame]]:
    """Очистить все сущности и вернуть аудиты вместе с lazy-сканами Parquet."""
    audits: list[CleaningAudit] = []
    cleaned_frames: dict[str, pl.LazyFrame] = {}
    for entity, frame in frames.items():
        output_path = output_dir / f"{entity}.parquet"
        audits.append(clean_entity(entity, frame, output_path, config))
        cleaned_frames[entity] = pl.scan_parquet(output_path)
    return tuple(audits), cleaned_frames


def run_stage2(
    settings: Settings | None = None,
    *,
    raw_dir: Path | None = None,
    output_dir: Path | None = None,
    report_dir: Path | None = None,
    config: CleaningConfig | None = None,
    exact_cardinality: bool = False,
) -> Stage2Result:
    """Выполнить профилирование raw, очистку и профилирование результата."""
    runtime = settings or get_settings()
    source_dir = raw_dir or runtime.raw_data_dir
    clean_dir = output_dir or runtime.interim_data_dir / "clean"
    quality_dir = report_dir or runtime.interim_data_dir / "quality"
    cleaning_config = config or CleaningConfig()

    frames = RawDataLoader(source_dir).scan_all()
    raw_profile = profile_dataset(
        frames,
        quality_dir / "raw",
        exact_cardinality=exact_cardinality,
    )
    audits, cleaned_frames = clean_dataset(frames, clean_dir, cleaning_config)
    cleaned_profile = profile_dataset(
        cleaned_frames,
        quality_dir / "cleaned",
        exact_cardinality=exact_cardinality,
    )
    audit_path = quality_dir / "cleaning_audit.json"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(
        json.dumps([asdict(audit) for audit in audits], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return Stage2Result(audits, raw_profile, cleaned_profile, audit_path)


def main() -> None:
    """Запустить Stage 2 из командной строки."""
    parser = argparse.ArgumentParser(
        description="Профилирование и безопасная очистка данных RetailHero."
    )
    parser.add_argument("--raw-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--report-dir", type=Path)
    parser.add_argument(
        "--keep-duplicates",
        action="store_true",
        help="Не удалять полные дубли строк.",
    )
    parser.add_argument(
        "--exact-cardinality",
        action="store_true",
        help="Использовать точный n_unique вместо экономной приближённой оценки.",
    )
    args = parser.parse_args()
    settings = get_settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        result = run_stage2(
            settings,
            raw_dir=args.raw_dir,
            output_dir=args.output_dir,
            report_dir=args.report_dir,
            config=CleaningConfig(drop_full_duplicates=not args.keep_duplicates),
            exact_cardinality=args.exact_cardinality,
        )
    except MissingRawDataError as error:
        parser.error(str(error))
    print(f"Stage 2 завершён. Аудит: {result.audit_path}")


if __name__ == "__main__":
    main()
