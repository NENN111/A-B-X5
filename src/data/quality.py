"""Профилирование качества данных без предположений о схеме источника."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import polars as pl

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ProfileArtifacts:
    """Пути к сформированным артефактам профилирования."""

    overview_path: Path
    columns_path: Path
    numeric_path: Path
    categorical_path: Path
    schema_path: Path


def _collect(frame: pl.LazyFrame) -> pl.DataFrame:
    """Выполнить lazy-план в streaming-режиме."""
    return frame.collect(engine="streaming")


def profile_entity(
    entity: str,
    frame: pl.LazyFrame,
    *,
    exact_cardinality: bool = False,
) -> tuple[dict[str, object], pl.DataFrame]:
    """Рассчитать профиль таблицы и её колонок за один агрегирующий проход."""
    schema = frame.collect_schema()
    expressions: list[pl.Expr] = [pl.len().alias("__row_count")]
    expressions.append(
        (pl.len() - pl.struct(pl.all()).n_unique()).alias("__duplicate_rows")
    )

    for index, (column, dtype) in enumerate(schema.items()):
        cardinality = (
            pl.col(column).n_unique()
            if exact_cardinality
            else pl.col(column).approx_n_unique()
        )
        expressions.extend(
            [
                pl.col(column).null_count().alias(f"__null_{index}"),
                cardinality.alias(f"__unique_{index}"),
            ]
        )
        nan_expression = (
            pl.col(column).is_nan().sum() if dtype.is_float() else pl.lit(0)
        )
        expressions.append(nan_expression.alias(f"__nan_{index}"))

    aggregate = _collect(frame.select(expressions)).row(0, named=True)
    row_count = int(aggregate["__row_count"])
    duplicate_rows = int(aggregate["__duplicate_rows"] or 0)
    column_records: list[dict[str, object]] = []

    for index, (column, dtype) in enumerate(schema.items()):
        null_count = int(aggregate[f"__null_{index}"] or 0)
        nan_count = int(aggregate[f"__nan_{index}"] or 0)
        unique_count = int(aggregate[f"__unique_{index}"] or 0)
        column_records.append(
            {
                "entity": entity,
                "column": column,
                "dtype": str(dtype),
                "row_count": row_count,
                "null_count": null_count,
                "null_rate": null_count / row_count if row_count else 0.0,
                "nan_count": nan_count,
                "unique_count": unique_count,
                "unique_rate": unique_count / row_count if row_count else 0.0,
                "cardinality_is_exact": exact_cardinality,
            }
        )

    overview = {
        "entity": entity,
        "row_count": row_count,
        "column_count": len(schema),
        "duplicate_rows": duplicate_rows,
        "duplicate_rate": duplicate_rows / row_count if row_count else 0.0,
        "total_nulls": sum(record["null_count"] for record in column_records),
        "total_nans": sum(record["nan_count"] for record in column_records),
    }
    return overview, pl.DataFrame(column_records)


def numeric_summary(entity: str, frame: pl.LazyFrame) -> pl.DataFrame:
    """Рассчитать описательные статистики числовых колонок."""
    schema = frame.collect_schema()
    numeric_columns = [column for column, dtype in schema.items() if dtype.is_numeric()]
    if not numeric_columns:
        return pl.DataFrame(
            schema={
                "entity": pl.String,
                "column": pl.String,
                "mean": pl.Float64,
                "std": pl.Float64,
                "min": pl.Float64,
                "q25": pl.Float64,
                "median": pl.Float64,
                "q75": pl.Float64,
                "max": pl.Float64,
            }
        )

    expressions: list[pl.Expr] = []
    for index, column in enumerate(numeric_columns):
        value = pl.col(column).cast(pl.Float64)
        if schema[column].is_float():
            value = value.fill_nan(None)
        expressions.extend(
            [
                value.mean().alias(f"__mean_{index}"),
                value.std(ddof=1).alias(f"__std_{index}"),
                value.min().alias(f"__min_{index}"),
                value.quantile(0.25).alias(f"__q25_{index}"),
                value.median().alias(f"__median_{index}"),
                value.quantile(0.75).alias(f"__q75_{index}"),
                value.max().alias(f"__max_{index}"),
            ]
        )
    values = _collect(frame.select(expressions)).row(0, named=True)
    records = []
    for index, column in enumerate(numeric_columns):
        records.append(
            {
                "entity": entity,
                "column": column,
                "mean": values[f"__mean_{index}"],
                "std": values[f"__std_{index}"],
                "min": values[f"__min_{index}"],
                "q25": values[f"__q25_{index}"],
                "median": values[f"__median_{index}"],
                "q75": values[f"__q75_{index}"],
                "max": values[f"__max_{index}"],
            }
        )
    return pl.DataFrame(records)


def categorical_top_values(
    entity: str,
    frame: pl.LazyFrame,
    column_profile: pl.DataFrame,
    *,
    max_cardinality: int = 50,
    top_k: int = 10,
) -> pl.DataFrame:
    """Получить частые значения низкокардинальных строковых и булевых колонок."""
    schema = frame.collect_schema()
    unique_by_column = dict(column_profile.select("column", "unique_count").iter_rows())
    candidates = [
        column
        for column, dtype in schema.items()
        if (dtype == pl.String or dtype == pl.Boolean)
        and unique_by_column[column] <= max_cardinality
    ]
    frames: list[pl.DataFrame] = []
    for column in candidates:
        values = _collect(
            frame.group_by(column)
            .agg(pl.len().alias("count"))
            .sort("count", descending=True)
            .head(top_k)
        )
        values = values.with_columns(
            pl.lit(entity).alias("entity"),
            pl.lit(column).alias("column"),
            pl.col(column).cast(pl.String).alias("value"),
        ).select("entity", "column", "value", "count")
        frames.append(values)

    if frames:
        return pl.concat(frames, how="vertical")
    return pl.DataFrame(
        schema={
            "entity": pl.String,
            "column": pl.String,
            "value": pl.String,
            "count": pl.UInt32,
        }
    )


def profile_dataset(
    frames: Mapping[str, pl.LazyFrame],
    output_dir: Path,
    *,
    exact_cardinality: bool = False,
    max_categorical_cardinality: int = 50,
    top_k: int = 10,
) -> ProfileArtifacts:
    """Сформировать табличные EDA-отчёты и JSON-инвентаризацию схем."""
    output_dir.mkdir(parents=True, exist_ok=True)
    overviews: list[dict[str, object]] = []
    column_frames: list[pl.DataFrame] = []
    numeric_frames: list[pl.DataFrame] = []
    categorical_frames: list[pl.DataFrame] = []
    schemas: dict[str, dict[str, str]] = {}

    for entity, frame in frames.items():
        overview, columns = profile_entity(
            entity,
            frame,
            exact_cardinality=exact_cardinality,
        )
        overviews.append(overview)
        column_frames.append(columns)
        numeric_frames.append(numeric_summary(entity, frame))
        categorical_frames.append(
            categorical_top_values(
                entity,
                frame,
                columns,
                max_cardinality=max_categorical_cardinality,
                top_k=top_k,
            )
        )
        schemas[entity] = {
            column: str(dtype) for column, dtype in frame.collect_schema().items()
        }
        LOGGER.info("Профиль сущности %s сформирован", entity)

    overview_frame = pl.DataFrame(overviews)
    columns_frame = pl.concat(column_frames, how="vertical")
    numeric_frame = pl.concat(numeric_frames, how="vertical_relaxed")
    categorical_frame = pl.concat(categorical_frames, how="vertical_relaxed")

    artifacts = ProfileArtifacts(
        overview_path=output_dir / "dataset_overview.csv",
        columns_path=output_dir / "column_profile.csv",
        numeric_path=output_dir / "numeric_summary.csv",
        categorical_path=output_dir / "categorical_top_values.csv",
        schema_path=output_dir / "schema_inventory.json",
    )
    overview_frame.write_csv(artifacts.overview_path)
    columns_frame.write_csv(artifacts.columns_path)
    numeric_frame.write_csv(artifacts.numeric_path)
    categorical_frame.write_csv(artifacts.categorical_path)
    artifacts.schema_path.write_text(
        json.dumps(schemas, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return artifacts
