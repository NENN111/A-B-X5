"""Тесты универсального профилирования данных."""

from __future__ import annotations

import math

import polars as pl

from src.data.quality import numeric_summary, profile_dataset, profile_entity


def _sample_frame() -> pl.LazyFrame:
    return pl.DataFrame(
        {
            "id": [1, 1, 2],
            "value": [10.0, 10.0, float("nan")],
            "segment": ["a", "a", None],
        }
    ).lazy()


def test_profile_entity_counts_quality_signals() -> None:
    overview, columns = profile_entity(
        "sample",
        _sample_frame(),
        exact_cardinality=True,
    )

    assert overview["row_count"] == 3
    assert overview["column_count"] == 3
    assert overview["duplicate_rows"] == 1
    assert overview["total_nulls"] == 1
    assert overview["total_nans"] == 1

    segment = columns.filter(pl.col("column") == "segment").row(0, named=True)
    assert segment["null_count"] == 1
    assert segment["unique_count"] == 2
    assert segment["cardinality_is_exact"] is True


def test_numeric_summary_uses_observed_numeric_columns() -> None:
    summary = numeric_summary("sample", _sample_frame())

    assert set(summary["column"].to_list()) == {"id", "value"}
    id_row = summary.filter(pl.col("column") == "id").row(0, named=True)
    assert math.isclose(id_row["mean"], 4 / 3)
    assert id_row["min"] == 1.0
    assert id_row["max"] == 2.0
    value_row = summary.filter(pl.col("column") == "value").row(0, named=True)
    assert value_row["mean"] == 10.0


def test_profile_dataset_writes_reproducible_artifacts(tmp_path) -> None:
    artifacts = profile_dataset(
        {"sample": _sample_frame()},
        tmp_path,
        exact_cardinality=True,
    )

    assert artifacts.overview_path.exists()
    assert artifacts.columns_path.exists()
    assert artifacts.numeric_path.exists()
    assert artifacts.categorical_path.exists()
    assert artifacts.schema_path.exists()
    assert '"sample"' in artifacts.schema_path.read_text(encoding="utf-8")
