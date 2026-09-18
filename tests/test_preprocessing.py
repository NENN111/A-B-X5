"""Тесты консервативной очистки данных."""

from __future__ import annotations

import polars as pl

from src.data.preprocessing import (
    CleaningConfig,
    clean_entity,
    clean_frame,
    run_stage2,
)


def _dirty_frame() -> pl.LazyFrame:
    return pl.DataFrame(
        {
            "id": [1, 1, 2],
            "value": [float("nan"), float("nan"), 5.0],
            "label": [" x ", " x ", "y"],
        }
    ).lazy()


def test_clean_frame_replaces_nan_and_removes_full_duplicates() -> None:
    cleaned = clean_frame(_dirty_frame(), CleaningConfig()).collect()

    assert cleaned.height == 2
    assert cleaned["value"].null_count() == 1
    assert cleaned["value"].is_nan().sum() == 0
    assert " x " in cleaned["label"].to_list()


def test_clean_frame_can_preserve_duplicates() -> None:
    config = CleaningConfig(nan_to_null=False, drop_full_duplicates=False)
    cleaned = clean_frame(_dirty_frame(), config).collect()

    assert cleaned.height == 3
    assert cleaned["value"].is_nan().sum() == 2


def test_clean_entity_writes_parquet_and_audit(tmp_path) -> None:
    output_path = tmp_path / "clean" / "sample.parquet"
    audit = clean_entity(
        "sample",
        _dirty_frame(),
        output_path,
        CleaningConfig(),
    )

    assert output_path.exists()
    assert audit.input_rows == 3
    assert audit.output_rows == 2
    assert audit.rows_removed == 1
    assert audit.nan_values_replaced == 2
    assert pl.read_parquet(output_path).height == 2


def test_run_stage2_creates_all_pipeline_artifacts(tmp_path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    entities = ("clients", "products", "purchases", "uplift_train")
    for entity in entities:
        pl.DataFrame(
            {
                "entity_id": [1, 1, 2],
                "value": [10.0, 10.0, 20.0],
                "segment": ["a", "a", "b"],
            }
        ).write_csv(raw_dir / f"{entity}.csv")

    result = run_stage2(
        raw_dir=raw_dir,
        output_dir=tmp_path / "clean",
        report_dir=tmp_path / "quality",
    )

    assert {audit.entity for audit in result.audits} == set(entities)
    assert all(audit.output_rows == 2 for audit in result.audits)
    assert result.audit_path.exists()
    assert result.raw_profile.overview_path.exists()
    assert result.cleaned_profile.overview_path.exists()
    assert all(
        (tmp_path / "clean" / f"{entity}.parquet").exists() for entity in entities
    )
