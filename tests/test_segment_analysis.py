"""Тесты сегментного анализа и multiple-comparison correction."""

import math

import polars as pl
import pytest
from statsmodels.stats.multitest import multipletests

from src.experiments.ab_test import ExperimentInputError
from src.experiments.segment_analysis import (
    add_validation_segments,
    analyze_segments,
    benjamini_hochberg,
)


def test_benjamini_hochberg_matches_statsmodels() -> None:
    p_values = [0.001, 0.01, 0.03, 0.2]
    adjusted, rejected = benjamini_hochberg(p_values)
    expected_rejected, expected_adjusted, _, _ = multipletests(
        p_values, alpha=0.05, method="fdr_bh"
    )

    assert adjusted == pytest.approx(expected_adjusted.tolist())
    assert rejected == expected_rejected.tolist()
    assert all(0 <= value <= 1 for value in adjusted)


def test_segment_result_matches_manual_rates_and_has_adjusted_p_values() -> None:
    frame = pl.DataFrame(
        {
            "segment": ["A"] * 200 + ["B"] * 200,
            "treatment": [0] * 100 + [1] * 100 + [0] * 100 + [1] * 100,
            "target": (
                [1] * 10
                + [0] * 90
                + [1] * 20
                + [0] * 80
                + [1] * 20
                + [0] * 80
                + [1] * 20
                + [0] * 80
            ),
        }
    ).lazy()
    result = analyze_segments(
        frame, segment_columns=["segment"], add_default_groups=False
    )
    row_a = result.filter(pl.col("segment") == "A").row(0, named=True)

    assert row_a["control_size"] == row_a["treatment_size"] == 100
    assert math.isclose(row_a["control_cr"], 0.1)
    assert math.isclose(row_a["treatment_cr"], 0.2)
    assert math.isclose(row_a["absolute_uplift"], 0.1)
    assert row_a["ci_lower"] < row_a["absolute_uplift"] < row_a["ci_upper"]
    assert "adjusted_p_value" in result.columns
    assert "significant_adjusted" in result.columns


def test_default_segment_derivation_covers_age_and_recency_boundaries() -> None:
    frame = pl.DataFrame(
        {
            "age": [24, 25, 65, None],
            "gender": ["M"] * 4,
            "rfm_segment": ["R5F5M5"] * 4,
            "spending_quantile": [1, 2, 3, 4],
            "purchase_frequency_quantile": [1, 2, 3, 4],
            "recency_score": [1, 2, 4, None],
        }
    ).lazy()
    result = add_validation_segments(frame).collect()

    assert result["age_group"].to_list() == ["до 25", "25–34", "65+", "Неизвестно"]
    assert result["recency_group"].to_list() == [
        "Давние",
        "Средняя давность",
        "Недавние",
        "Неизвестно",
    ]


def test_segment_analysis_rejects_bad_binary_target() -> None:
    frame = pl.DataFrame(
        {"segment": ["A", "A"], "treatment": [0, 1], "target": [0, 2]}
    ).lazy()
    with pytest.raises(ExperimentInputError):
        analyze_segments(frame, segment_columns=["segment"], add_default_groups=False)
