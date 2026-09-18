"""Тесты two-proportion A/B testing engine."""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest
from statsmodels.stats.proportion import (
    confint_proportions_2indep,
    proportions_ztest,
)

from src.experiments.ab_test import (
    ExperimentInputError,
    analyze_experiment_frame,
    materialize_experiment_result,
    run_ab_test,
    two_proportion_z_test,
)


def test_z_test_matches_statsmodels_reference() -> None:
    z_statistic, p_value, _ = two_proportion_z_test(100, 1000, 130, 1000)
    expected_z, expected_p = proportions_ztest(
        count=np.array([130, 100]),
        nobs=np.array([1000, 1000]),
        alternative="two-sided",
    )

    assert math.isclose(z_statistic, expected_z, rel_tol=1e-12)
    assert math.isclose(p_value, expected_p, rel_tol=1e-12)


def test_wald_difference_interval_matches_statsmodels() -> None:
    result = run_ab_test(100, 1000, 130, 1000)
    expected_lower, expected_upper = confint_proportions_2indep(
        count1=130,
        nobs1=1000,
        count2=100,
        nobs2=1000,
        method="wald",
        compare="diff",
        alpha=0.05,
    )

    assert math.isclose(result.ci_lower, expected_lower, rel_tol=1e-12)
    assert math.isclose(result.ci_upper, expected_upper, rel_tol=1e-12)
    assert result.ci_lower < result.absolute_uplift < result.ci_upper


def test_group_swap_preserves_two_sided_p_value_and_flips_sign() -> None:
    forward = run_ab_test(100, 1000, 130, 1000)
    reverse = run_ab_test(130, 1000, 100, 1000)

    assert math.isclose(forward.p_value, reverse.p_value, rel_tol=1e-12)
    assert math.isclose(forward.z_statistic, -reverse.z_statistic, rel_tol=1e-12)
    assert math.isclose(
        forward.absolute_uplift,
        -reverse.absolute_uplift,
        rel_tol=1e-12,
    )


def test_significance_agrees_with_interval_for_reference_case() -> None:
    result = run_ab_test(100, 1000, 160, 1000)

    assert result.statistically_significant is True
    assert result.p_value < result.alpha
    assert result.ci_lower > 0


def test_all_zero_outcomes_are_handled_without_nan() -> None:
    result = run_ab_test(0, 100, 0, 100)

    assert result.z_statistic == 0
    assert result.p_value == 1
    assert result.absolute_uplift == 0
    assert result.relative_uplift is None
    assert result.statistically_significant is False
    assert any("Relative uplift" in warning for warning in result.warnings)


def test_frame_analysis_and_parquet_materialization(tmp_path) -> None:
    frame = pl.DataFrame(
        {
            "treatment": [0] * 100 + [1] * 100,
            "target": [1] * 10 + [0] * 90 + [1] * 15 + [0] * 85,
        }
    ).lazy()
    result = analyze_experiment_frame(frame)
    output_path = tmp_path / "experiment_results.parquet"

    materialize_experiment_result(result, output_path)
    stored = pl.read_parquet(output_path).row(0, named=True)

    assert result.control_size == 100
    assert result.treatment_size == 100
    assert result.control_conversions == 10
    assert result.treatment_conversions == 15
    assert math.isclose(result.absolute_uplift, 0.05)
    assert stored["p_value"] == result.p_value


@pytest.mark.parametrize(
    "frame",
    [
        pl.DataFrame({"treatment": [0, 1], "target": [0, 2]}).lazy(),
        pl.DataFrame({"treatment": [0, None], "target": [0, 1]}).lazy(),
        pl.DataFrame({"treatment": [0, 0], "target": [0, 1]}).lazy(),
    ],
)
def test_frame_analysis_rejects_invalid_experiment_data(frame) -> None:
    with pytest.raises(ExperimentInputError):
        analyze_experiment_frame(frame)
