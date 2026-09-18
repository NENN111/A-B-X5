"""Тесты bootstrap uplift."""

import math

import polars as pl
import pytest

from src.experiments.bootstrap import (
    BootstrapInputError,
    bootstrap_uplift,
    bootstrap_uplift_frame,
)


def test_bootstrap_is_reproducible_and_contains_observed_effect() -> None:
    control = [1] * 20 + [0] * 80
    treatment = [1] * 30 + [0] * 70
    first = bootstrap_uplift(control, treatment, iterations=500, random_state=7)
    second = bootstrap_uplift(control, treatment, iterations=500, random_state=7)

    assert first.summary == second.summary
    assert first.distribution.equals(second.distribution)
    assert math.isclose(first.summary.observed_uplift, 0.1)
    assert first.summary.ci_lower < 0.1 < first.summary.ci_upper


def test_bootstrap_degenerate_sample_has_zero_width_interval() -> None:
    result = bootstrap_uplift([0] * 10, [1] * 10, iterations=50)

    assert result.summary.mean_uplift == 1
    assert result.summary.median_uplift == 1
    assert result.summary.ci_lower == result.summary.ci_upper == 1


def test_bootstrap_frame_extracts_both_groups() -> None:
    frame = pl.DataFrame({"treatment": [0, 0, 1, 1], "target": [0, 1, 1, 1]}).lazy()
    result = bootstrap_uplift_frame(frame, iterations=20)

    assert result.summary.control_size == 2
    assert result.summary.treatment_size == 2
    assert result.distribution.height == 20


@pytest.mark.parametrize(
    ("control", "treatment"),
    [([], [0]), ([0], []), ([0, 2], [1]), ([float("nan")], [1])],
)
def test_bootstrap_rejects_invalid_outcomes(
    control: list[float], treatment: list[float]
) -> None:
    with pytest.raises(BootstrapInputError):
        bootstrap_uplift(control, treatment)
