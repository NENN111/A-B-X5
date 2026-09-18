"""Тесты базовых метрик бинарного эксперимента."""

from __future__ import annotations

import math

import pytest

from src.experiments.metrics import (
    MetricInputError,
    absolute_uplift,
    cohens_h,
    conversion_rate,
    relative_uplift,
)


def test_conversion_and_uplift_match_manual_calculation() -> None:
    control_rate = conversion_rate(100, 1000)
    treatment_rate = conversion_rate(130, 1000)

    assert control_rate == 0.1
    assert treatment_rate == 0.13
    assert math.isclose(absolute_uplift(control_rate, treatment_rate), 0.03)
    assert math.isclose(relative_uplift(control_rate, treatment_rate), 0.3)
    assert cohens_h(control_rate, treatment_rate) > 0


def test_relative_uplift_is_undefined_for_zero_baseline() -> None:
    assert relative_uplift(0.0, 0.1) is None


@pytest.mark.parametrize(
    ("successes", "total"),
    [(-1, 10), (11, 10), (0, 0), (1.5, 10), (True, 10)],
)
def test_conversion_rate_rejects_invalid_counts(successes, total) -> None:
    with pytest.raises(MetricInputError):
        conversion_rate(successes, total)
