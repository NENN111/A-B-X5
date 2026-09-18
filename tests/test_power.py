"""Тесты power analysis и планирования выборки."""

from __future__ import annotations

import math

import pytest

from src.experiments.power import (
    PowerAnalysisError,
    achieved_power,
    minimum_detectable_effect,
    required_sample_size,
)


def test_power_increases_with_sample_size() -> None:
    small = achieved_power(0.10, 0.12, 500, 500)
    large = achieved_power(0.10, 0.12, 5000, 5000)

    assert 0 < small < large < 1


def test_required_sample_reaches_target_power() -> None:
    required = required_sample_size(0.10, 0.12, target_power=0.8)
    actual_power = achieved_power(
        0.10,
        0.12,
        required.control_size,
        required.treatment_size,
    )

    assert actual_power >= 0.8
    assert required.total_size == required.control_size + required.treatment_size


def test_mde_decreases_with_sample_size() -> None:
    small_sample_mde = minimum_detectable_effect(0.10, 500, 500)
    large_sample_mde = minimum_detectable_effect(0.10, 5000, 5000)

    assert 0 < large_sample_mde < small_sample_mde


def test_equal_rates_have_alpha_power() -> None:
    assert math.isclose(achieved_power(0.1, 0.1, 1000, 1000), 0.05)


def test_required_sample_rejects_zero_effect() -> None:
    with pytest.raises(PowerAnalysisError, match="нулевого эффекта"):
        required_sample_size(0.1, 0.1)


def test_positive_mde_is_impossible_at_full_baseline() -> None:
    with pytest.raises(PowerAnalysisError, match="baseline_rate=1"):
        minimum_detectable_effect(1.0, 1000, 1000)
