"""Тесты A/A-симуляции."""

import math

import numpy as np
import polars as pl
import pytest

from src.experiments.aa_test import (
    AATestInputError,
    run_aa_from_frame,
    run_aa_simulation,
)


def test_aa_simulation_is_reproducible() -> None:
    outcomes = [0, 1] * 100
    first = run_aa_simulation(outcomes, simulations=100, random_state=17)
    second = run_aa_simulation(outcomes, simulations=100, random_state=17)

    assert first.summary == second.summary
    assert first.simulations.equals(second.simulations)


def test_aa_false_positive_rate_is_close_to_alpha_for_reference_sample() -> None:
    outcomes = np.random.default_rng(3).binomial(1, 0.2, size=2_000)
    result = run_aa_simulation(outcomes, simulations=1_000, random_state=11)

    assert 0.02 <= result.summary.false_positive_rate <= 0.08
    assert math.isclose(
        result.summary.false_positive_rate,
        result.summary.false_positives / 1_000,
    )
    assert result.simulations.height == 1_000


def test_aa_constant_outcome_has_no_false_positives() -> None:
    result = run_aa_simulation([0] * 20, simulations=20)

    assert result.summary.false_positive_rate == 0
    assert result.simulations["p_value"].to_list() == [1.0] * 20


def test_aa_frame_uses_control_only() -> None:
    frame = pl.DataFrame(
        {"treatment": [0, 0, 0, 0, 1, 1], "target": [0, 1, 0, 1, 1, 1]}
    ).lazy()
    result = run_aa_from_frame(frame, simulations=10)

    assert result.summary.sample_size == 4


@pytest.mark.parametrize("values", [[], [1], [0, 2], [0, float("nan")]])
def test_aa_rejects_invalid_outcomes(values: list[float]) -> None:
    with pytest.raises(AATestInputError):
        run_aa_simulation(values)
