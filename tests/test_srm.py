"""Тесты Sample Ratio Mismatch."""

import math

import polars as pl
import pytest
from scipy.stats import chisquare

from src.experiments.srm import SRMInputError, check_srm, check_srm_frame


def test_srm_matches_scipy_reference() -> None:
    result = check_srm(480, 520)
    statistic, p_value = chisquare([480, 520], f_exp=[500, 500])

    assert math.isclose(result.chi_square_statistic, statistic, rel_tol=1e-12)
    assert math.isclose(result.p_value, p_value, rel_tol=1e-12)
    assert result.srm_detected is False
    assert result.expected_control == result.expected_treatment == 500


def test_srm_detects_material_imbalance() -> None:
    result = check_srm(300, 700)

    assert result.srm_detected is True
    assert result.p_value < 0.05


def test_srm_frame_supports_non_equal_planned_allocation() -> None:
    frame = pl.DataFrame({"arm": [0] * 750 + [1] * 250}).lazy()
    result = check_srm_frame(
        frame, treatment_column="arm", expected_treatment_share=0.25
    )

    assert result.p_value == 1
    assert result.srm_detected is False


@pytest.mark.parametrize(
    ("control", "treatment", "share"),
    [(-1, 1, 0.5), (0, 0, 0.5), (10, 10, 0), (10, 10, 1)],
)
def test_srm_rejects_invalid_inputs(control: int, treatment: int, share: float) -> None:
    with pytest.raises(SRMInputError):
        check_srm(control, treatment, expected_treatment_share=share)


def test_srm_frame_rejects_non_binary_assignment() -> None:
    with pytest.raises(SRMInputError):
        check_srm_frame(pl.DataFrame({"treatment": [0, 1, 2]}).lazy())
