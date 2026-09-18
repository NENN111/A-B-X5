"""Диагностика Sample Ratio Mismatch для A/B-эксперимента."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from numbers import Integral
from typing import Any

import polars as pl
from scipy.stats import chisquare


class SRMInputError(ValueError):
    """Ошибка входных данных SRM-проверки."""


@dataclass(frozen=True, slots=True)
class SRMResult:
    """Результат chi-square goodness-of-fit для распределения по группам."""

    observed_control: int
    observed_treatment: int
    expected_control: float
    expected_treatment: float
    expected_control_share: float
    expected_treatment_share: float
    chi_square_statistic: float
    p_value: float
    alpha: float
    srm_detected: bool

    def to_record(self) -> dict[str, Any]:
        """Преобразовать результат в плоскую запись."""
        return asdict(self)


def check_srm(
    observed_control: int,
    observed_treatment: int,
    *,
    expected_treatment_share: float = 0.5,
    alpha: float = 0.05,
) -> SRMResult:
    """Проверить фактическое распределение групп против планового."""
    for name, value in (
        ("observed_control", observed_control),
        ("observed_treatment", observed_treatment),
    ):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
            raise SRMInputError(f"{name} должен быть неотрицательным целым числом")
    if observed_control + observed_treatment == 0:
        raise SRMInputError("Для SRM нужна хотя бы одна наблюдаемая единица")
    if not 0 < expected_treatment_share < 1:
        raise SRMInputError("expected_treatment_share должен быть в диапазоне (0, 1)")
    if not 0 < alpha < 1:
        raise SRMInputError("alpha должен быть в диапазоне (0, 1)")

    total = observed_control + observed_treatment
    expected_treatment = total * expected_treatment_share
    expected_control = total - expected_treatment
    statistic, p_value = chisquare(
        [observed_control, observed_treatment],
        f_exp=[expected_control, expected_treatment],
    )
    return SRMResult(
        observed_control=int(observed_control),
        observed_treatment=int(observed_treatment),
        expected_control=float(expected_control),
        expected_treatment=float(expected_treatment),
        expected_control_share=float(1 - expected_treatment_share),
        expected_treatment_share=float(expected_treatment_share),
        chi_square_statistic=float(statistic),
        p_value=float(p_value),
        alpha=float(alpha),
        srm_detected=bool(p_value < alpha),
    )


def check_srm_frame(
    frame: pl.LazyFrame,
    *,
    treatment_column: str = "treatment",
    expected_treatment_share: float = 0.5,
    alpha: float = 0.05,
) -> SRMResult:
    """Проверить SRM по бинарной treatment-колонке Polars."""
    if treatment_column not in frame.collect_schema().names():
        raise SRMInputError(f"Отсутствует колонка {treatment_column!r}")
    counts = (
        frame.group_by(treatment_column)
        .agg(pl.len().alias("count"))
        .collect(engine="streaming")
    )
    if counts[treatment_column].null_count() or not set(counts[treatment_column]) <= {
        0,
        1,
    }:
        raise SRMInputError(
            "Treatment должен содержать только значения 0/1 без пропусков"
        )
    by_group = dict(counts.iter_rows())
    return check_srm(
        int(by_group.get(0, 0)),
        int(by_group.get(1, 0)),
        expected_treatment_share=expected_treatment_share,
        alpha=alpha,
    )
