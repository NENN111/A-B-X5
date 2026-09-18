"""Доверительные интервалы для бинарных метрик эксперимента."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from scipy.stats import norm

from src.experiments.metrics import conversion_rate, validate_binomial_counts

Alternative = Literal["two-sided", "larger", "smaller"]


@dataclass(frozen=True, slots=True)
class ConfidenceInterval:
    """Границы доверительного интервала и его уровень."""

    lower: float
    upper: float
    confidence_level: float


def _validate_alpha(alpha: float) -> None:
    if not 0 < alpha < 1:
        raise ValueError("alpha должен находиться в диапазоне (0, 1)")


def _validate_alternative(alternative: Alternative) -> None:
    if alternative not in {"two-sided", "larger", "smaller"}:
        raise ValueError(f"Неизвестная alternative: {alternative}")


def difference_standard_error(
    control_successes: int,
    control_total: int,
    treatment_successes: int,
    treatment_total: int,
) -> float:
    """Рассчитать unpooled standard error абсолютной разницы долей."""
    control_rate = conversion_rate(control_successes, control_total)
    treatment_rate = conversion_rate(treatment_successes, treatment_total)
    variance = (
        control_rate * (1 - control_rate) / control_total
        + treatment_rate * (1 - treatment_rate) / treatment_total
    )
    return float(math.sqrt(variance))


def difference_confidence_interval(
    control_successes: int,
    control_total: int,
    treatment_successes: int,
    treatment_total: int,
    *,
    alpha: float = 0.05,
    alternative: Alternative = "two-sided",
) -> ConfidenceInterval:
    """Построить Wald CI для разницы treatment minus control."""
    _validate_alpha(alpha)
    _validate_alternative(alternative)
    control_rate = conversion_rate(control_successes, control_total)
    treatment_rate = conversion_rate(treatment_successes, treatment_total)
    difference = treatment_rate - control_rate
    standard_error = difference_standard_error(
        control_successes,
        control_total,
        treatment_successes,
        treatment_total,
    )
    if alternative == "two-sided":
        critical_value = float(norm.ppf(1 - alpha / 2))
        lower = difference - critical_value * standard_error
        upper = difference + critical_value * standard_error
    elif alternative == "larger":
        critical_value = float(norm.ppf(1 - alpha))
        lower = difference - critical_value * standard_error
        upper = 1.0
    else:
        critical_value = float(norm.ppf(1 - alpha))
        lower = -1.0
        upper = difference + critical_value * standard_error
    return ConfidenceInterval(
        lower=max(-1.0, float(lower)),
        upper=min(1.0, float(upper)),
        confidence_level=1 - alpha,
    )


def wilson_interval(
    successes: int,
    total: int,
    *,
    alpha: float = 0.05,
) -> ConfidenceInterval:
    """Построить устойчивый Wilson CI для одной конверсии."""
    validate_binomial_counts(successes, total)
    _validate_alpha(alpha)
    rate = successes / total
    z_value = float(norm.ppf(1 - alpha / 2))
    denominator = 1 + z_value**2 / total
    center = (rate + z_value**2 / (2 * total)) / denominator
    half_width = (
        z_value
        * math.sqrt(rate * (1 - rate) / total + z_value**2 / (4 * total**2))
        / denominator
    )
    return ConfidenceInterval(
        lower=max(0.0, float(center - half_width)),
        upper=min(1.0, float(center + half_width)),
        confidence_level=1 - alpha,
    )
