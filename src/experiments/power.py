"""Power analysis и планирование выборки для двух долей."""

from __future__ import annotations

import math
from dataclasses import dataclass

from scipy.optimize import brentq
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize

from src.experiments.confidence_intervals import Alternative


class PowerAnalysisError(ValueError):
    """Ошибка некорректного или недостижимого power-сценария."""


@dataclass(frozen=True, slots=True)
class RequiredSampleSize:
    """Требуемые размеры групп при заданном allocation ratio."""

    control_size: int
    treatment_size: int
    total_size: int
    target_power: float


def _validate_probability(value: float, name: str) -> None:
    if not 0 <= value <= 1:
        raise PowerAnalysisError(f"{name} должен находиться в диапазоне [0, 1]")


def _validate_power_inputs(
    control_rate: float,
    treatment_rate: float,
    alpha: float,
) -> None:
    _validate_probability(control_rate, "control_rate")
    _validate_probability(treatment_rate, "treatment_rate")
    if not 0 < alpha < 1:
        raise PowerAnalysisError("alpha должен находиться в диапазоне (0, 1)")


def achieved_power(
    control_rate: float,
    treatment_rate: float,
    control_size: int,
    treatment_size: int,
    *,
    alpha: float = 0.05,
    alternative: Alternative = "two-sided",
) -> float:
    """Рассчитать мощность z-теста двух независимых долей."""
    _validate_power_inputs(control_rate, treatment_rate, alpha)
    if control_size <= 0 or treatment_size <= 0:
        raise PowerAnalysisError("Размеры обеих групп должны быть положительными")
    effect_size = proportion_effectsize(treatment_rate, control_rate)
    if effect_size == 0:
        return float(alpha)
    power = NormalIndPower().power(
        effect_size=effect_size,
        nobs1=control_size,
        alpha=alpha,
        ratio=treatment_size / control_size,
        alternative=alternative,
    )
    return float(power)


def required_sample_size(
    control_rate: float,
    treatment_rate: float,
    *,
    target_power: float = 0.8,
    alpha: float = 0.05,
    allocation_ratio: float = 1.0,
    alternative: Alternative = "two-sided",
) -> RequiredSampleSize:
    """Рассчитать требуемую выборку для заданных ожидаемых долей."""
    _validate_power_inputs(control_rate, treatment_rate, alpha)
    if not 0 < target_power < 1:
        raise PowerAnalysisError("target_power должен находиться в диапазоне (0, 1)")
    if allocation_ratio <= 0:
        raise PowerAnalysisError("allocation_ratio должен быть положительным")
    effect_size = proportion_effectsize(treatment_rate, control_rate)
    if effect_size == 0:
        raise PowerAnalysisError("Для нулевого эффекта размер выборки не определён")
    control_size = math.ceil(
        NormalIndPower().solve_power(
            effect_size=effect_size,
            power=target_power,
            alpha=alpha,
            ratio=allocation_ratio,
            alternative=alternative,
        )
    )
    treatment_size = math.ceil(control_size * allocation_ratio)
    return RequiredSampleSize(
        control_size=control_size,
        treatment_size=treatment_size,
        total_size=control_size + treatment_size,
        target_power=target_power,
    )


def minimum_detectable_effect(
    baseline_rate: float,
    control_size: int,
    treatment_size: int,
    *,
    target_power: float = 0.8,
    alpha: float = 0.05,
    alternative: Alternative = "two-sided",
) -> float:
    """Найти минимальный положительный absolute uplift при заданной выборке."""
    _validate_probability(baseline_rate, "baseline_rate")
    if baseline_rate >= 1:
        raise PowerAnalysisError("Положительный MDE невозможен при baseline_rate=1")
    if not 0 < target_power < 1:
        raise PowerAnalysisError("target_power должен находиться в диапазоне (0, 1)")
    if control_size <= 0 or treatment_size <= 0:
        raise PowerAnalysisError("Размеры обеих групп должны быть положительными")

    maximum_delta = 1 - baseline_rate

    def objective(delta: float) -> float:
        return (
            achieved_power(
                baseline_rate,
                baseline_rate + delta,
                control_size,
                treatment_size,
                alpha=alpha,
                alternative=alternative,
            )
            - target_power
        )

    lower = max(1e-12, maximum_delta * 1e-12)
    if objective(maximum_delta) < 0:
        raise PowerAnalysisError(
            "Заданная мощность недостижима даже при treatment_rate=1"
        )
    return float(brentq(objective, lower, maximum_delta))
