"""Базовые метрики бинарного A/B-эксперимента."""

from __future__ import annotations

import math
from numbers import Integral


class MetricInputError(ValueError):
    """Ошибка входных binomial counts."""


def validate_binomial_counts(successes: int, total: int) -> None:
    """Проверить число успехов и размер группы."""
    if isinstance(successes, bool) or not isinstance(successes, Integral):
        raise MetricInputError("successes должен быть целым числом")
    if isinstance(total, bool) or not isinstance(total, Integral):
        raise MetricInputError("total должен быть целым числом")
    if total <= 0:
        raise MetricInputError("total должен быть положительным")
    if successes < 0 or successes > total:
        raise MetricInputError("successes должен находиться в диапазоне [0, total]")


def conversion_rate(successes: int, total: int) -> float:
    """Рассчитать долю конверсий."""
    validate_binomial_counts(successes, total)
    return float(successes / total)


def absolute_uplift(control_rate: float, treatment_rate: float) -> float:
    """Рассчитать абсолютную разницу treatment minus control."""
    return float(treatment_rate - control_rate)


def relative_uplift(
    control_rate: float,
    treatment_rate: float,
) -> float | None:
    """Рассчитать относительный uplift; при нулевом baseline вернуть None."""
    if control_rate == 0:
        return None
    return float((treatment_rate - control_rate) / control_rate)


def cohens_h(control_rate: float, treatment_rate: float) -> float:
    """Рассчитать signed Cohen's h для двух долей."""
    if not 0 <= control_rate <= 1 or not 0 <= treatment_rate <= 1:
        raise MetricInputError("Доли должны находиться в диапазоне [0, 1]")
    return float(
        2 * math.asin(math.sqrt(treatment_rate))
        - 2 * math.asin(math.sqrt(control_rate))
    )
