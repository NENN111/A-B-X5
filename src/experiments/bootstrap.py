"""Bootstrap-оценка неопределённости абсолютного uplift."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import polars as pl


class BootstrapInputError(ValueError):
    """Ошибка данных или параметров bootstrap."""


@dataclass(frozen=True, slots=True)
class BootstrapSummary:
    """Сводка bootstrap-распределения treatment CR minus control CR."""

    iterations: int
    control_size: int
    treatment_size: int
    observed_uplift: float
    mean_uplift: float
    median_uplift: float
    ci_lower: float
    ci_upper: float
    confidence_level: float
    random_state: int | None

    def to_record(self) -> dict[str, Any]:
        """Преобразовать сводку в плоскую запись."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class BootstrapResult:
    """Сводка и полное распределение bootstrap uplift."""

    summary: BootstrapSummary
    distribution: pl.DataFrame


def _binary_array(values: Iterable[int], name: str) -> np.ndarray:
    array = np.asarray(list(values))
    if array.ndim != 1 or array.size == 0:
        raise BootstrapInputError(f"{name} не должен быть пустым")
    if np.issubdtype(array.dtype, np.floating) and np.isnan(array).any():
        raise BootstrapInputError(f"{name} содержит пропуски")
    if not np.isin(array, [0, 1]).all():
        raise BootstrapInputError(f"{name} должен содержать только 0/1")
    return array.astype(np.int8, copy=False)


def bootstrap_uplift(
    control_outcomes: Iterable[int],
    treatment_outcomes: Iterable[int],
    *,
    iterations: int = 10_000,
    confidence_level: float = 0.95,
    random_state: int | None = 42,
    batch_size: int = 1_000,
) -> BootstrapResult:
    """Оценить uplift непараметрическим bootstrap с ограничением памяти по batch."""
    control = _binary_array(control_outcomes, "control_outcomes")
    treatment = _binary_array(treatment_outcomes, "treatment_outcomes")
    if (
        isinstance(iterations, bool)
        or not isinstance(iterations, int)
        or iterations <= 0
    ):
        raise BootstrapInputError("iterations должен быть положительным целым числом")
    if not 0 < confidence_level < 1:
        raise BootstrapInputError("confidence_level должен быть в диапазоне (0, 1)")
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size <= 0
    ):
        raise BootstrapInputError("batch_size должен быть положительным целым числом")

    rng = np.random.default_rng(random_state)
    uplift = np.empty(iterations, dtype=float)
    control_probability = float(control.mean())
    treatment_probability = float(treatment.mean())
    for start in range(0, iterations, batch_size):
        stop = min(start + batch_size, iterations)
        size = stop - start
        control_means = (
            rng.binomial(control.size, control_probability, size=size) / control.size
        )
        treatment_means = (
            rng.binomial(treatment.size, treatment_probability, size=size)
            / treatment.size
        )
        uplift[start:stop] = treatment_means - control_means

    tail = (1 - confidence_level) / 2
    lower, upper = np.quantile(uplift, [tail, 1 - tail])
    summary = BootstrapSummary(
        iterations=iterations,
        control_size=int(control.size),
        treatment_size=int(treatment.size),
        observed_uplift=float(treatment.mean() - control.mean()),
        mean_uplift=float(uplift.mean()),
        median_uplift=float(np.median(uplift)),
        ci_lower=float(lower),
        ci_upper=float(upper),
        confidence_level=float(confidence_level),
        random_state=random_state,
    )
    distribution = pl.DataFrame(
        {"iteration": np.arange(1, iterations + 1), "absolute_uplift": uplift}
    )
    return BootstrapResult(summary=summary, distribution=distribution)


def bootstrap_uplift_frame(
    frame: pl.LazyFrame,
    *,
    treatment_column: str = "treatment",
    target_column: str = "target",
    iterations: int = 10_000,
    confidence_level: float = 0.95,
    random_state: int | None = 42,
) -> BootstrapResult:
    """Извлечь обе группы из Polars frame и оценить bootstrap uplift."""
    missing = {treatment_column, target_column} - set(frame.collect_schema().names())
    if missing:
        raise BootstrapInputError(f"Отсутствуют колонки: {sorted(missing)}")
    invalid = (
        frame.filter(
            pl.col(treatment_column).is_null()
            | pl.col(target_column).is_null()
            | ~pl.col(treatment_column).is_in([0, 1])
            | ~pl.col(target_column).is_in([0, 1])
        )
        .limit(1)
        .collect(engine="streaming")
    )
    if invalid.height:
        raise BootstrapInputError(
            "Treatment и target должны быть бинарными 0/1 без пропусков"
        )
    groups = (
        frame.select(treatment_column, target_column)
        .collect(engine="streaming")
        .partition_by(treatment_column, as_dict=True)
    )
    try:
        control = groups[(0,)][target_column].to_list()
        treatment = groups[(1,)][target_column].to_list()
    except KeyError as error:
        raise BootstrapInputError(
            "В данных должны присутствовать группы 0 и 1"
        ) from error
    return bootstrap_uplift(
        control,
        treatment,
        iterations=iterations,
        confidence_level=confidence_level,
        random_state=random_state,
    )
