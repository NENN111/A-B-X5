"""Воспроизводимая A/A-симуляция на наблюдениях контрольной группы."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import polars as pl
from scipy.stats import kstest

from src.experiments.ab_test import two_proportion_z_test


class AATestInputError(ValueError):
    """Ошибка параметров или данных A/A-симуляции."""


@dataclass(frozen=True, slots=True)
class AASummary:
    """Сводка распределения p-value из A/A-симуляций."""

    simulations: int
    sample_size: int
    split_a_size: int
    split_b_size: int
    alpha: float
    false_positives: int
    false_positive_rate: float
    mean_p_value: float
    median_p_value: float
    ks_uniform_statistic: float
    ks_uniform_p_value: float
    random_state: int | None

    def to_record(self) -> dict[str, Any]:
        """Преобразовать сводку в плоскую запись."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class AATestResult:
    """Сводка и таблица отдельных A/A-итераций."""

    summary: AASummary
    simulations: pl.DataFrame


def _binary_array(values: Iterable[int], name: str) -> np.ndarray:
    array = np.asarray(list(values))
    if array.ndim != 1 or array.size < 2:
        raise AATestInputError(f"{name} должен содержать минимум два наблюдения")
    if np.issubdtype(array.dtype, np.floating) and np.isnan(array).any():
        raise AATestInputError(f"{name} содержит пропуски")
    if not np.isin(array, [0, 1]).all():
        raise AATestInputError(f"{name} должен содержать только 0/1")
    return array.astype(np.int8, copy=False)


def run_aa_simulation(
    control_outcomes: Iterable[int],
    *,
    simulations: int = 1_000,
    alpha: float = 0.05,
    random_state: int | None = 42,
) -> AATestResult:
    """Многократно случайно разделить control и сохранить p-value каждой итерации."""
    outcomes = _binary_array(control_outcomes, "control_outcomes")
    if (
        isinstance(simulations, bool)
        or not isinstance(simulations, int)
        or simulations <= 0
    ):
        raise AATestInputError("simulations должен быть положительным целым числом")
    if not 0 < alpha < 1:
        raise AATestInputError("alpha должен быть в диапазоне (0, 1)")

    split_a_size = outcomes.size // 2
    split_b_size = outcomes.size - split_a_size
    total_successes = int(outcomes.sum())
    total_failures = int(outcomes.size - total_successes)
    rng = np.random.default_rng(random_state)
    rows: list[dict[str, float | int | bool]] = []
    for iteration in range(simulations):
        successes_a = int(
            rng.hypergeometric(total_successes, total_failures, split_a_size)
        )
        successes_b = total_successes - successes_a
        rate_a = successes_a / split_a_size
        rate_b = successes_b / split_b_size
        z_statistic, p_value, _ = two_proportion_z_test(
            successes_a, split_a_size, successes_b, split_b_size
        )
        rows.append(
            {
                "iteration": iteration + 1,
                "split_a_cr": rate_a,
                "split_b_cr": rate_b,
                "absolute_uplift": rate_b - rate_a,
                "z_statistic": z_statistic,
                "p_value": p_value,
                "false_positive": p_value < alpha,
            }
        )

    distribution = pl.DataFrame(rows)
    p_values = distribution["p_value"].to_numpy()
    ks_statistic, ks_p_value = kstest(p_values, "uniform")
    false_positives = int(distribution["false_positive"].sum())
    summary = AASummary(
        simulations=simulations,
        sample_size=int(outcomes.size),
        split_a_size=split_a_size,
        split_b_size=split_b_size,
        alpha=float(alpha),
        false_positives=false_positives,
        false_positive_rate=float(false_positives / simulations),
        mean_p_value=float(np.mean(p_values)),
        median_p_value=float(np.median(p_values)),
        ks_uniform_statistic=float(ks_statistic),
        ks_uniform_p_value=float(ks_p_value),
        random_state=random_state,
    )
    return AATestResult(summary=summary, simulations=distribution)


def run_aa_from_frame(
    frame: pl.LazyFrame,
    *,
    treatment_column: str = "treatment",
    target_column: str = "target",
    simulations: int = 1_000,
    alpha: float = 0.05,
    random_state: int | None = 42,
) -> AATestResult:
    """Извлечь outcomes контрольной группы и запустить A/A-симуляцию."""
    missing = {treatment_column, target_column} - set(frame.collect_schema().names())
    if missing:
        raise AATestInputError(f"Отсутствуют колонки: {sorted(missing)}")
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
        raise AATestInputError(
            "Treatment и target должны быть бинарными 0/1 без пропусков"
        )
    control = (
        frame.filter(pl.col(treatment_column) == 0)
        .select(target_column)
        .collect(engine="streaming")[target_column]
        .to_list()
    )
    return run_aa_simulation(
        control,
        simulations=simulations,
        alpha=alpha,
        random_state=random_state,
    )
