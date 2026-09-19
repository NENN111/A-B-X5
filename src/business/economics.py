"""Сценарная unit economics для стратегий маркетинговой коммуникации."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np
import polars as pl


class EconomicsInputError(ValueError):
    """Ошибка параметров сценария или predicted uplift."""


@dataclass(frozen=True, slots=True)
class EconomicsConfig:
    """Явные сценарные предположения unit economics."""

    communication_cost: float
    profit_per_conversion: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.communication_cost) or self.communication_cost < 0:
            raise EconomicsInputError("communication_cost должен быть конечным и >= 0")
        if (
            not math.isfinite(self.profit_per_conversion)
            or self.profit_per_conversion < 0
        ):
            raise EconomicsInputError(
                "profit_per_conversion должен быть конечным и >= 0"
            )


@dataclass(frozen=True, slots=True)
class StrategyEconomics:
    """Экономические показатели одной стратегии коммуникации."""

    strategy: str
    customers_total: int
    customers_targeted: int
    target_share: float
    targeting_threshold: float | None
    communication_cost: float
    profit_per_conversion: float
    marketing_cost: float
    incremental_conversions: float
    incremental_value: float
    incremental_profit: float
    roi: float | None
    scenario_analysis: bool

    def to_record(self) -> dict[str, object]:
        """Преобразовать результат в плоскую запись."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ThresholdOptimization:
    """Лучший сценарий и полный набор проверенных порогов."""

    best: StrategyEconomics
    scenarios: pl.DataFrame


def _uplift_array(frame: pl.DataFrame, uplift_column: str) -> np.ndarray:
    if uplift_column not in frame.columns:
        raise EconomicsInputError(f"Отсутствует колонка {uplift_column!r}")
    if frame.height == 0:
        raise EconomicsInputError("Таблица predicted uplift не должна быть пустой")
    uplift = frame[uplift_column].to_numpy().astype(float)
    if not np.isfinite(uplift).all():
        raise EconomicsInputError("Predicted uplift должен состоять из конечных чисел")
    return uplift


def _strategy_result(
    uplift: np.ndarray,
    targeted: np.ndarray,
    config: EconomicsConfig,
    *,
    strategy: str,
    threshold: float | None,
) -> StrategyEconomics:
    customers_targeted = int(targeted.sum())
    incremental_conversions = float(uplift[targeted].sum())
    marketing_cost = float(customers_targeted * config.communication_cost)
    incremental_value = float(incremental_conversions * config.profit_per_conversion)
    incremental_profit = incremental_value - marketing_cost
    roi = incremental_profit / marketing_cost if marketing_cost > 0 else None
    return StrategyEconomics(
        strategy=strategy,
        customers_total=int(uplift.size),
        customers_targeted=customers_targeted,
        target_share=float(customers_targeted / uplift.size),
        targeting_threshold=threshold,
        communication_cost=float(config.communication_cost),
        profit_per_conversion=float(config.profit_per_conversion),
        marketing_cost=marketing_cost,
        incremental_conversions=incremental_conversions,
        incremental_value=incremental_value,
        incremental_profit=float(incremental_profit),
        roi=float(roi) if roi is not None else None,
        scenario_analysis=True,
    )


def evaluate_target_all(
    frame: pl.DataFrame,
    config: EconomicsConfig,
    *,
    uplift_column: str = "predicted_uplift",
) -> StrategyEconomics:
    """Оценить коммуникацию со всей аудиторией."""
    uplift = _uplift_array(frame, uplift_column)
    return _strategy_result(
        uplift,
        np.ones(uplift.size, dtype=bool),
        config,
        strategy="target_all",
        threshold=None,
    )


def evaluate_uplift_targeting(
    frame: pl.DataFrame,
    config: EconomicsConfig,
    *,
    threshold: float,
    uplift_column: str = "predicted_uplift",
) -> StrategyEconomics:
    """Оценить правило predicted uplift > threshold."""
    if not math.isfinite(threshold):
        raise EconomicsInputError("threshold должен быть конечным числом")
    uplift = _uplift_array(frame, uplift_column)
    return _strategy_result(
        uplift,
        uplift > threshold,
        config,
        strategy="uplift_targeting",
        threshold=float(threshold),
    )


def compare_strategies(
    frame: pl.DataFrame,
    config: EconomicsConfig,
    *,
    targeting_threshold: float,
    uplift_column: str = "predicted_uplift",
) -> pl.DataFrame:
    """Сравнить Target All и Uplift Targeting в одном сценарии."""
    results = [
        evaluate_target_all(frame, config, uplift_column=uplift_column),
        evaluate_uplift_targeting(
            frame,
            config,
            threshold=targeting_threshold,
            uplift_column=uplift_column,
        ),
    ]
    return pl.DataFrame([result.to_record() for result in results])


def evaluate_thresholds(
    frame: pl.DataFrame,
    config: EconomicsConfig,
    thresholds: Sequence[float],
    *,
    uplift_column: str = "predicted_uplift",
) -> pl.DataFrame:
    """Рассчитать economics для явно заданной сетки порогов."""
    if len(thresholds) == 0:
        raise EconomicsInputError("Список thresholds не должен быть пустым")
    results = [
        evaluate_uplift_targeting(
            frame,
            config,
            threshold=float(threshold),
            uplift_column=uplift_column,
        )
        for threshold in thresholds
    ]
    return pl.DataFrame([result.to_record() for result in results]).sort(
        "targeting_threshold"
    )


def optimize_targeting_threshold(
    frame: pl.DataFrame,
    config: EconomicsConfig,
    *,
    uplift_column: str = "predicted_uplift",
) -> ThresholdOptimization:
    """Найти threshold с максимальной ожидаемой incremental profit."""
    uplift = _uplift_array(frame, uplift_column)
    unique_scores, counts = np.unique(uplift, return_counts=True)
    order = np.argsort(unique_scores)[::-1]
    unique_scores = unique_scores[order]
    counts = counts[order]
    cumulative_customers = np.cumsum(counts)
    cumulative_conversions = np.cumsum(unique_scores * counts)
    targeted = np.concatenate(([0], cumulative_customers)).astype(int)
    incremental_conversions = np.concatenate(([0.0], cumulative_conversions))
    thresholds = np.concatenate(
        ([unique_scores[0]], np.nextafter(unique_scores, -np.inf))
    )
    marketing_cost = targeted * config.communication_cost
    incremental_value = incremental_conversions * config.profit_per_conversion
    incremental_profit = incremental_value - marketing_cost
    roi = [
        float(profit / cost) if cost > 0 else None
        for profit, cost in zip(incremental_profit, marketing_cost, strict=True)
    ]
    scenarios = pl.DataFrame(
        {
            "strategy": ["uplift_targeting"] * targeted.size,
            "customers_total": [int(uplift.size)] * targeted.size,
            "customers_targeted": targeted,
            "target_share": targeted / uplift.size,
            "targeting_threshold": thresholds,
            "communication_cost": [float(config.communication_cost)] * targeted.size,
            "profit_per_conversion": [float(config.profit_per_conversion)]
            * targeted.size,
            "marketing_cost": marketing_cost,
            "incremental_conversions": incremental_conversions,
            "incremental_value": incremental_value,
            "incremental_profit": incremental_profit,
            "roi": roi,
            "scenario_analysis": [True] * targeted.size,
        }
    ).sort(
        ["incremental_profit", "customers_targeted"],
        descending=[True, False],
    )
    best_record = scenarios.row(0, named=True)
    best = StrategyEconomics(
        **{
            field: best_record[field]
            for field in StrategyEconomics.__dataclass_fields__
        }
    )
    return ThresholdOptimization(best=best, scenarios=scenarios)
