"""Метрики ранжирования и таблицы оценки uplift-моделей."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from src.models.preprocessing import UpliftDataError


@dataclass(frozen=True, slots=True)
class UpliftEvaluation:
    """Сводные метрики и подробные таблицы uplift-ранжирования."""

    auuc: float
    qini_coefficient: float
    overall_observed_uplift: float
    final_incremental_purchases: float
    curve: pl.DataFrame
    deciles: pl.DataFrame


def _validated_arrays(
    frame: pl.DataFrame,
    *,
    treatment_column: str,
    target_column: str,
    prediction_column: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    required = {treatment_column, target_column, prediction_column}
    missing = required - set(frame.columns)
    if missing:
        raise UpliftDataError(f"Отсутствуют колонки оценки: {sorted(missing)}")
    treatment = frame[treatment_column].to_numpy()
    target = frame[target_column].to_numpy()
    prediction = frame[prediction_column].to_numpy().astype(float)
    if frame.height == 0:
        raise UpliftDataError("Выборка оценки не должна быть пустой")
    if not np.isin(treatment, [0, 1]).all() or not np.isin(target, [0, 1]).all():
        raise UpliftDataError("Treatment и target должны быть бинарными 0/1")
    if not np.isfinite(prediction).all():
        raise UpliftDataError("Predicted uplift должен состоять из конечных чисел")
    if set(treatment.tolist()) != {0, 1}:
        raise UpliftDataError("Для оценки нужны одновременно control и treatment")
    return treatment.astype(np.int8), target.astype(np.int8), prediction


def uplift_curve(
    frame: pl.DataFrame,
    *,
    treatment_column: str = "treatment",
    target_column: str = "target",
    prediction_column: str = "predicted_uplift",
) -> pl.DataFrame:
    """Построить IPW cumulative uplift и Qini curve по убыванию score."""
    treatment, target, prediction = _validated_arrays(
        frame,
        treatment_column=treatment_column,
        target_column=target_column,
        prediction_column=prediction_column,
    )
    order = np.argsort(-prediction, kind="stable")
    treatment = treatment[order]
    target = target[order]
    prediction = prediction[order]
    propensity = float(treatment.mean())
    transformed_outcome = target * treatment / propensity - target * (1 - treatment) / (
        1 - propensity
    )
    cumulative_gain = np.cumsum(transformed_outcome)
    customers = np.arange(1, frame.height + 1)
    fraction = customers / frame.height
    cumulative_uplift = cumulative_gain / customers
    random_baseline = fraction * cumulative_gain[-1]
    return (
        pl.DataFrame(
            {
                "rank": np.arange(1, frame.height + 1),
                "customers": customers,
                "targeted_fraction": fraction,
                "predicted_uplift": prediction,
                "cumulative_incremental_purchases": cumulative_gain,
                "cumulative_uplift": cumulative_uplift,
                "qini_random_baseline": random_baseline,
            }
        )
        .vstack(
            pl.DataFrame(
                {
                    "rank": [0],
                    "customers": [0],
                    "targeted_fraction": [0.0],
                    "predicted_uplift": [prediction[0]],
                    "cumulative_incremental_purchases": [0.0],
                    "cumulative_uplift": [0.0],
                    "qini_random_baseline": [0.0],
                }
            )
        )
        .sort("rank")
    )


def uplift_by_decile(
    frame: pl.DataFrame,
    *,
    treatment_column: str = "treatment",
    target_column: str = "target",
    prediction_column: str = "predicted_uplift",
    bins: int = 10,
) -> pl.DataFrame:
    """Агрегировать observed uplift и incremental purchases по score-бинам."""
    if isinstance(bins, bool) or not isinstance(bins, int) or bins <= 0:
        raise UpliftDataError("bins должен быть положительным целым числом")
    treatment, target, prediction = _validated_arrays(
        frame,
        treatment_column=treatment_column,
        target_column=target_column,
        prediction_column=prediction_column,
    )
    order = np.argsort(-prediction, kind="stable")
    treatment = treatment[order]
    target = target[order]
    prediction = prediction[order]
    actual_bins = min(bins, frame.height)
    decile = (
        np.floor(np.arange(frame.height) * actual_bins / frame.height).astype(int) + 1
    )
    rows: list[dict[str, int | float | None]] = []
    cumulative = 0.0
    cumulative_available = True
    for current in range(1, actual_bins + 1):
        mask = decile == current
        group_treatment = treatment[mask]
        group_target = target[mask]
        treatment_mask = group_treatment == 1
        control_mask = ~treatment_mask
        treatment_size = int(treatment_mask.sum())
        control_size = int(control_mask.sum())
        treatment_cr = (
            float(group_target[treatment_mask].mean()) if treatment_size else None
        )
        control_cr = float(group_target[control_mask].mean()) if control_size else None
        observed_uplift = (
            treatment_cr - control_cr
            if treatment_cr is not None and control_cr is not None
            else None
        )
        incremental = (
            float(mask.sum() * observed_uplift) if observed_uplift is not None else None
        )
        if incremental is None:
            cumulative_available = False
        elif cumulative_available:
            cumulative += incremental
        rows.append(
            {
                "decile": current,
                "customers": int(mask.sum()),
                "control_size": control_size,
                "treatment_size": treatment_size,
                "avg_predicted_uplift": float(prediction[mask].mean()),
                "treatment_cr": treatment_cr,
                "control_cr": control_cr,
                "observed_uplift": observed_uplift,
                "incremental_purchases": incremental,
                "cumulative_incremental_purchases": (
                    cumulative if cumulative_available else None
                ),
            }
        )
    return pl.DataFrame(rows)


def evaluate_uplift(
    frame: pl.DataFrame,
    *,
    treatment_column: str = "treatment",
    target_column: str = "target",
    prediction_column: str = "predicted_uplift",
    bins: int = 10,
) -> UpliftEvaluation:
    """Рассчитать AUUC, Qini coefficient, curve и таблицу децилей."""
    curve = uplift_curve(
        frame,
        treatment_column=treatment_column,
        target_column=target_column,
        prediction_column=prediction_column,
    )
    deciles = uplift_by_decile(
        frame,
        treatment_column=treatment_column,
        target_column=target_column,
        prediction_column=prediction_column,
        bins=bins,
    )
    fraction = curve["targeted_fraction"].to_numpy()
    cumulative_uplift = curve["cumulative_uplift"].to_numpy()
    cumulative_gain = curve["cumulative_incremental_purchases"].to_numpy()
    random_baseline = curve["qini_random_baseline"].to_numpy()
    treatment = frame[treatment_column].to_numpy().astype(bool)
    target = frame[target_column].to_numpy()
    treatment_cr = float(target[treatment].mean())
    control_cr = float(target[~treatment].mean())
    return UpliftEvaluation(
        auuc=float(np.trapezoid(cumulative_uplift, fraction)),
        qini_coefficient=float(
            np.trapezoid(cumulative_gain - random_baseline, fraction)
        ),
        overall_observed_uplift=treatment_cr - control_cr,
        final_incremental_purchases=float(cumulative_gain[-1]),
        curve=curve,
        deciles=deciles,
    )
