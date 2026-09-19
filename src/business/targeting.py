"""Правила uplift-таргетинга и модельная интерпретация типов клиентов."""

from __future__ import annotations

import math

import polars as pl


class TargetingInputError(ValueError):
    """Ошибка входных score или параметров таргетинга."""


def _validate_probability_scores(
    frame: pl.DataFrame,
    *,
    control_probability_column: str,
    treatment_probability_column: str,
    uplift_column: str,
    tolerance: float = 1e-9,
) -> None:
    required = {
        control_probability_column,
        treatment_probability_column,
        uplift_column,
    }
    missing = required - set(frame.columns)
    if missing:
        raise TargetingInputError(f"Отсутствуют score-колонки: {sorted(missing)}")
    if frame.height == 0:
        raise TargetingInputError("Таблица клиентских score не должна быть пустой")
    invalid = frame.filter(
        pl.col(control_probability_column).is_null()
        | pl.col(treatment_probability_column).is_null()
        | pl.col(uplift_column).is_null()
        | ~pl.col(control_probability_column).is_finite()
        | ~pl.col(treatment_probability_column).is_finite()
        | ~pl.col(uplift_column).is_finite()
        | ~pl.col(control_probability_column).is_between(0, 1, closed="both")
        | ~pl.col(treatment_probability_column).is_between(0, 1, closed="both")
        | (
            (
                pl.col(treatment_probability_column)
                - pl.col(control_probability_column)
                - pl.col(uplift_column)
            ).abs()
            > tolerance
        )
    )
    if invalid.height:
        raise TargetingInputError(
            "Вероятности должны быть конечными значениями из [0, 1], а uplift — "
            "их разностью treatment minus control"
        )


def assign_customer_types(
    frame: pl.DataFrame,
    *,
    control_probability_column: str = "probability_control",
    treatment_probability_column: str = "probability_treatment",
    uplift_column: str = "predicted_uplift",
    effect_threshold: float = 0.05,
    outcome_probability_threshold: float = 0.5,
) -> pl.DataFrame:
    """Добавить модельный тип клиента без утверждения знания counterfactual."""
    if not math.isfinite(effect_threshold) or effect_threshold < 0:
        raise TargetingInputError("effect_threshold должен быть конечным и >= 0")
    if not 0 <= outcome_probability_threshold <= 1:
        raise TargetingInputError(
            "outcome_probability_threshold должен находиться в диапазоне [0, 1]"
        )
    _validate_probability_scores(
        frame,
        control_probability_column=control_probability_column,
        treatment_probability_column=treatment_probability_column,
        uplift_column=uplift_column,
    )
    average_outcome_probability = (
        pl.col(control_probability_column) + pl.col(treatment_probability_column)
    ) / 2
    customer_type = (
        pl.when(pl.col(uplift_column) > effect_threshold)
        .then(pl.lit("Persuadables"))
        .when(pl.col(uplift_column) < -effect_threshold)
        .then(pl.lit("Sleeping Dogs"))
        .when(average_outcome_probability >= outcome_probability_threshold)
        .then(pl.lit("Sure Things"))
        .otherwise(pl.lit("Lost Causes"))
    )
    return frame.with_columns(
        customer_type.alias("modeled_customer_type"),
        pl.lit(True).alias("customer_type_is_model_interpretation"),
    )


def apply_targeting_rule(
    frame: pl.DataFrame,
    *,
    threshold: float,
    uplift_column: str = "predicted_uplift",
    output_column: str = "is_targeted",
) -> pl.DataFrame:
    """Таргетировать клиентов по строгому правилу predicted uplift > threshold."""
    if not math.isfinite(threshold):
        raise TargetingInputError("threshold должен быть конечным числом")
    if uplift_column not in frame.columns:
        raise TargetingInputError(f"Отсутствует колонка {uplift_column!r}")
    invalid = frame.filter(
        pl.col(uplift_column).is_null() | ~pl.col(uplift_column).is_finite()
    )
    if invalid.height:
        raise TargetingInputError("Predicted uplift должен состоять из конечных чисел")
    return frame.with_columns(
        (pl.col(uplift_column) > threshold).alias(output_column),
        pl.lit(float(threshold)).alias("targeting_threshold"),
    )
