"""Анализ неоднородности A/B-эффекта по клиентским сегментам."""

from __future__ import annotations

import math
from collections.abc import Sequence

import polars as pl
from statsmodels.stats.multitest import multipletests

from src.experiments.ab_test import ExperimentInputError, two_proportion_z_test
from src.experiments.confidence_intervals import difference_confidence_interval
from src.experiments.metrics import conversion_rate, relative_uplift

DEFAULT_SEGMENT_COLUMNS = (
    "age_group",
    "gender",
    "rfm_segment",
    "spending_quantile",
    "purchase_frequency_quantile",
    "recency_group",
)


def add_validation_segments(frame: pl.LazyFrame) -> pl.LazyFrame:
    """Добавить согласованные возрастные и recency-группы к feature mart."""
    required = {
        "age",
        "gender",
        "rfm_segment",
        "spending_quantile",
        "purchase_frequency_quantile",
        "recency_score",
    }
    missing = required - set(frame.collect_schema().names())
    if missing:
        raise ExperimentInputError(
            f"Для полного сегментного анализа отсутствуют колонки: {sorted(missing)}"
        )
    age = pl.col("age")
    age_group = (
        pl.when(age.is_null())
        .then(pl.lit("Неизвестно"))
        .when(age < 25)
        .then(pl.lit("до 25"))
        .when(age < 35)
        .then(pl.lit("25–34"))
        .when(age < 45)
        .then(pl.lit("35–44"))
        .when(age < 55)
        .then(pl.lit("45–54"))
        .when(age < 65)
        .then(pl.lit("55–64"))
        .otherwise(pl.lit("65+"))
    )
    recency = pl.col("recency_score")
    recency_group = (
        pl.when(recency.is_null())
        .then(pl.lit("Неизвестно"))
        .when(recency >= 4)
        .then(pl.lit("Недавние"))
        .when(recency >= 2)
        .then(pl.lit("Средняя давность"))
        .otherwise(pl.lit("Давние"))
    )
    return frame.with_columns(
        age_group.alias("age_group"),
        recency_group.alias("recency_group"),
    )


def benjamini_hochberg(
    p_values: Sequence[float], *, alpha: float = 0.05
) -> tuple[list[float], list[bool]]:
    """Скорректировать семейство p-value методом Benjamini–Hochberg."""
    if not 0 < alpha < 1:
        raise ExperimentInputError("alpha должен быть в диапазоне (0, 1)")
    if not p_values:
        return [], []
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in p_values):
        raise ExperimentInputError("p_values должны быть конечными числами из [0, 1]")
    rejected, adjusted, _, _ = multipletests(p_values, alpha=alpha, method="fdr_bh")
    return adjusted.astype(float).tolist(), rejected.astype(bool).tolist()


def analyze_segments(
    frame: pl.LazyFrame,
    *,
    segment_columns: Sequence[str] = DEFAULT_SEGMENT_COLUMNS,
    treatment_column: str = "treatment",
    target_column: str = "target",
    alpha: float = 0.05,
    add_default_groups: bool = True,
) -> pl.DataFrame:
    """Посчитать uplift внутри сегментов и глобально скорректировать p-value."""
    if not segment_columns:
        raise ExperimentInputError("Нужно передать хотя бы одну колонку сегмента")
    prepared = add_validation_segments(frame) if add_default_groups else frame
    required = set(segment_columns) | {treatment_column, target_column}
    missing = required - set(prepared.collect_schema().names())
    if missing:
        raise ExperimentInputError(f"Отсутствуют колонки: {sorted(missing)}")

    invalid = (
        prepared.filter(
            pl.col(treatment_column).is_null()
            | pl.col(target_column).is_null()
            | ~pl.col(treatment_column).is_in([0, 1])
            | ~pl.col(target_column).is_in([0, 1])
        )
        .limit(1)
        .collect(engine="streaming")
    )
    if invalid.height:
        raise ExperimentInputError(
            "Treatment и target должны быть бинарными 0/1 без пропусков"
        )

    rows: list[dict[str, object]] = []
    for dimension in segment_columns:
        aggregates = (
            prepared.with_columns(
                pl.col(dimension)
                .cast(pl.String)
                .fill_null("Неизвестно")
                .alias("_segment")
            )
            .group_by("_segment", treatment_column)
            .agg(
                pl.len().alias("size"),
                pl.col(target_column).cast(pl.Int64).sum().alias("conversions"),
            )
            .collect(engine="streaming")
        )
        grouped: dict[str, dict[int, dict[str, object]]] = {}
        for row in aggregates.iter_rows(named=True):
            grouped.setdefault(str(row["_segment"]), {})[int(row[treatment_column])] = (
                row
            )
        for segment, groups in grouped.items():
            if set(groups) != {0, 1}:
                continue
            control_size = int(groups[0]["size"])
            treatment_size = int(groups[1]["size"])
            control_conversions = int(groups[0]["conversions"])
            treatment_conversions = int(groups[1]["conversions"])
            control_cr = conversion_rate(control_conversions, control_size)
            treatment_cr = conversion_rate(treatment_conversions, treatment_size)
            uplift = treatment_cr - control_cr
            _, p_value, _ = two_proportion_z_test(
                control_conversions,
                control_size,
                treatment_conversions,
                treatment_size,
            )
            interval = difference_confidence_interval(
                control_conversions,
                control_size,
                treatment_conversions,
                treatment_size,
                alpha=alpha,
            )
            rows.append(
                {
                    "dimension": dimension,
                    "segment": segment,
                    "control_size": control_size,
                    "treatment_size": treatment_size,
                    "control_conversions": control_conversions,
                    "treatment_conversions": treatment_conversions,
                    "control_cr": control_cr,
                    "treatment_cr": treatment_cr,
                    "absolute_uplift": uplift,
                    "relative_uplift": relative_uplift(control_cr, treatment_cr),
                    "ci_lower": interval.lower,
                    "ci_upper": interval.upper,
                    "confidence_interval": f"[{interval.lower:.6f}, {interval.upper:.6f}]",
                    "p_value": p_value,
                    "significant_raw": p_value < alpha,
                }
            )
    if not rows:
        raise ExperimentInputError(
            "Нет сегментов, в которых одновременно присутствуют группы 0 и 1"
        )
    adjusted, rejected = benjamini_hochberg(
        [float(row["p_value"]) for row in rows], alpha=alpha
    )
    for row, adjusted_p, significant in zip(rows, adjusted, rejected, strict=True):
        row["adjusted_p_value"] = adjusted_p
        row["significant_adjusted"] = significant
    return pl.DataFrame(rows).sort("dimension", "segment")
