"""Переиспользуемый two-proportion z-test и интеграция с feature mart."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import polars as pl
from scipy.stats import norm

from src.config.settings import get_settings
from src.experiments.confidence_intervals import (
    Alternative,
    difference_confidence_interval,
    difference_standard_error,
    wilson_interval,
)
from src.experiments.metrics import (
    absolute_uplift,
    cohens_h,
    conversion_rate,
    relative_uplift,
)
from src.experiments.power import (
    PowerAnalysisError,
    achieved_power,
    minimum_detectable_effect,
    required_sample_size,
)


class ExperimentInputError(ValueError):
    """Ошибка данных бинарного эксперимента."""


@dataclass(frozen=True, slots=True)
class ExperimentResult:
    """Полный результат анализа основной бинарной метрики."""

    control_size: int
    treatment_size: int
    control_conversions: int
    treatment_conversions: int
    control_cr: float
    treatment_cr: float
    absolute_uplift: float
    relative_uplift: float | None
    standard_error: float
    null_standard_error: float
    z_statistic: float
    p_value: float
    ci_lower: float
    ci_upper: float
    control_ci_lower: float
    control_ci_upper: float
    treatment_ci_lower: float
    treatment_ci_upper: float
    effect_size: float
    achieved_power: float
    required_control_size: int | None
    required_treatment_size: int | None
    required_total_size: int | None
    mde_absolute: float | None
    alpha: float
    target_power: float
    alternative: Alternative
    statistically_significant: bool
    warnings: tuple[str, ...]

    def to_record(self) -> dict[str, Any]:
        """Преобразовать результат в плоскую запись для Parquet/SQL."""
        record = asdict(self)
        record["warnings"] = json.dumps(self.warnings, ensure_ascii=False)
        return record


def _p_value(z_statistic: float, alternative: Alternative) -> float:
    """Рассчитать p-value по выбранной альтернативе."""
    if alternative == "two-sided":
        return float(2 * norm.sf(abs(z_statistic)))
    if alternative == "larger":
        return float(norm.sf(z_statistic))
    if alternative == "smaller":
        return float(norm.cdf(z_statistic))
    raise ExperimentInputError(f"Неизвестная alternative: {alternative}")


def two_proportion_z_test(
    control_successes: int,
    control_total: int,
    treatment_successes: int,
    treatment_total: int,
    *,
    alternative: Alternative = "two-sided",
) -> tuple[float, float, float]:
    """Выполнить pooled two-proportion z-test под нулевой гипотезой."""
    control_rate = conversion_rate(control_successes, control_total)
    treatment_rate = conversion_rate(treatment_successes, treatment_total)
    pooled_rate = (control_successes + treatment_successes) / (
        control_total + treatment_total
    )
    null_variance = (
        pooled_rate * (1 - pooled_rate) * (1 / control_total + 1 / treatment_total)
    )
    null_standard_error = math.sqrt(null_variance)
    difference = treatment_rate - control_rate
    if null_standard_error == 0:
        z_statistic = 0.0 if difference == 0 else math.copysign(math.inf, difference)
    else:
        z_statistic = difference / null_standard_error
    return (
        float(z_statistic),
        _p_value(z_statistic, alternative),
        float(null_standard_error),
    )


def run_ab_test(
    control_successes: int,
    control_total: int,
    treatment_successes: int,
    treatment_total: int,
    *,
    alpha: float = 0.05,
    target_power: float = 0.8,
    alternative: Alternative = "two-sided",
) -> ExperimentResult:
    """Рассчитать эффект, inference и планирование выборки эксперимента."""
    if not 0 < alpha < 1:
        raise ExperimentInputError("alpha должен находиться в диапазоне (0, 1)")
    if not 0 < target_power < 1:
        raise ExperimentInputError("target_power должен находиться в диапазоне (0, 1)")

    control_rate = conversion_rate(control_successes, control_total)
    treatment_rate = conversion_rate(treatment_successes, treatment_total)
    uplift = absolute_uplift(control_rate, treatment_rate)
    relative = relative_uplift(control_rate, treatment_rate)
    z_statistic, p_value, null_standard_error = two_proportion_z_test(
        control_successes,
        control_total,
        treatment_successes,
        treatment_total,
        alternative=alternative,
    )
    interval = difference_confidence_interval(
        control_successes,
        control_total,
        treatment_successes,
        treatment_total,
        alpha=alpha,
        alternative=alternative,
    )
    control_interval = wilson_interval(control_successes, control_total, alpha=alpha)
    treatment_interval = wilson_interval(
        treatment_successes,
        treatment_total,
        alpha=alpha,
    )
    power = achieved_power(
        control_rate,
        treatment_rate,
        control_total,
        treatment_total,
        alpha=alpha,
        alternative=alternative,
    )

    warnings: list[str] = []
    if relative is None:
        warnings.append("Relative uplift не определён при control_cr=0")

    required_control: int | None = None
    required_treatment: int | None = None
    required_total: int | None = None
    if uplift == 0:
        warnings.append("Required sample size не определён для нулевого эффекта")
    else:
        try:
            required = required_sample_size(
                control_rate,
                treatment_rate,
                target_power=target_power,
                alpha=alpha,
                allocation_ratio=treatment_total / control_total,
                alternative=alternative,
            )
            required_control = required.control_size
            required_treatment = required.treatment_size
            required_total = required.total_size
        except (PowerAnalysisError, ValueError) as error:
            warnings.append(f"Required sample size не рассчитан: {error}")

    mde: float | None = None
    try:
        mde = minimum_detectable_effect(
            control_rate,
            control_total,
            treatment_total,
            target_power=target_power,
            alpha=alpha,
            alternative=alternative,
        )
    except (PowerAnalysisError, ValueError) as error:
        warnings.append(f"MDE не рассчитан: {error}")

    return ExperimentResult(
        control_size=control_total,
        treatment_size=treatment_total,
        control_conversions=control_successes,
        treatment_conversions=treatment_successes,
        control_cr=control_rate,
        treatment_cr=treatment_rate,
        absolute_uplift=uplift,
        relative_uplift=relative,
        standard_error=difference_standard_error(
            control_successes,
            control_total,
            treatment_successes,
            treatment_total,
        ),
        null_standard_error=null_standard_error,
        z_statistic=z_statistic,
        p_value=p_value,
        ci_lower=interval.lower,
        ci_upper=interval.upper,
        control_ci_lower=control_interval.lower,
        control_ci_upper=control_interval.upper,
        treatment_ci_lower=treatment_interval.lower,
        treatment_ci_upper=treatment_interval.upper,
        effect_size=cohens_h(control_rate, treatment_rate),
        achieved_power=power,
        required_control_size=required_control,
        required_treatment_size=required_treatment,
        required_total_size=required_total,
        mde_absolute=mde,
        alpha=alpha,
        target_power=target_power,
        alternative=alternative,
        statistically_significant=p_value < alpha,
        warnings=tuple(warnings),
    )


def analyze_experiment_frame(
    frame: pl.LazyFrame,
    *,
    treatment_column: str = "treatment",
    target_column: str = "target",
    alpha: float = 0.05,
    target_power: float = 0.8,
    alternative: Alternative = "two-sided",
) -> ExperimentResult:
    """Проверить бинарные поля Polars и проанализировать experiment frame."""
    schema = frame.collect_schema()
    missing = {treatment_column, target_column} - set(schema.names())
    if missing:
        raise ExperimentInputError(
            f"Отсутствуют колонки эксперимента: {sorted(missing)}"
        )
    invalid = (
        frame.filter(
            pl.col(treatment_column).is_null()
            | pl.col(target_column).is_null()
            | ~pl.col(treatment_column).is_in([0, 1])
            | ~pl.col(target_column).is_in([0, 1])
        )
        .select(treatment_column, target_column)
        .limit(5)
        .collect(engine="streaming")
    )
    if invalid.height:
        raise ExperimentInputError(
            "Treatment и target должны быть заполненными бинарными значениями 0/1. "
            f"Примеры ошибок: {invalid.to_dicts()}"
        )
    aggregates = (
        frame.group_by(treatment_column)
        .agg(
            pl.len().alias("total"),
            pl.col(target_column).cast(pl.Int64).sum().alias("conversions"),
        )
        .collect(engine="streaming")
    )
    groups = {
        int(row[treatment_column]): row for row in aggregates.iter_rows(named=True)
    }
    if set(groups) != {0, 1}:
        raise ExperimentInputError("В данных должны присутствовать группы 0 и 1")
    return run_ab_test(
        int(groups[0]["conversions"]),
        int(groups[0]["total"]),
        int(groups[1]["conversions"]),
        int(groups[1]["total"]),
        alpha=alpha,
        target_power=target_power,
        alternative=alternative,
    )


def materialize_experiment_result(
    result: ExperimentResult,
    output_path: Path,
) -> Path:
    """Атомарно записать одну строку результата в Parquet."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(f".{uuid4().hex}.tmp.parquet")
    try:
        pl.DataFrame([result.to_record()]).write_parquet(temporary_path)
        temporary_path.replace(output_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return output_path


def main() -> None:
    """Запустить A/B-анализ клиентской витрины из командной строки."""
    parser = argparse.ArgumentParser(description="Two-proportion A/B analysis.")
    parser.add_argument("--input-path", type=Path)
    parser.add_argument("--output-path", type=Path)
    parser.add_argument("--treatment-column", default="treatment")
    parser.add_argument("--target-column", default="target")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--target-power", type=float, default=0.8)
    parser.add_argument(
        "--alternative",
        choices=("two-sided", "larger", "smaller"),
        default="two-sided",
    )
    args = parser.parse_args()
    settings = get_settings()
    input_path = (
        args.input_path or settings.processed_data_dir / "customer_features.parquet"
    )
    output_path = (
        args.output_path or settings.processed_data_dir / "experiment_results.parquet"
    )
    if not input_path.exists():
        parser.error(
            f"Feature mart не найдена: {input_path}. Сначала выполните Stage 3."
        )
    try:
        result = analyze_experiment_frame(
            pl.scan_parquet(input_path),
            treatment_column=args.treatment_column,
            target_column=args.target_column,
            alpha=args.alpha,
            target_power=args.target_power,
            alternative=args.alternative,
        )
    except (ExperimentInputError, ValueError) as error:
        parser.error(str(error))
    materialize_experiment_result(result, output_path)
    print(
        f"Stage 4 завершён: uplift={result.absolute_uplift:.6f}, "
        f"p-value={result.p_value:.6g}, результат={output_path}"
    )


if __name__ == "__main__":
    main()
