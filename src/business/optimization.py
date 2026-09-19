"""Оркестрация клиентского таргетинга и сценарной бизнес-экономики."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import polars as pl

from src.business.economics import (
    EconomicsConfig,
    ThresholdOptimization,
    compare_strategies,
    optimize_targeting_threshold,
)
from src.business.targeting import apply_targeting_rule, assign_customer_types
from src.config.settings import get_settings

SUPPORTED_MODELS = {"s_learner", "t_learner"}


@dataclass(frozen=True, slots=True)
class BusinessOptimizationResult:
    """Клиентские решения и таблицы сценарной экономики."""

    selected_model: str
    customer_targeting: pl.DataFrame
    strategy_comparison: pl.DataFrame
    threshold_optimization: ThresholdOptimization


def select_model_by_qini(comparison: pl.DataFrame) -> str:
    """Выбрать learner с максимальным holdout Qini coefficient."""
    required = {"model", "qini_coefficient"}
    missing = required - set(comparison.columns)
    if missing:
        raise ValueError(f"В model comparison отсутствуют колонки: {sorted(missing)}")
    candidates = comparison.filter(
        pl.col("model").is_in(sorted(SUPPORTED_MODELS))
        & pl.col("qini_coefficient").is_not_null()
        & pl.col("qini_coefficient").is_finite()
    ).sort(["qini_coefficient", "model"], descending=[True, False])
    if candidates.height == 0:
        raise ValueError("Нет поддерживаемой модели с конечным holdout Qini")
    return str(candidates["model"][0])


def prepare_model_scores(
    customer_scores: pl.DataFrame,
    model: str,
) -> pl.DataFrame:
    """Преобразовать score выбранной модели к общему бизнес-контракту."""
    if model not in SUPPORTED_MODELS:
        raise ValueError(f"Неизвестная модель {model!r}: {sorted(SUPPORTED_MODELS)}")
    mapping = {
        f"{model}_probability_control": "probability_control",
        f"{model}_probability_treatment": "probability_treatment",
        f"{model}_predicted_uplift": "predicted_uplift",
    }
    missing = set(mapping) - set(customer_scores.columns)
    if missing:
        raise ValueError(f"В customer scores отсутствуют колонки: {sorted(missing)}")
    identity = [
        column for column in ("client_id", "source_row_id") if column in customer_scores
    ]
    if not identity:
        raise ValueError("В customer scores отсутствует client_id/source_row_id")
    return (
        customer_scores.select(*identity, *mapping)
        .rename(mapping)
        .with_columns(pl.lit(model).alias("selected_model"))
    )


def optimize_business_targeting(
    customer_scores: pl.DataFrame,
    config: EconomicsConfig,
    *,
    model: str,
    targeting_threshold: float,
    effect_threshold: float = 0.05,
    outcome_probability_threshold: float = 0.5,
) -> BusinessOptimizationResult:
    """Классифицировать score, применить правило и сравнить economics."""
    scores = prepare_model_scores(customer_scores, model)
    typed = assign_customer_types(
        scores,
        effect_threshold=effect_threshold,
        outcome_probability_threshold=outcome_probability_threshold,
    )
    customer_targeting = apply_targeting_rule(
        typed,
        threshold=targeting_threshold,
    )
    comparison = compare_strategies(
        scores,
        config,
        targeting_threshold=targeting_threshold,
    ).with_columns(pl.lit(model).alias("selected_model"))
    optimization = optimize_targeting_threshold(scores, config)
    optimization_table = optimization.scenarios.with_columns(
        pl.lit(model).alias("selected_model")
    )
    optimization = ThresholdOptimization(
        best=optimization.best,
        scenarios=optimization_table,
    )
    return BusinessOptimizationResult(
        selected_model=model,
        customer_targeting=customer_targeting,
        strategy_comparison=comparison,
        threshold_optimization=optimization,
    )


def _write_parquet_atomic(frame: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{uuid4().hex}.tmp.parquet")
    try:
        frame.write_parquet(temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def materialize_business_optimization(
    result: BusinessOptimizationResult,
    output_dir: Path,
) -> dict[str, Path]:
    """Атомарно сохранить клиентские решения и сценарные таблицы."""
    outputs = {
        "customer_targeting": output_dir / "customer_targeting.parquet",
        "strategy_comparison": output_dir / "business_scenarios.parquet",
        "threshold_optimization": output_dir / "threshold_optimization.parquet",
    }
    _write_parquet_atomic(result.customer_targeting, outputs["customer_targeting"])
    _write_parquet_atomic(result.strategy_comparison, outputs["strategy_comparison"])
    _write_parquet_atomic(
        result.threshold_optimization.scenarios,
        outputs["threshold_optimization"],
    )
    return outputs


def main() -> None:
    """Запустить бизнес-оптимизацию на сохранённых uplift-score."""
    parser = argparse.ArgumentParser(description="Uplift targeting economics.")
    parser.add_argument("--scores-path", type=Path)
    parser.add_argument("--comparison-path", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--model", choices=sorted(SUPPORTED_MODELS))
    parser.add_argument("--communication-cost", type=float, required=True)
    parser.add_argument("--profit-per-conversion", type=float, required=True)
    parser.add_argument("--targeting-threshold", type=float, required=True)
    parser.add_argument("--effect-threshold", type=float, default=0.05)
    parser.add_argument("--outcome-probability-threshold", type=float, default=0.5)
    args = parser.parse_args()
    settings = get_settings()
    modeling_dir = settings.processed_data_dir / "uplift_modeling"
    scores_path = args.scores_path or modeling_dir / "uplift_customer_scores.parquet"
    comparison_path = (
        args.comparison_path or modeling_dir / "uplift_model_comparison.parquet"
    )
    output_dir = (
        args.output_dir or settings.processed_data_dir / "business_optimization"
    )
    missing = [path for path in (scores_path, comparison_path) if not path.exists()]
    if missing:
        parser.error(
            f"Не найдены результаты uplift-моделей: {[str(path) for path in missing]}. "
            "Сначала выполните python -m src.models.model_evaluation."
        )
    try:
        customer_scores = pl.read_parquet(scores_path)
        comparison = pl.read_parquet(comparison_path)
        model = args.model or select_model_by_qini(comparison)
        result = optimize_business_targeting(
            customer_scores,
            EconomicsConfig(
                communication_cost=args.communication_cost,
                profit_per_conversion=args.profit_per_conversion,
            ),
            model=model,
            targeting_threshold=args.targeting_threshold,
            effect_threshold=args.effect_threshold,
            outcome_probability_threshold=args.outcome_probability_threshold,
        )
    except ValueError as error:
        parser.error(str(error))
    outputs = materialize_business_optimization(result, output_dir)
    print(
        f"Бизнес-сценарии рассчитаны для {result.selected_model}; "
        f"файлов сохранено: {len(outputs)}, каталог: {output_dir}"
    )


if __name__ == "__main__":
    main()
