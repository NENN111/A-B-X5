"""Holdout-обучение, сравнение и материализация baseline uplift-моделей."""

from __future__ import annotations

import argparse
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from uuid import uuid4

import numpy as np
import polars as pl
from sklearn.model_selection import train_test_split

from src.config.settings import get_settings
from src.models.preprocessing import (
    UpliftDataError,
    infer_feature_columns,
    validate_uplift_frame,
)
from src.models.s_learner import SLearner
from src.models.t_learner import TLearner
from src.models.uplift_metrics import UpliftEvaluation, evaluate_uplift

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class UpliftSplit:
    """Непересекающиеся train и holdout части клиентской витрины."""

    train: pl.DataFrame
    holdout: pl.DataFrame
    train_row_ids: tuple[int, ...]
    holdout_row_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ModelScore:
    """Сводные holdout-метрики одной uplift-модели."""

    model: str
    train_customers: int
    holdout_customers: int
    auuc: float
    qini_coefficient: float
    overall_observed_uplift: float
    final_incremental_purchases: float
    random_state: int


@dataclass(slots=True)
class ModelingResult:
    """Модели и все таблицы честной holdout-оценки."""

    s_learner: SLearner
    t_learner: TLearner
    predictions: pl.DataFrame
    customer_scores: pl.DataFrame
    comparison: pl.DataFrame
    curves: pl.DataFrame
    deciles: pl.DataFrame
    split: UpliftSplit


def split_uplift_frame(
    frame: pl.DataFrame,
    feature_columns: list[str],
    *,
    treatment_column: str = "treatment",
    target_column: str = "target",
    test_size: float = 0.3,
    random_state: int = 42,
) -> UpliftSplit:
    """Создать воспроизводимый split со стратификацией по treatment × target."""
    validate_uplift_frame(
        frame,
        feature_columns,
        treatment_column=treatment_column,
        target_column=target_column,
    )
    if not 0 < test_size < 1:
        raise UpliftDataError("test_size должен находиться в диапазоне (0, 1)")
    if "__row_id__" in frame.columns:
        raise UpliftDataError("Имя __row_id__ зарезервировано для split")
    indexed = frame.with_row_index("__row_id__")
    row_ids = indexed["__row_id__"].to_numpy()
    treatment = indexed[treatment_column].to_numpy()
    target = indexed[target_column].to_numpy()
    strata = treatment * 2 + target
    _, counts = np.unique(strata, return_counts=True)
    if len(counts) < 4 or counts.min() < 2:
        raise UpliftDataError(
            "Для stratified holdout в каждой комбинации treatment × target "
            "нужно минимум два клиента"
        )
    try:
        train_ids, holdout_ids = train_test_split(
            row_ids,
            test_size=test_size,
            random_state=random_state,
            stratify=strata,
        )
    except ValueError as error:
        raise UpliftDataError(
            f"Не удалось построить stratified holdout: {error}"
        ) from error
    train_ids = np.sort(train_ids)
    holdout_ids = np.sort(holdout_ids)
    train = (
        indexed.filter(pl.col("__row_id__").is_in(train_ids.tolist()))
        .sort("__row_id__")
        .drop("__row_id__")
    )
    holdout = (
        indexed.filter(pl.col("__row_id__").is_in(holdout_ids.tolist()))
        .sort("__row_id__")
        .drop("__row_id__")
    )
    return UpliftSplit(
        train=train,
        holdout=holdout,
        train_row_ids=tuple(int(value) for value in train_ids),
        holdout_row_ids=tuple(int(value) for value in holdout_ids),
    )


def _evaluation_tables(
    evaluation: UpliftEvaluation, model_name: str
) -> tuple[pl.DataFrame, pl.DataFrame]:
    return (
        evaluation.curve.with_columns(pl.lit(model_name).alias("model")),
        evaluation.deciles.with_columns(pl.lit(model_name).alias("model")),
    )


def train_and_evaluate_uplift(
    frame: pl.DataFrame,
    *,
    feature_columns: list[str] | None = None,
    treatment_column: str = "treatment",
    target_column: str = "target",
    client_id_column: str = "client_id",
    test_size: float = 0.3,
    bins: int = 10,
    random_state: int = 42,
) -> ModelingResult:
    """Обучить S/T-Learner и сравнить ранжирование на одном holdout."""
    selected_features = feature_columns or infer_feature_columns(
        frame,
        treatment_column=treatment_column,
        target_column=target_column,
        id_columns=(client_id_column,),
    )
    split = split_uplift_frame(
        frame,
        selected_features,
        treatment_column=treatment_column,
        target_column=target_column,
        test_size=test_size,
        random_state=random_state,
    )
    s_learner = SLearner(random_state=random_state).fit(
        split.train,
        selected_features,
        treatment_column=treatment_column,
        target_column=target_column,
    )
    t_learner = TLearner(random_state=random_state).fit(
        split.train,
        selected_features,
        treatment_column=treatment_column,
        target_column=target_column,
    )
    s_prediction = s_learner.predict_potential_outcomes(split.holdout)
    t_prediction = t_learner.predict_potential_outcomes(split.holdout)
    identity_columns = [
        column
        for column in (client_id_column, treatment_column, target_column)
        if column in split.holdout.columns
    ]
    predictions = split.holdout.select(identity_columns).with_columns(
        pl.Series("source_row_id", split.holdout_row_ids)
    )
    predictions = predictions.hstack(
        s_prediction.rename(
            {
                "probability_control": "s_learner_probability_control",
                "probability_treatment": "s_learner_probability_treatment",
                "predicted_uplift": "s_learner_predicted_uplift",
            }
        )
    ).hstack(
        t_prediction.rename(
            {
                "probability_control": "t_learner_probability_control",
                "probability_treatment": "t_learner_probability_treatment",
                "predicted_uplift": "t_learner_predicted_uplift",
            }
        )
    )

    scores: list[ModelScore] = []
    curves: list[pl.DataFrame] = []
    deciles: list[pl.DataFrame] = []
    for model_name in ("s_learner", "t_learner"):
        evaluation = evaluate_uplift(
            predictions,
            treatment_column=treatment_column,
            target_column=target_column,
            prediction_column=f"{model_name}_predicted_uplift",
            bins=bins,
        )
        scores.append(
            ModelScore(
                model=model_name,
                train_customers=split.train.height,
                holdout_customers=split.holdout.height,
                auuc=evaluation.auuc,
                qini_coefficient=evaluation.qini_coefficient,
                overall_observed_uplift=evaluation.overall_observed_uplift,
                final_incremental_purchases=evaluation.final_incremental_purchases,
                random_state=random_state,
            )
        )
        curve, model_deciles = _evaluation_tables(evaluation, model_name)
        curves.append(curve)
        deciles.append(model_deciles)
    s_learner = SLearner(random_state=random_state).fit(
        frame,
        selected_features,
        treatment_column=treatment_column,
        target_column=target_column,
    )
    t_learner = TLearner(random_state=random_state).fit(
        frame,
        selected_features,
        treatment_column=treatment_column,
        target_column=target_column,
    )
    all_identity = (
        frame.select(client_id_column)
        if client_id_column in frame.columns
        else pl.DataFrame({"source_row_id": np.arange(frame.height)})
    )
    all_s_prediction = s_learner.predict_potential_outcomes(frame).rename(
        {
            "probability_control": "s_learner_probability_control",
            "probability_treatment": "s_learner_probability_treatment",
            "predicted_uplift": "s_learner_predicted_uplift",
        }
    )
    all_t_prediction = t_learner.predict_potential_outcomes(frame).rename(
        {
            "probability_control": "t_learner_probability_control",
            "probability_treatment": "t_learner_probability_treatment",
            "predicted_uplift": "t_learner_predicted_uplift",
        }
    )
    customer_scores = all_identity.hstack(all_s_prediction).hstack(all_t_prediction)
    return ModelingResult(
        s_learner=s_learner,
        t_learner=t_learner,
        predictions=predictions,
        customer_scores=customer_scores,
        comparison=pl.DataFrame([asdict(score) for score in scores]),
        curves=pl.concat(curves, how="vertical"),
        deciles=pl.concat(deciles, how="vertical"),
        split=split,
    )


def _write_parquet_atomic(frame: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{uuid4().hex}.tmp.parquet")
    try:
        frame.write_parquet(temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def materialize_modeling(result: ModelingResult, output_dir: Path) -> dict[str, Path]:
    """Сохранить holdout predictions и таблицы оценки атомарными файлами."""
    outputs = {
        "predictions": output_dir / "uplift_predictions_holdout.parquet",
        "customer_scores": output_dir / "uplift_customer_scores.parquet",
        "comparison": output_dir / "uplift_model_comparison.parquet",
        "curves": output_dir / "uplift_curves.parquet",
        "deciles": output_dir / "uplift_deciles.parquet",
    }
    _write_parquet_atomic(result.predictions, outputs["predictions"])
    _write_parquet_atomic(result.customer_scores, outputs["customer_scores"])
    _write_parquet_atomic(result.comparison, outputs["comparison"])
    _write_parquet_atomic(result.curves, outputs["curves"])
    _write_parquet_atomic(result.deciles, outputs["deciles"])
    return outputs


def main() -> None:
    """Обучить и оценить baseline uplift-модели из командной строки."""
    parser = argparse.ArgumentParser(description="Baseline uplift modeling.")
    parser.add_argument("--input-path", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--random-state", type=int)
    args = parser.parse_args()
    settings = get_settings()
    input_path = (
        args.input_path or settings.processed_data_dir / "customer_features.parquet"
    )
    output_dir = args.output_dir or settings.processed_data_dir / "uplift_modeling"
    random_state = (
        args.random_state if args.random_state is not None else settings.random_state
    )
    if not input_path.exists():
        parser.error(
            f"Feature mart не найдена: {input_path}. Сначала материализуйте реальные данные."
        )
    try:
        result = train_and_evaluate_uplift(
            pl.read_parquet(input_path),
            test_size=args.test_size,
            bins=args.bins,
            random_state=random_state,
        )
    except (UpliftDataError, ValueError) as error:
        parser.error(str(error))
    outputs = materialize_modeling(result, output_dir)
    LOGGER.info("Сохранено файлов: %s", len(outputs))
    print(f"Uplift-модели оценены на holdout; результаты: {output_dir}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
