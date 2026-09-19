"""Тесты baseline uplift-моделей и специализированных метрик."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from src.models.model_evaluation import (
    materialize_modeling,
    split_uplift_frame,
    train_and_evaluate_uplift,
)
from src.models.preprocessing import UpliftDataError
from src.models.s_learner import SLearner
from src.models.t_learner import TLearner
from src.models.uplift_metrics import evaluate_uplift, uplift_by_decile


def _modeling_frame(size: int = 1_000) -> pl.DataFrame:
    rng = np.random.default_rng(19)
    feature = rng.normal(size=size)
    treatment = rng.binomial(1, 0.5, size=size)
    logit = -1.8 + 0.5 * feature + treatment * (0.9 + 0.7 * (feature > 0))
    probability = 1 / (1 + np.exp(-logit))
    target = rng.binomial(1, probability)
    category = np.where(feature > 0, "high", "low").astype(object)
    category[::97] = None
    return pl.DataFrame(
        {
            "client_id": np.arange(size),
            "numeric_feature": feature,
            "category_feature": category.tolist(),
            "treatment": treatment,
            "target": target,
        }
    )


def _ranking_frame(reverse: bool = False) -> pl.DataFrame:
    treatment = np.tile([0, 1], 100)
    high_effect = np.arange(200) < 100
    target = np.where(high_effect & (treatment == 1), 1, 0)
    prediction = high_effect.astype(float)
    if reverse:
        prediction = -prediction
    return pl.DataFrame(
        {
            "treatment": treatment,
            "target": target,
            "predicted_uplift": prediction,
        }
    )


@pytest.mark.parametrize("learner_class", [SLearner, TLearner])
def test_learners_return_valid_potential_outcomes(learner_class: type) -> None:
    frame = _modeling_frame()
    learner = learner_class(random_state=7).fit(
        frame, ["numeric_feature", "category_feature"]
    )
    prediction = learner.predict_potential_outcomes(frame.head(100))

    assert prediction.shape == (100, 3)
    assert prediction["probability_control"].is_between(0, 1).all()
    assert prediction["probability_treatment"].is_between(0, 1).all()
    assert prediction["predicted_uplift"].mean() > 0


def test_s_learner_is_reproducible() -> None:
    frame = _modeling_frame(600)
    first = SLearner(random_state=11).fit(frame, ["numeric_feature"])
    second = SLearner(random_state=11).fit(frame, ["numeric_feature"])

    assert first.predict_potential_outcomes(frame).equals(
        second.predict_potential_outcomes(frame)
    )


def test_uplift_metrics_reward_correct_ranking() -> None:
    correct = evaluate_uplift(_ranking_frame())
    reverse = evaluate_uplift(_ranking_frame(reverse=True))

    assert correct.qini_coefficient > reverse.qini_coefficient
    assert correct.auuc > reverse.auuc
    assert math.isclose(correct.final_incremental_purchases, 100.0)
    assert correct.curve.row(0, named=True)["customers"] == 0
    assert correct.curve.row(-1, named=True)["customers"] == 200


def test_uplift_deciles_have_required_schema_and_preserve_population() -> None:
    deciles = uplift_by_decile(_ranking_frame(), bins=10)

    assert deciles.height == 10
    assert deciles["customers"].sum() == 200
    assert deciles["customers"].to_list() == [20] * 10
    assert {
        "decile",
        "customers",
        "avg_predicted_uplift",
        "treatment_cr",
        "control_cr",
        "observed_uplift",
        "incremental_purchases",
        "cumulative_incremental_purchases",
    } <= set(deciles.columns)


def test_split_is_reproducible_stratified_and_non_overlapping() -> None:
    frame = _modeling_frame(800)
    first = split_uplift_frame(frame, ["numeric_feature"], random_state=5)
    second = split_uplift_frame(frame, ["numeric_feature"], random_state=5)

    assert first.train_row_ids == second.train_row_ids
    assert first.holdout_row_ids == second.holdout_row_ids
    assert set(first.train_row_ids).isdisjoint(first.holdout_row_ids)
    assert len(first.train_row_ids) + len(first.holdout_row_ids) == frame.height
    assert first.train.group_by("treatment", "target").len().height == 4
    assert first.holdout.group_by("treatment", "target").len().height == 4


def test_full_modeling_pipeline_materializes_holdout_outputs(tmp_path: Path) -> None:
    frame = _modeling_frame(800)
    result = train_and_evaluate_uplift(frame, bins=5, random_state=3)
    outputs = materialize_modeling(result, tmp_path / "uplift")

    assert set(outputs) == {
        "predictions",
        "customer_scores",
        "comparison",
        "curves",
        "deciles",
    }
    assert all(path.exists() for path in outputs.values())
    assert result.comparison["model"].to_list() == ["s_learner", "t_learner"]
    assert result.predictions.height == result.split.holdout.height
    assert result.customer_scores.height == frame.height
    assert result.deciles.group_by("model").len()["len"].to_list() == [5, 5]
    assert pl.read_parquet(outputs["predictions"]).height == result.split.holdout.height


@pytest.mark.parametrize(
    "feature_columns",
    [["target"], ["treatment"], [], ["missing"]],
)
def test_training_rejects_leakage_and_bad_feature_contract(
    feature_columns: list[str],
) -> None:
    with pytest.raises(UpliftDataError):
        SLearner().fit(_modeling_frame(100), feature_columns)


def test_metrics_reject_non_finite_predictions() -> None:
    frame = _ranking_frame().with_columns(
        pl.when(pl.int_range(pl.len()) == 0)
        .then(float("nan"))
        .otherwise(pl.col("predicted_uplift"))
        .alias("predicted_uplift")
    )
    with pytest.raises(UpliftDataError):
        evaluate_uplift(frame)
