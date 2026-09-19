"""Тесты uplift-таргетинга и сценарной unit economics."""

from __future__ import annotations

import math
from pathlib import Path

import polars as pl
import pytest

from src.business.economics import (
    EconomicsConfig,
    EconomicsInputError,
    compare_strategies,
    evaluate_uplift_targeting,
    optimize_targeting_threshold,
)
from src.business.optimization import (
    materialize_business_optimization,
    optimize_business_targeting,
    select_model_by_qini,
)
from src.business.targeting import (
    TargetingInputError,
    apply_targeting_rule,
    assign_customer_types,
)


def _economics_frame() -> pl.DataFrame:
    return pl.DataFrame({"predicted_uplift": [0.2, 0.1, -0.05, 0.0]})


def _customer_scores() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "client_id": [1, 2, 3, 4],
            "s_learner_probability_control": [0.2, 0.4, 0.3, 0.6],
            "s_learner_probability_treatment": [0.3, 0.42, 0.2, 0.61],
            "s_learner_predicted_uplift": [0.1, 0.02, -0.1, 0.01],
            "t_learner_probability_control": [0.2, 0.7, 0.4, 0.1],
            "t_learner_probability_treatment": [0.4, 0.72, 0.2, 0.11],
            "t_learner_predicted_uplift": [0.2, 0.02, -0.2, 0.01],
        }
    )


def test_customer_type_rules_are_exhaustive_and_marked_as_interpretation() -> None:
    typed = assign_customer_types(
        pl.DataFrame(
            {
                "probability_control": [0.2, 0.7, 0.4, 0.1],
                "probability_treatment": [0.4, 0.72, 0.2, 0.11],
                "predicted_uplift": [0.2, 0.02, -0.2, 0.01],
            }
        ),
        effect_threshold=0.05,
    )

    assert typed["modeled_customer_type"].to_list() == [
        "Persuadables",
        "Sure Things",
        "Sleeping Dogs",
        "Lost Causes",
    ]
    assert typed["customer_type_is_model_interpretation"].all()


def test_targeting_rule_uses_strict_threshold() -> None:
    result = apply_targeting_rule(
        pl.DataFrame({"predicted_uplift": [0.2, 0.1, 0.0]}), threshold=0.1
    )

    assert result["is_targeted"].to_list() == [True, False, False]
    assert result["targeting_threshold"].to_list() == [0.1] * 3


def test_strategy_economics_matches_manual_calculation() -> None:
    config = EconomicsConfig(communication_cost=2, profit_per_conversion=100)
    comparison = compare_strategies(
        _economics_frame(), config, targeting_threshold=0.05
    )
    target_all = comparison.filter(pl.col("strategy") == "target_all").row(
        0, named=True
    )
    targeted = comparison.filter(pl.col("strategy") == "uplift_targeting").row(
        0, named=True
    )

    assert target_all["customers_targeted"] == 4
    assert math.isclose(target_all["incremental_conversions"], 0.25)
    assert target_all["marketing_cost"] == 8
    assert target_all["incremental_profit"] == pytest.approx(17)
    assert math.isclose(target_all["roi"], 2.125)
    assert targeted["customers_targeted"] == 2
    assert math.isclose(targeted["incremental_conversions"], 0.3)
    assert targeted["marketing_cost"] == 4
    assert targeted["incremental_profit"] == pytest.approx(26)
    assert math.isclose(targeted["roi"], 6.5)
    assert targeted["scenario_analysis"] is True


def test_zero_communication_cost_has_undefined_roi_not_infinity() -> None:
    result = evaluate_uplift_targeting(
        _economics_frame(),
        EconomicsConfig(communication_cost=0, profit_per_conversion=100),
        threshold=0.05,
    )

    assert result.marketing_cost == 0
    assert result.incremental_profit == pytest.approx(30)
    assert result.roi is None


def test_threshold_optimization_finds_maximum_profit() -> None:
    optimization = optimize_targeting_threshold(
        _economics_frame(),
        EconomicsConfig(communication_cost=2, profit_per_conversion=100),
    )

    assert optimization.best.customers_targeted == 2
    assert math.isclose(optimization.best.incremental_profit, 26)
    assert optimization.scenarios["incremental_profit"].max() == pytest.approx(26)
    assert optimization.scenarios["customers_targeted"].min() == 0


def test_model_selection_uses_holdout_qini() -> None:
    comparison = pl.DataFrame(
        {
            "model": ["s_learner", "t_learner"],
            "qini_coefficient": [1.5, 2.5],
        }
    )

    assert select_model_by_qini(comparison) == "t_learner"


def test_full_business_optimization_materializes_outputs(tmp_path: Path) -> None:
    result = optimize_business_targeting(
        _customer_scores(),
        EconomicsConfig(communication_cost=2, profit_per_conversion=100),
        model="t_learner",
        targeting_threshold=0.05,
    )
    outputs = materialize_business_optimization(result, tmp_path / "business")

    assert set(outputs) == {
        "customer_targeting",
        "strategy_comparison",
        "threshold_optimization",
    }
    assert all(path.exists() for path in outputs.values())
    assert result.customer_targeting["is_targeted"].sum() == 1
    assert result.strategy_comparison["scenario_analysis"].all()
    assert result.threshold_optimization.scenarios.height > 1
    assert pl.read_parquet(outputs["customer_targeting"]).height == 4


def test_customer_type_rejects_inconsistent_uplift() -> None:
    frame = pl.DataFrame(
        {
            "probability_control": [0.2],
            "probability_treatment": [0.4],
            "predicted_uplift": [0.5],
        }
    )
    with pytest.raises(TargetingInputError):
        assign_customer_types(frame)


@pytest.mark.parametrize(
    "config",
    [
        {"communication_cost": -1, "profit_per_conversion": 100},
        {"communication_cost": 1, "profit_per_conversion": -100},
    ],
)
def test_economics_rejects_negative_assumptions(config: dict[str, float]) -> None:
    with pytest.raises(EconomicsInputError):
        EconomicsConfig(**config)
