"""Тесты PostgreSQL DDL, нормализации и загрузочных контрактов."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import polars as pl
import pytest

from src.data.warehouse import (
    WarehouseInputError,
    apply_sql_files,
    build_upsert_sql,
    customer_dimension,
    experiment_fact,
    load_warehouse,
    normalize_products,
    normalize_purchases,
    prediction_fact,
    validation_summary,
)
from src.features.schema import (
    ClientColumns,
    FeatureMartConfig,
    ProductColumns,
    PurchaseColumns,
)


def _feature_config(*, with_products: bool = True) -> FeatureMartConfig:
    return FeatureMartConfig(
        experiment_start=datetime(2024, 2, 1, tzinfo=timezone.utc),
        clients=ClientColumns(client_id="client"),
        purchases=PurchaseColumns(
            client_id="client",
            transaction_id="transaction",
            transaction_datetime="purchased_at",
            amount="amount_raw",
            product_id="sku",
            regular_amount="regular_raw",
        ),
        products=(
            ProductColumns(product_id="sku", category="category_raw")
            if with_products
            else None
        ),
    )


def test_customer_dimension_and_experiment_fact_use_canonical_grain() -> None:
    mart = pl.DataFrame(
        {
            "client_id": [10, 20],
            "age": [30.0, 40.0],
            "rfm_segment": ["R5F5M5", "R1F1M1"],
            "treatment": [0, 1],
            "target": [0, 1],
        }
    )

    customers = customer_dimension(mart)
    experiment = experiment_fact(mart, experiment_key=7)

    assert customers.columns == ["customer_id", "age", "rfm_segment"]
    assert customers["customer_id"].to_list() == ["10", "20"]
    assert experiment["experiment_key"].to_list() == [7, 7]
    assert experiment["customer_id"].to_list() == ["10", "20"]


def test_customer_dimension_rejects_duplicate_key() -> None:
    with pytest.raises(WarehouseInputError):
        customer_dimension(pl.DataFrame({"client_id": [1, 1]}))


def test_purchase_and_date_normalization_respects_mapping() -> None:
    purchases = pl.DataFrame(
        {
            "client": [1, 2],
            "transaction": ["t1", "t2"],
            "purchased_at": [
                "2024-01-05T10:00:00+00:00",
                "2024-01-06T11:00:00+00:00",
            ],
            "amount_raw": [100.5, 200.0],
            "regular_raw": [110.0, 200.0],
            "sku": ["p1", "p2"],
        }
    )

    fact, dates = normalize_purchases(purchases, _feature_config())

    assert fact.columns == [
        "transaction_id",
        "customer_id",
        "product_id",
        "date_key",
        "purchase_datetime",
        "amount",
        "regular_amount",
    ]
    assert fact["date_key"].to_list() == [20240105, 20240106]
    assert dates["date_key"].to_list() == [20240105, 20240106]
    assert dates["day_of_week"].to_list() == [5, 6]
    assert dates["is_weekend"].to_list() == [False, True]


def test_product_normalization_rejects_duplicate_key() -> None:
    products = pl.DataFrame({"sku": ["p1", "p1"], "category_raw": ["A", "B"]})
    with pytest.raises(WarehouseInputError):
        normalize_products(products, _feature_config())


def test_prediction_fact_converts_wide_scores_to_model_rows() -> None:
    scores = pl.DataFrame(
        {
            "client_id": [1, 2],
            "s_learner_probability_control": [0.1, 0.2],
            "s_learner_probability_treatment": [0.2, 0.25],
            "s_learner_predicted_uplift": [0.1, 0.05],
            "t_learner_probability_control": [0.15, 0.2],
            "t_learner_probability_treatment": [0.3, 0.22],
            "t_learner_predicted_uplift": [0.15, 0.02],
        }
    )

    result = prediction_fact(scores, experiment_key=3)

    assert result.height == 4
    assert set(result["model_name"]) == {"s_learner", "t_learner"}
    assert result["experiment_key"].to_list() == [3] * 4


def test_validation_summaries_are_joined_by_explicit_mapping(tmp_path: Path) -> None:
    validation_dir = tmp_path / "validation"
    validation_dir.mkdir()
    pl.DataFrame(
        [
            {
                "observed_control": 500,
                "observed_treatment": 500,
                "expected_control": 500.0,
                "expected_treatment": 500.0,
                "chi_square_statistic": 0.0,
                "p_value": 1.0,
                "srm_detected": False,
            }
        ]
    ).write_parquet(validation_dir / "srm.parquet")
    pl.DataFrame(
        [
            {
                "simulations": 1000,
                "false_positive_rate": 0.05,
                "mean_p_value": 0.5,
                "ks_uniform_p_value": 0.4,
            }
        ]
    ).write_parquet(validation_dir / "aa_summary.parquet")
    pl.DataFrame(
        [
            {
                "iterations": 2000,
                "observed_uplift": 0.03,
                "mean_uplift": 0.031,
                "median_uplift": 0.03,
                "ci_lower": 0.01,
                "ci_upper": 0.05,
            }
        ]
    ).write_parquet(validation_dir / "bootstrap_summary.parquet")

    result = validation_summary(validation_dir, experiment_key=9)

    assert result is not None
    assert result.row(0, named=True)["experiment_key"] == 9
    assert result.row(0, named=True)["aa_false_positive_rate"] == 0.05
    assert result.row(0, named=True)["bootstrap_ci_upper"] == 0.05


def test_upsert_sql_is_parameterized_and_rejects_identifiers() -> None:
    statement = build_upsert_sql(
        "dim_customer", ["customer_id", "age"], ["customer_id"]
    )

    assert "VALUES (:customer_id, :age)" in statement
    assert "ON CONFLICT (customer_id) DO UPDATE" in statement
    with pytest.raises(WarehouseInputError):
        build_upsert_sql("dim_customer; DROP TABLE x", ["customer_id"], ["customer_id"])


def test_sql_files_define_required_grains_constraints_and_views() -> None:
    root = Path(__file__).resolve().parents[1]
    ddl = (root / "sql" / "ddl" / "01_analytics.sql").read_text(encoding="utf-8")
    views = (root / "sql" / "marts" / "01_power_bi_views.sql").read_text(
        encoding="utf-8"
    )
    required_tables = {
        "dim_customer",
        "dim_product",
        "dim_date",
        "dim_experiment",
        "fact_purchases",
        "fact_experiment",
        "experiment_results",
        "segment_experiment_results",
        "uplift_predictions",
        "uplift_deciles",
        "business_scenarios",
    }

    assert all(f"analytics.{table}" in ddl for table in required_tables)
    assert "PRIMARY KEY (experiment_key, customer_id)" in ddl
    assert "REFERENCES analytics.dim_customer(customer_id)" in ddl
    assert "CREATE INDEX IF NOT EXISTS" in ddl
    assert "vw_experiment_overview" in views
    assert "vw_business_strategy_comparison" in views


def test_apply_sql_files_executes_each_statement(tmp_path: Path) -> None:
    sql_path = tmp_path / "migration.sql"
    sql_path.write_text("SELECT 1; SELECT 2;", encoding="utf-8")

    class FakeConnection:
        def __init__(self) -> None:
            self.statements: list[str] = []

        def exec_driver_sql(self, statement: str) -> None:
            self.statements.append(statement)

    connection = FakeConnection()
    apply_sql_files(connection, [sql_path])  # type: ignore[arg-type]

    assert connection.statements == ["SELECT 1", "SELECT 2"]


def test_warehouse_orchestration_loads_available_artifacts(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    modeling = processed / "uplift_modeling"
    business = processed / "business_optimization"
    for directory in (processed, modeling, business):
        directory.mkdir(parents=True, exist_ok=True)

    pl.DataFrame(
        {
            "client_id": [1, 2, 3, 4],
            "age": [20.0, 30.0, 40.0, 50.0],
            "treatment": [0, 0, 1, 1],
            "target": [0, 1, 0, 1],
        }
    ).write_parquet(processed / "customer_features.parquet")
    score_data: dict[str, list[float] | list[int]] = {"client_id": [1, 2, 3, 4]}
    for model in ("s_learner", "t_learner"):
        score_data[f"{model}_probability_control"] = [0.1] * 4
        score_data[f"{model}_probability_treatment"] = [0.2] * 4
        score_data[f"{model}_predicted_uplift"] = [0.1] * 4
    pl.DataFrame(score_data).write_parquet(modeling / "uplift_customer_scores.parquet")
    scenario = {
        "selected_model": "s_learner",
        "strategy": "uplift_targeting",
        "customers_total": 4,
        "customers_targeted": 2,
        "target_share": 0.5,
        "targeting_threshold": 0.05,
        "communication_cost": 2.0,
        "profit_per_conversion": 100.0,
        "marketing_cost": 4.0,
        "incremental_conversions": 0.2,
        "incremental_value": 20.0,
        "incremental_profit": 16.0,
        "roi": 4.0,
        "scenario_analysis": True,
    }
    pl.DataFrame([scenario]).write_parquet(business / "business_scenarios.parquet")
    pl.DataFrame([scenario]).write_parquet(business / "threshold_optimization.parquet")

    class FakeResult:
        def scalar_one(self) -> int:
            return 11

    class FakeConnection:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def execute(self, statement: object, parameters: object = None) -> FakeResult:
            self.calls.append(str(statement))
            return FakeResult()

    class FakeBegin:
        def __init__(self, connection: FakeConnection) -> None:
            self.connection = connection

        def __enter__(self) -> FakeConnection:
            return self.connection

        def __exit__(self, *_: object) -> None:
            return None

    class FakeEngine:
        def __init__(self) -> None:
            self.connection = FakeConnection()

        def begin(self) -> FakeBegin:
            return FakeBegin(self.connection)

    engine = FakeEngine()
    result = load_warehouse(
        engine,  # type: ignore[arg-type]
        experiment_name="test_experiment",
        processed_dir=processed,
        clean_dir=tmp_path / "clean",
        apply_ddl=False,
    )

    assert result.experiment_key == 11
    assert result.row_counts["dim_customer"] == 4
    assert result.row_counts["fact_experiment"] == 4
    assert result.row_counts["uplift_predictions"] == 8
    assert result.row_counts["business_scenarios"] == 2
    assert any("ON CONFLICT" in call for call in engine.connection.calls)
