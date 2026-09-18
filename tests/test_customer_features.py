"""Тесты leakage-safe клиентской витрины."""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime

import polars as pl
import pytest

from src.features.customer_features import (
    FeatureSchemaError,
    build_customer_feature_mart,
    materialize_customer_feature_mart,
    scan_clean_dataset,
)
from src.features.schema import (
    ClientColumns,
    ExperimentColumns,
    FeatureConfigError,
    FeatureMartConfig,
    ProductColumns,
    PurchaseColumns,
)


def _config() -> FeatureMartConfig:
    return FeatureMartConfig(
        experiment_start=datetime(2024, 1, 1, tzinfo=UTC),
        lookback_days=365,
        clients=ClientColumns(
            client_id="customer_key",
            age="age_years",
            gender="sex",
        ),
        purchases=PurchaseColumns(
            client_id="customer_key",
            transaction_id="receipt_key",
            transaction_datetime="purchased_at",
            amount="paid_amount",
            product_id="sku_key",
            regular_amount="regular_amount",
        ),
        products=ProductColumns(
            product_id="sku_key",
            category="category_name",
        ),
        experiment=ExperimentColumns(
            client_id="customer_key",
            treatment="is_treatment",
            target="converted",
        ),
    )


def _frames() -> dict[str, pl.LazyFrame]:
    return {
        "clients": pl.DataFrame(
            {
                "customer_key": [1, 2],
                "age_years": [30, 40],
                "sex": ["F", "M"],
            }
        ).lazy(),
        "purchases": pl.DataFrame(
            {
                "customer_key": [1, 1, 1, 1, 2],
                "receipt_key": ["t1", "t1", "t2", "post", "old"],
                "purchased_at": [
                    datetime(2023, 12, 20, tzinfo=UTC),
                    datetime(2023, 12, 20, tzinfo=UTC),
                    datetime(2023, 11, 15, tzinfo=UTC),
                    datetime(2024, 1, 2, tzinfo=UTC),
                    datetime(2022, 12, 1, tzinfo=UTC),
                ],
                "paid_amount": [10.0, 20.0, 30.0, 100.0, 50.0],
                "regular_amount": [20.0, 20.0, 40.0, 100.0, 50.0],
                "sku_key": ["p1", "p2", "p1", "p2", "p1"],
            }
        )
        .with_columns(pl.col("purchased_at").dt.replace_time_zone("UTC"))
        .lazy(),
        "products": pl.DataFrame(
            {
                "sku_key": ["p1", "p2"],
                "category_name": ["A", "B"],
            }
        ).lazy(),
        "uplift_train": pl.DataFrame(
            {
                "customer_key": [1, 2],
                "is_treatment": [1, 0],
                "converted": [1, 0],
            }
        ).lazy(),
    }


def test_feature_mart_uses_only_pre_treatment_window() -> None:
    result = (
        build_customer_feature_mart(_frames(), _config()).collect().sort("client_id")
    )

    first = result.row(0, named=True)
    assert first["age"] == 30.0
    assert first["gender"] == "F"
    assert first["total_spend"] == 60.0
    assert first["transactions_count"] == 2
    assert first["avg_check"] == 30.0
    assert first["median_check"] == 30.0
    assert first["max_check"] == 30.0
    assert first["purchases_30d"] == 1
    assert first["purchases_60d"] == 2
    assert first["purchases_90d"] == 2
    assert first["days_since_last_purchase"] == 12
    assert first["unique_products"] == 2
    assert first["unique_categories"] == 2
    assert first["favorite_category"] == "A"
    assert math.isclose(first["avg_discount"], 0.25)
    assert math.isclose(first["discounted_purchase_share"], 2 / 3)
    assert first["treatment"] == 1
    assert first["target"] == 1

    second = result.row(1, named=True)
    assert second["total_spend"] == 0.0
    assert second["transactions_count"] == 0
    assert second["days_since_last_purchase"] is None
    assert second["rfm_segment"] is None


def test_materialization_records_excluded_rows_and_mapping(tmp_path) -> None:
    output_path = tmp_path / "customer_features.parquet"

    result = materialize_customer_feature_mart(
        _frames(),
        _config(),
        output_path,
    )

    assert result.customer_count == 2
    assert result.eligible_purchase_rows == 3
    assert result.excluded_purchase_rows == 2
    assert output_path.exists()
    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert metadata["experiment_start"] == "2024-01-01T00:00:00+00:00"
    assert metadata["lookback_start"] == "2023-01-01T00:00:00+00:00"
    assert metadata["mapping"]["purchases"]["amount"] == "paid_amount"


def test_missing_mapped_column_raises_clear_error() -> None:
    frames = _frames()
    frames["clients"] = pl.DataFrame({"wrong_id": [1, 2]}).lazy()

    with pytest.raises(FeatureSchemaError, match="customer_key"):
        build_customer_feature_mart(frames, _config())


def test_duplicate_customer_key_is_rejected() -> None:
    frames = _frames()
    frames["clients"] = pl.DataFrame(
        {
            "customer_key": [1, 1],
            "age_years": [30, 30],
            "sex": ["F", "F"],
        }
    ).lazy()

    with pytest.raises(FeatureSchemaError, match="не уникален"):
        build_customer_feature_mart(frames, _config())


def test_orphan_purchase_customer_is_rejected() -> None:
    frames = _frames()
    frames["purchases"] = pl.concat(
        [
            frames["purchases"],
            pl.DataFrame(
                {
                    "customer_key": [999],
                    "receipt_key": ["orphan"],
                    "purchased_at": [datetime(2023, 12, 30, tzinfo=UTC)],
                    "paid_amount": [10.0],
                    "regular_amount": [10.0],
                    "sku_key": ["p1"],
                }
            )
            .with_columns(pl.col("purchased_at").dt.replace_time_zone("UTC"))
            .lazy(),
        ]
    )

    with pytest.raises(FeatureSchemaError, match="без записи в clients"):
        build_customer_feature_mart(frames, _config())


def test_timezone_mismatch_is_rejected() -> None:
    config = _config()
    naive_config = FeatureMartConfig(
        experiment_start=datetime.fromisoformat("2024-01-01T00:00:00"),
        clients=config.clients,
        purchases=config.purchases,
        products=config.products,
        experiment=config.experiment,
    )

    with pytest.raises(FeatureSchemaError, match="Timezone"):
        build_customer_feature_mart(_frames(), naive_config)


def test_scan_clean_dataset_reports_missing_stage2_outputs(tmp_path) -> None:
    with pytest.raises(FeatureConfigError, match="Сначала выполните"):
        scan_clean_dataset(tmp_path, _config())
