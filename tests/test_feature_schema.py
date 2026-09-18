"""Тесты конфигурации customer feature mart."""

from __future__ import annotations

import json

import pytest

from src.features.schema import FeatureConfigError, load_feature_config


def test_load_feature_config_requires_real_column_mapping(tmp_path) -> None:
    path = tmp_path / "mapping.json"
    path.write_text(
        json.dumps(
            {
                "experiment_start": "2024-01-01T00:00:00",
                "clients": {"client_id": None},
                "purchases": {},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(FeatureConfigError, match="clients.client_id"):
        load_feature_config(path)


def test_load_feature_config_allows_product_id_without_category_mapping(
    tmp_path,
) -> None:
    path = tmp_path / "mapping.json"
    path.write_text(
        json.dumps(
            {
                "experiment_start": "2024-01-01T00:00:00",
                "clients": {"client_id": "customer"},
                "purchases": {
                    "client_id": "customer",
                    "transaction_id": "receipt",
                    "transaction_datetime": "purchased_at",
                    "amount": "paid",
                    "product_id": "sku",
                },
            }
        ),
        encoding="utf-8",
    )

    config = load_feature_config(path)

    assert config.purchases.product_id == "sku"
    assert config.products is None
