"""Тесты RFM-оценок."""

from __future__ import annotations

import polars as pl

from src.features.rfm import add_rfm_features


def test_rfm_scores_respect_direction_and_missing_recency() -> None:
    frame = pl.DataFrame(
        {
            "client_id": [1, 2, 3],
            "days_since_last_purchase": [1, 10, None],
            "transactions_count": [10, 1, 0],
            "total_spend": [100.0, 10.0, 0.0],
        }
    ).lazy()

    result = add_rfm_features(frame).collect().sort("client_id")

    first = result.row(0, named=True)
    assert first["recency_score"] == 5
    assert first["frequency_score"] == 5
    assert first["monetary_score"] == 5
    assert first["rfm_segment"] == "R5F5M5"

    second = result.row(1, named=True)
    assert second["recency_score"] == 1
    assert second["frequency_score"] == 3
    assert second["monetary_score"] == 3

    third = result.row(2, named=True)
    assert third["recency_score"] is None
    assert third["rfm_segment"] is None
