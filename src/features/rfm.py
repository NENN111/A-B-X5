"""Расчёт воспроизводимых RFM-оценок."""

from __future__ import annotations

import polars as pl


def _quintile_score(column: str, *, higher_is_better: bool) -> pl.Expr:
    """Построить score 1–5 по percentile rank с устойчивостью к малым выборкам."""
    value = pl.col(column)
    non_null_count = value.count()
    rank = value.rank(
        method="average",
        descending=not higher_is_better,
    )
    score = (
        pl.when(value.is_null())
        .then(None)
        .when(non_null_count <= 1)
        .then(3)
        .otherwise((((rank - 1) * 4 / (non_null_count - 1)) + 1).round(0))
    )
    return score.cast(pl.Int8)


def add_rfm_features(frame: pl.LazyFrame) -> pl.LazyFrame:
    """Добавить RFM scores и стабильный код сегмента к клиентской витрине."""
    scored = frame.with_columns(
        _quintile_score(
            "days_since_last_purchase",
            higher_is_better=False,
        ).alias("recency_score"),
        _quintile_score(
            "transactions_count",
            higher_is_better=True,
        ).alias("frequency_score"),
        _quintile_score(
            "total_spend",
            higher_is_better=True,
        ).alias("monetary_score"),
    )
    complete_rfm = pl.all_horizontal(
        pl.col("recency_score").is_not_null(),
        pl.col("frequency_score").is_not_null(),
        pl.col("monetary_score").is_not_null(),
    )
    return scored.with_columns(
        pl.when(complete_rfm)
        .then(
            pl.concat_str(
                pl.lit("R"),
                pl.col("recency_score"),
                pl.lit("F"),
                pl.col("frequency_score"),
                pl.lit("M"),
                pl.col("monetary_score"),
            )
        )
        .otherwise(None)
        .alias("rfm_segment")
    )
