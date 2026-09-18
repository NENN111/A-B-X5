"""Построение leakage-safe клиентской витрины признаков."""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import timedelta
from pathlib import Path
from uuid import uuid4

import polars as pl

from src.config.settings import get_settings
from src.features.rfm import add_rfm_features
from src.features.schema import (
    FeatureConfigError,
    FeatureMartConfig,
    load_feature_config,
)

LOGGER = logging.getLogger(__name__)


class FeatureSchemaError(ValueError):
    """Ошибка несоответствия фактической схемы настроенному mapping."""


@dataclass(frozen=True, slots=True)
class FeatureMartResult:
    """Результат материализации клиентской витрины."""

    output_path: Path
    metadata_path: Path
    customer_count: int
    feature_count: int
    eligible_purchase_rows: int
    excluded_purchase_rows: int


def _collect(frame: pl.LazyFrame) -> pl.DataFrame:
    """Выполнить lazy-план в streaming-режиме."""
    return frame.collect(engine="streaming")


def _require_columns(
    entity: str,
    frame: pl.LazyFrame,
    columns: Sequence[str | None],
) -> None:
    """Проверить наличие всех настроенных колонок сущности."""
    actual = set(frame.collect_schema().names())
    required = {column for column in columns if column is not None}
    missing = sorted(required - actual)
    if missing:
        raise FeatureSchemaError(
            f"В сущности {entity} отсутствуют колонки из mapping: {missing}. "
            f"Фактические колонки: {sorted(actual)}"
        )


def _assert_unique(frame: pl.LazyFrame, column: str, entity: str) -> None:
    """Проверить непустой уникальный ключ сущности."""
    stats = _collect(
        frame.select(
            pl.len().alias("rows"),
            pl.col(column).n_unique().alias("unique"),
            pl.col(column).null_count().alias("nulls"),
        )
    ).row(0, named=True)
    if stats["nulls"]:
        raise FeatureSchemaError(f"Ключ {entity}.{column} содержит null")
    if stats["rows"] != stats["unique"]:
        raise FeatureSchemaError(f"Ключ {entity}.{column} не уникален")


def _assert_no_nulls(
    frame: pl.LazyFrame,
    columns: Sequence[str],
    entity: str,
) -> None:
    """Запретить пропуски в обязательных полях вычислений."""
    counts = _collect(
        frame.select(pl.col(column).null_count().alias(column) for column in columns)
    ).row(0, named=True)
    invalid = {column: count for column, count in counts.items() if count}
    if invalid:
        raise FeatureSchemaError(
            f"Обязательные поля сущности {entity} содержат null: {invalid}"
        )


def _datetime_expression(
    frame: pl.LazyFrame,
    column: str,
    datetime_format: str | None,
) -> pl.Expr:
    """Привести настроенную временную колонку к Datetime."""
    dtype = frame.collect_schema()[column]
    value = pl.col(column)
    if dtype == pl.String:
        return value.str.to_datetime(format=datetime_format, strict=True)
    if isinstance(dtype, pl.Datetime):
        return value
    if dtype == pl.Date:
        return value.cast(pl.Datetime)
    raise FeatureSchemaError(
        f"Колонка времени {column} имеет неподдерживаемый тип {dtype}"
    )


def _prepare_clients(
    clients: pl.LazyFrame,
    config: FeatureMartConfig,
) -> pl.LazyFrame:
    """Выбрать демографические признаки с каноническими именами."""
    mapping = config.clients
    _require_columns(
        "clients",
        clients,
        (mapping.client_id, mapping.age, mapping.birth_date, mapping.gender),
    )
    _assert_unique(clients, mapping.client_id, "clients")
    expressions = [pl.col(mapping.client_id).alias("client_id")]
    if mapping.age:
        expressions.append(pl.col(mapping.age).cast(pl.Float64).alias("age"))
    elif mapping.birth_date:
        birth_dtype = clients.collect_schema()[mapping.birth_date]
        birth_date = pl.col(mapping.birth_date)
        if birth_dtype == pl.String:
            birth_date = birth_date.str.to_date(strict=True)
        elif birth_dtype.is_temporal():
            birth_date = birth_date.cast(pl.Date)
        else:
            raise FeatureSchemaError(
                f"Колонка даты рождения {mapping.birth_date} имеет тип {birth_dtype}"
            )
        age = (
            (pl.lit(config.experiment_start.date()) - birth_date)
            .dt.total_days()
            .truediv(365.2425)
            .floor()
            .cast(pl.Int16)
        )
        expressions.append(age.alias("age"))
    if mapping.gender:
        expressions.append(pl.col(mapping.gender).cast(pl.String).alias("gender"))
    return clients.select(expressions)


def _prepare_purchases(
    purchases: pl.LazyFrame,
    config: FeatureMartConfig,
) -> tuple[pl.LazyFrame, pl.LazyFrame]:
    """Нормализовать роли колонок и применить строго pre-treatment окно."""
    mapping = config.purchases
    _require_columns(
        "purchases",
        purchases,
        (
            mapping.client_id,
            mapping.transaction_id,
            mapping.transaction_datetime,
            mapping.amount,
            mapping.product_id,
            mapping.regular_amount,
        ),
    )
    _assert_no_nulls(
        purchases,
        (
            mapping.client_id,
            mapping.transaction_id,
            mapping.transaction_datetime,
            mapping.amount,
        ),
        "purchases",
    )
    schema = purchases.collect_schema()
    if not schema[mapping.amount].is_numeric():
        raise FeatureSchemaError(
            f"Колонка суммы {mapping.amount} должна быть числовой, "
            f"получен {schema[mapping.amount]}"
        )
    expressions = [
        pl.col(mapping.client_id).alias("client_id"),
        pl.col(mapping.transaction_id).alias("transaction_id"),
        _datetime_expression(
            purchases,
            mapping.transaction_datetime,
            mapping.datetime_format,
        ).alias("purchase_datetime"),
        pl.col(mapping.amount).cast(pl.Float64).alias("amount"),
    ]
    if mapping.product_id:
        expressions.append(pl.col(mapping.product_id).alias("product_id"))
    if mapping.regular_amount:
        if not schema[mapping.regular_amount].is_numeric():
            raise FeatureSchemaError(
                f"Колонка {mapping.regular_amount} должна быть числовой"
            )
        expressions.append(
            pl.col(mapping.regular_amount).cast(pl.Float64).alias("regular_amount")
        )
    normalized = purchases.select(expressions)
    normalized_datetime = normalized.collect_schema()["purchase_datetime"]
    source_timezone = getattr(normalized_datetime, "time_zone", None)
    cutoff_has_timezone = config.experiment_start.utcoffset() is not None
    if bool(source_timezone) != cutoff_has_timezone:
        raise FeatureSchemaError(
            "Timezone experiment_start должна соответствовать "
            f"purchases.{mapping.transaction_datetime}: "
            f"cutoff={config.experiment_start.isoformat()}, "
            f"dtype={normalized_datetime}"
        )
    lookback_start = config.experiment_start - timedelta(days=config.lookback_days)
    eligible = normalized.filter(
        pl.col("purchase_datetime").is_between(
            pl.lit(lookback_start),
            pl.lit(config.experiment_start),
            closed="left",
        )
    )
    return normalized, eligible


def _validate_customer_links(
    clients: pl.LazyFrame,
    purchases: pl.LazyFrame,
    *,
    strict: bool,
) -> None:
    """Проверить ссылки покупок на клиентов."""
    if not strict:
        return
    orphan_count = _collect(
        purchases.select("client_id")
        .unique()
        .join(clients.select("client_id"), on="client_id", how="anti")
        .select(pl.len().alias("count"))
    ).item()
    if orphan_count:
        raise FeatureSchemaError(
            f"Найдено client_id покупок без записи в clients: {orphan_count}"
        )


def _transaction_features(
    purchases: pl.LazyFrame,
    config: FeatureMartConfig,
) -> pl.LazyFrame:
    """Рассчитать monetary, frequency и recency на уровне транзакций."""
    checks = purchases.group_by("client_id", "transaction_id").agg(
        pl.col("amount").sum().alias("transaction_spend"),
        pl.col("purchase_datetime").max().alias("purchase_datetime"),
    )
    start = config.experiment_start
    return (
        checks.group_by("client_id")
        .agg(
            pl.col("transaction_spend").sum().alias("total_spend"),
            pl.col("transaction_spend").mean().alias("avg_check"),
            pl.col("transaction_spend").median().alias("median_check"),
            pl.col("transaction_spend").max().alias("max_check"),
            pl.len().alias("transactions_count"),
            pl.col("purchase_datetime")
            .filter(pl.col("purchase_datetime") >= pl.lit(start - timedelta(days=30)))
            .count()
            .alias("purchases_30d"),
            pl.col("purchase_datetime")
            .filter(pl.col("purchase_datetime") >= pl.lit(start - timedelta(days=60)))
            .count()
            .alias("purchases_60d"),
            pl.col("purchase_datetime")
            .filter(pl.col("purchase_datetime") >= pl.lit(start - timedelta(days=90)))
            .count()
            .alias("purchases_90d"),
            pl.col("purchase_datetime").max().alias("last_purchase_datetime"),
        )
        .with_columns(
            (
                pl.col("transactions_count")
                .cast(pl.Float64)
                .truediv(config.lookback_days)
                .mul(30)
            ).alias("purchase_frequency"),
            (pl.lit(start) - pl.col("last_purchase_datetime"))
            .dt.total_days()
            .alias("days_since_last_purchase"),
        )
        .drop("last_purchase_datetime")
    )


def _product_features(
    purchases: pl.LazyFrame,
    products: pl.LazyFrame | None,
    config: FeatureMartConfig,
) -> pl.LazyFrame | None:
    """Рассчитать товарное и категорийное поведение при наличии mapping."""
    if config.purchases.product_id is None:
        return None
    base = purchases.group_by("client_id").agg(
        pl.col("product_id").n_unique().alias("unique_products")
    )
    if config.products is None or products is None:
        return base

    mapping = config.products
    _require_columns("products", products, (mapping.product_id, mapping.category))
    _assert_unique(products, mapping.product_id, "products")
    normalized_products = products.select(
        pl.col(mapping.product_id).alias("product_id"),
        pl.col(mapping.category).cast(pl.String).alias("category"),
    )
    if config.strict_referential_integrity:
        orphan_count = _collect(
            purchases.select("product_id")
            .unique()
            .join(normalized_products.select("product_id"), on="product_id", how="anti")
            .select(pl.len().alias("count"))
        ).item()
        if orphan_count:
            raise FeatureSchemaError(
                f"Найдено product_id покупок без записи в products: {orphan_count}"
            )
    category_spend = (
        purchases.join(normalized_products, on="product_id", how="left")
        .filter(pl.col("category").is_not_null())
        .group_by("client_id", "category")
        .agg(pl.col("amount").sum().alias("category_spend"))
    )
    category_features = (
        category_spend.sort(
            ["client_id", "category_spend", "category"],
            descending=[False, True, False],
        )
        .group_by("client_id", maintain_order=True)
        .agg(
            pl.col("category").n_unique().alias("unique_categories"),
            pl.col("category").first().alias("favorite_category"),
        )
    )
    return base.join(category_features, on="client_id", how="left")


def _discount_features(
    purchases: pl.LazyFrame,
    config: FeatureMartConfig,
) -> pl.LazyFrame | None:
    """Рассчитать среднюю ставку скидки и долю покупок со скидкой."""
    if config.purchases.regular_amount is None:
        return None
    valid_regular = pl.col("regular_amount") > 0
    discount_rate = (
        (pl.col("regular_amount") - pl.col("amount"))
        .truediv(pl.col("regular_amount"))
        .filter(valid_regular)
    )
    discounted = (
        (pl.col("amount") < pl.col("regular_amount"))
        .cast(pl.Float64)
        .filter(valid_regular)
    )
    return purchases.group_by("client_id").agg(
        discount_rate.mean().alias("avg_discount"),
        discounted.mean().alias("discounted_purchase_share"),
    )


def _attach_experiment(
    mart: pl.LazyFrame,
    experiment: pl.LazyFrame | None,
    config: FeatureMartConfig,
) -> pl.LazyFrame:
    """Присоединить treatment и target после завершения расчёта признаков."""
    if config.experiment is None:
        return mart
    if experiment is None:
        raise FeatureSchemaError(
            "В mapping настроен experiment, но uplift_train не передан"
        )
    mapping = config.experiment
    _require_columns(
        "uplift_train",
        experiment,
        (mapping.client_id, mapping.treatment, mapping.target),
    )
    _assert_unique(experiment, mapping.client_id, "uplift_train")
    labels = experiment.select(
        pl.col(mapping.client_id).alias("client_id"),
        pl.col(mapping.treatment).alias("treatment"),
        pl.col(mapping.target).alias("target"),
    )
    feature_columns = [
        column for column in mart.collect_schema().names() if column != "client_id"
    ]
    return labels.join(mart, on="client_id", how="left").select(
        "client_id",
        *feature_columns,
        "treatment",
        "target",
    )


def build_customer_feature_mart(
    frames: Mapping[str, pl.LazyFrame],
    config: FeatureMartConfig,
) -> pl.LazyFrame:
    """Построить клиентскую витрину только из pre-treatment информации."""
    missing_entities = {"clients", "purchases"} - set(frames)
    if missing_entities:
        raise FeatureSchemaError(
            f"Не переданы обязательные сущности: {sorted(missing_entities)}"
        )
    clients = _prepare_clients(frames["clients"], config)
    _, purchases = _prepare_purchases(frames["purchases"], config)
    _validate_customer_links(
        clients,
        purchases,
        strict=config.strict_referential_integrity,
    )

    mart = clients.join(
        _transaction_features(purchases, config),
        on="client_id",
        how="left",
    )
    product_features = _product_features(purchases, frames.get("products"), config)
    if product_features is not None:
        mart = mart.join(product_features, on="client_id", how="left")
    discount_features = _discount_features(purchases, config)
    if discount_features is not None:
        mart = mart.join(discount_features, on="client_id", how="left")

    zero_fill_columns = [
        "total_spend",
        "transactions_count",
        "purchase_frequency",
        "purchases_30d",
        "purchases_60d",
        "purchases_90d",
        "unique_products",
        "unique_categories",
    ]
    existing = set(mart.collect_schema().names())
    mart = mart.with_columns(
        pl.col(column).fill_null(0)
        for column in zero_fill_columns
        if column in existing
    )
    mart = add_rfm_features(mart).with_columns(
        pl.col("monetary_score").alias("spending_quantile"),
        pl.col("frequency_score").alias("purchase_frequency_quantile"),
    )
    return _attach_experiment(mart, frames.get("uplift_train"), config)


def scan_clean_dataset(
    input_dir: Path,
    config: FeatureMartConfig,
) -> dict[str, pl.LazyFrame]:
    """Открыть необходимые результаты Stage 2 как LazyFrame."""
    entities = {"clients", "purchases"}
    if config.products:
        entities.add("products")
    if config.experiment:
        entities.add("uplift_train")
    missing = [
        entity for entity in entities if not (input_dir / f"{entity}.parquet").exists()
    ]
    if missing:
        raise FeatureConfigError(
            f"В {input_dir} отсутствуют Parquet Stage 2: {sorted(missing)}. "
            "Сначала выполните python -m src.data.preprocessing"
        )
    return {
        entity: pl.scan_parquet(input_dir / f"{entity}.parquet") for entity in entities
    }


def materialize_customer_feature_mart(
    frames: Mapping[str, pl.LazyFrame],
    config: FeatureMartConfig,
    output_path: Path,
) -> FeatureMartResult:
    """Атомарно записать feature mart и metadata с границами окна."""
    normalized, eligible = _prepare_purchases(frames["purchases"], config)
    counts = _collect(
        pl.concat(
            [
                normalized.select(pl.len().alias("count")).with_columns(
                    pl.lit("all").alias("kind")
                ),
                eligible.select(pl.len().alias("count")).with_columns(
                    pl.lit("eligible").alias("kind")
                ),
            ],
            how="vertical",
        )
    )
    count_by_kind = dict(counts.select("kind", "count").iter_rows())
    mart = build_customer_feature_mart(frames, config)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(f".{uuid4().hex}.tmp.parquet")
    try:
        mart.sink_parquet(temporary_path, mkdir=True)
        schema = pl.scan_parquet(temporary_path).collect_schema()
        customer_count = int(
            _collect(
                pl.scan_parquet(temporary_path).select(pl.len().alias("count"))
            ).item()
        )
        temporary_path.replace(output_path)
    finally:
        temporary_path.unlink(missing_ok=True)

    metadata_path = output_path.with_suffix(".metadata.json")
    config_payload = asdict(config)
    config_payload["experiment_start"] = config.experiment_start.isoformat()
    metadata = {
        "experiment_start": config.experiment_start.isoformat(),
        "lookback_start": (
            config.experiment_start - timedelta(days=config.lookback_days)
        ).isoformat(),
        "lookback_days": config.lookback_days,
        "customer_count": customer_count,
        "feature_count": len(
            [
                column
                for column in schema
                if column not in {"client_id", "treatment", "target"}
            ]
        ),
        "eligible_purchase_rows": int(count_by_kind["eligible"]),
        "excluded_purchase_rows": int(count_by_kind["all"] - count_by_kind["eligible"]),
        "columns": {column: str(dtype) for column, dtype in schema.items()},
        "mapping": config_payload,
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    LOGGER.info("Клиентская витрина записана: %s", output_path)
    return FeatureMartResult(
        output_path=output_path,
        metadata_path=metadata_path,
        customer_count=customer_count,
        feature_count=len(
            [
                column
                for column in schema
                if column not in {"client_id", "treatment", "target"}
            ]
        ),
        eligible_purchase_rows=int(count_by_kind["eligible"]),
        excluded_purchase_rows=int(count_by_kind["all"] - count_by_kind["eligible"]),
    )


def main() -> None:
    """Запустить построение customer feature mart из командной строки."""
    parser = argparse.ArgumentParser(
        description="Построить leakage-safe клиентскую витрину признаков."
    )
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--output-path", type=Path)
    args = parser.parse_args()
    settings = get_settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        config = load_feature_config(args.mapping)
        input_dir = args.input_dir or settings.interim_data_dir / "clean"
        output_path = (
            args.output_path
            or settings.processed_data_dir / "customer_features.parquet"
        )
        frames = scan_clean_dataset(input_dir, config)
        result = materialize_customer_feature_mart(frames, config, output_path)
    except (FeatureConfigError, FeatureSchemaError) as error:
        parser.error(str(error))
    print(
        f"Stage 3 завершён: {result.customer_count} клиентов, "
        f"витрина {result.output_path}"
    )


if __name__ == "__main__":
    main()
