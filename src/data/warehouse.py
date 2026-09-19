"""Нормализация и идемпотентная загрузка аналитических витрин PostgreSQL."""

from __future__ import annotations

import argparse
import logging
import math
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl
from sqlalchemy import Connection, Engine, text

from src.config.settings import PROJECT_ROOT, get_settings
from src.data.database import create_db_engine
from src.features.schema import FeatureMartConfig, load_feature_config

LOGGER = logging.getLogger(__name__)
IDENTIFIER = re.compile(r"^[a-z_][a-z0-9_]*$")

CUSTOMER_COLUMNS = (
    "age",
    "gender",
    "total_spend",
    "avg_check",
    "median_check",
    "max_check",
    "transactions_count",
    "purchase_frequency",
    "days_since_last_purchase",
    "unique_products",
    "unique_categories",
    "favorite_category",
    "avg_discount",
    "discounted_purchase_share",
    "recency_score",
    "frequency_score",
    "monetary_score",
    "rfm_segment",
    "spending_quantile",
    "purchase_frequency_quantile",
)


class WarehouseInputError(ValueError):
    """Ошибка контракта аналитической витрины или SQL-загрузки."""


@dataclass(frozen=True, slots=True)
class WarehouseLoadResult:
    """Число загруженных строк по таблицам."""

    experiment_key: int
    row_counts: dict[str, int]


def _require_columns(frame: pl.DataFrame, columns: Iterable[str], entity: str) -> None:
    missing = set(columns) - set(frame.columns)
    if missing:
        raise WarehouseInputError(f"В {entity} отсутствуют колонки: {sorted(missing)}")


def customer_dimension(feature_mart: pl.DataFrame) -> pl.DataFrame:
    """Построить dim_customer из канонической клиентской feature mart."""
    _require_columns(feature_mart, ["client_id"], "customer feature mart")
    if (
        feature_mart["client_id"].null_count()
        or feature_mart["client_id"].n_unique() != feature_mart.height
    ):
        raise WarehouseInputError("client_id должен быть заполненным уникальным ключом")
    available = [
        column for column in CUSTOMER_COLUMNS if column in feature_mart.columns
    ]
    return feature_mart.select(
        pl.col("client_id").cast(pl.String).alias("customer_id"),
        *available,
    )


def experiment_fact(feature_mart: pl.DataFrame, experiment_key: int) -> pl.DataFrame:
    """Построить fact_experiment на уровне клиент × эксперимент."""
    _require_columns(
        feature_mart, ["client_id", "treatment", "target"], "customer feature mart"
    )
    invalid = feature_mart.filter(
        pl.col("treatment").is_null()
        | pl.col("target").is_null()
        | ~pl.col("treatment").is_in([0, 1])
        | ~pl.col("target").is_in([0, 1])
    )
    if invalid.height:
        raise WarehouseInputError("Treatment и target должны быть бинарными 0/1")
    return feature_mart.select(
        pl.lit(experiment_key).cast(pl.Int64).alias("experiment_key"),
        pl.col("client_id").cast(pl.String).alias("customer_id"),
        pl.col("treatment").cast(pl.Int8),
        pl.col("target").cast(pl.Int8),
    )


def _datetime_expression(frame: pl.DataFrame, config: FeatureMartConfig) -> pl.Expr:
    column = config.purchases.transaction_datetime
    dtype = frame.schema[column]
    expression = pl.col(column)
    if dtype == pl.String:
        datetime_format = config.purchases.datetime_format
        sample = (
            frame[column].drop_nulls().head(1).item()
            if frame[column].drop_nulls().len()
            else ""
        )
        has_timezone = bool(
            re.search(r"(Z|[+-]\d{2}:?\d{2})$", str(sample))
            or (
                datetime_format and ("%z" in datetime_format or "%Z" in datetime_format)
            )
        )
        return expression.str.to_datetime(
            format=datetime_format,
            time_zone="UTC" if has_timezone else None,
            strict=True,
        )
    if not dtype.is_temporal():
        raise WarehouseInputError(
            f"Колонка {column!r} должна быть строкой или датой/временем, получено {dtype}"
        )
    return expression.cast(pl.Datetime("us"))


def normalize_products(
    products: pl.DataFrame, config: FeatureMartConfig
) -> pl.DataFrame:
    """Нормализовать dim_product строго по пользовательскому mapping."""
    if config.products is None:
        raise WarehouseInputError("В mapping отсутствует секция products")
    mapping = config.products
    _require_columns(products, [mapping.product_id, mapping.category], "products")
    normalized = products.select(
        pl.col(mapping.product_id).cast(pl.String).alias("product_id"),
        pl.col(mapping.category).cast(pl.String).alias("category"),
    )
    if (
        normalized["product_id"].null_count()
        or normalized["product_id"].n_unique() != normalized.height
    ):
        raise WarehouseInputError(
            "product_id должен быть заполненным уникальным ключом"
        )
    return normalized


def normalize_purchases(
    purchases: pl.DataFrame, config: FeatureMartConfig
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Нормализовать fact_purchases и вывести dim_date из mapping."""
    mapping = config.purchases
    required = [
        mapping.client_id,
        mapping.transaction_id,
        mapping.transaction_datetime,
        mapping.amount,
    ]
    if mapping.product_id:
        required.append(mapping.product_id)
    if mapping.regular_amount:
        required.append(mapping.regular_amount)
    _require_columns(purchases, required, "purchases")
    timestamp = _datetime_expression(purchases, config)
    normalized = purchases.select(
        pl.col(mapping.transaction_id).cast(pl.String).alias("transaction_id"),
        pl.col(mapping.client_id).cast(pl.String).alias("customer_id"),
        (
            pl.col(mapping.product_id).cast(pl.String)
            if mapping.product_id
            else pl.lit(None, dtype=pl.String)
        ).alias("product_id"),
        timestamp.alias("purchase_datetime"),
        pl.col(mapping.amount).cast(pl.Float64, strict=True).alias("amount"),
        (
            pl.col(mapping.regular_amount).cast(pl.Float64, strict=True)
            if mapping.regular_amount
            else pl.lit(None, dtype=pl.Float64)
        ).alias("regular_amount"),
    ).with_columns(
        pl.col("purchase_datetime")
        .dt.strftime("%Y%m%d")
        .cast(pl.Int32)
        .alias("date_key")
    )
    if normalized.select(
        pl.any_horizontal(
            pl.col("transaction_id").is_null(),
            pl.col("customer_id").is_null(),
            pl.col("purchase_datetime").is_null(),
            pl.col("amount").is_null(),
        ).any()
    ).item():
        raise WarehouseInputError("Обязательные поля fact_purchases содержат null")
    dates = (
        normalized.select(pl.col("purchase_datetime").dt.date().alias("full_date"))
        .unique()
        .sort("full_date")
        .with_columns(
            pl.col("full_date").dt.strftime("%Y%m%d").cast(pl.Int32).alias("date_key"),
            pl.col("full_date").dt.year().cast(pl.Int16).alias("calendar_year"),
            pl.col("full_date").dt.quarter().cast(pl.Int8).alias("calendar_quarter"),
            pl.col("full_date").dt.month().cast(pl.Int8).alias("calendar_month"),
            pl.col("full_date").dt.strftime("%B").alias("month_name"),
            pl.col("full_date").dt.day().cast(pl.Int8).alias("day_of_month"),
            pl.col("full_date").dt.week().cast(pl.Int8).alias("iso_week"),
            pl.col("full_date").dt.weekday().cast(pl.Int8).alias("day_of_week"),
            pl.col("full_date").dt.strftime("%A").alias("day_name"),
            (pl.col("full_date").dt.weekday() >= 6).alias("is_weekend"),
        )
        .select(
            "date_key",
            "full_date",
            "calendar_year",
            "calendar_quarter",
            "calendar_month",
            "month_name",
            "day_of_month",
            "iso_week",
            "day_of_week",
            "day_name",
            "is_weekend",
        )
    )
    return normalized.select(
        "transaction_id",
        "customer_id",
        "product_id",
        "date_key",
        "purchase_datetime",
        "amount",
        "regular_amount",
    ), dates


def prediction_fact(scores: pl.DataFrame, experiment_key: int) -> pl.DataFrame:
    """Преобразовать wide S/T scores в нормализованный uplift_predictions."""
    _require_columns(scores, ["client_id"], "uplift customer scores")
    frames: list[pl.DataFrame] = []
    for model in ("s_learner", "t_learner"):
        columns = [
            f"{model}_probability_control",
            f"{model}_probability_treatment",
            f"{model}_predicted_uplift",
        ]
        _require_columns(scores, columns, "uplift customer scores")
        frames.append(
            scores.select(
                pl.lit(experiment_key).alias("experiment_key"),
                pl.col("client_id").cast(pl.String).alias("customer_id"),
                pl.lit(model).alias("model_name"),
                pl.col(columns[0]).alias("probability_control"),
                pl.col(columns[1]).alias("probability_treatment"),
                pl.col(columns[2]).alias("predicted_uplift"),
            )
        )
    return pl.concat(frames, how="vertical")


def _safe_identifier(value: str) -> str:
    if not IDENTIFIER.fullmatch(value):
        raise WarehouseInputError(f"Недопустимый SQL identifier: {value!r}")
    return value


def build_upsert_sql(
    table: str,
    columns: Sequence[str],
    conflict_columns: Sequence[str],
) -> str:
    """Построить безопасный PostgreSQL INSERT ... ON CONFLICT."""
    table = _safe_identifier(table)
    checked_columns = [_safe_identifier(column) for column in columns]
    checked_conflicts = [_safe_identifier(column) for column in conflict_columns]
    if not checked_columns or not checked_conflicts:
        raise WarehouseInputError("Для upsert нужны columns и conflict_columns")
    if not set(checked_conflicts) <= set(checked_columns):
        raise WarehouseInputError("Conflict columns должны входить в columns")
    insert_columns = ", ".join(checked_columns)
    values = ", ".join(f":{column}" for column in checked_columns)
    conflicts = ", ".join(checked_conflicts)
    updates = [column for column in checked_columns if column not in checked_conflicts]
    action = (
        "DO UPDATE SET "
        + ", ".join(f"{column} = EXCLUDED.{column}" for column in updates)
        if updates
        else "DO NOTHING"
    )
    return (
        f"INSERT INTO analytics.{table} ({insert_columns}) VALUES ({values}) "
        f"ON CONFLICT ({conflicts}) {action}"
    )


def _clean_value(value: Any) -> Any:
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def upsert_frame(
    connection: Connection,
    table: str,
    frame: pl.DataFrame,
    conflict_columns: Sequence[str],
    *,
    batch_size: int = 5_000,
) -> int:
    """Загрузить DataFrame батчами через параметризованный upsert."""
    if frame.height == 0:
        return 0
    statement = text(build_upsert_sql(table, frame.columns, conflict_columns))
    records = [
        {key: _clean_value(value) for key, value in row.items()}
        for row in frame.to_dicts()
    ]
    for start in range(0, len(records), batch_size):
        connection.execute(statement, records[start : start + batch_size])
    return len(records)


def replace_scope(
    connection: Connection,
    table: str,
    frame: pl.DataFrame,
    *,
    where_column: str | None = None,
    where_value: Any = None,
    batch_size: int = 5_000,
) -> int:
    """Транзакционно заменить таблицу или один experiment scope."""
    table = _safe_identifier(table)
    if where_column is None:
        connection.execute(text(f"DELETE FROM analytics.{table}"))
    else:
        where_column = _safe_identifier(where_column)
        connection.execute(
            text(f"DELETE FROM analytics.{table} WHERE {where_column} = :scope"),
            {"scope": where_value},
        )
    if frame.height == 0:
        return 0
    columns = [_safe_identifier(column) for column in frame.columns]
    statement = text(
        f"INSERT INTO analytics.{table} ({', '.join(columns)}) VALUES "
        f"({', '.join(f':{column}' for column in columns)})"
    )
    records = [
        {key: _clean_value(value) for key, value in row.items()}
        for row in frame.to_dicts()
    ]
    for start in range(0, len(records), batch_size):
        connection.execute(statement, records[start : start + batch_size])
    return len(records)


def apply_sql_files(connection: Connection, paths: Sequence[Path]) -> None:
    """Последовательно применить идемпотентные SQL-файлы без процедурных блоков."""
    for path in paths:
        if not path.exists():
            raise WarehouseInputError(f"SQL-файл не найден: {path}")
        statements = [
            part.strip() for part in path.read_text(encoding="utf-8").split(";")
        ]
        for statement in statements:
            if statement:
                connection.exec_driver_sql(statement)


def _with_experiment_key(
    frame: pl.DataFrame, experiment_key: int, rename: dict[str, str] | None = None
) -> pl.DataFrame:
    result = frame.rename(rename or {})
    return result.with_columns(pl.lit(experiment_key).alias("experiment_key"))


def _read_if_exists(path: Path) -> pl.DataFrame | None:
    return pl.read_parquet(path) if path.exists() else None


def validation_summary(
    validation_dir: Path, experiment_key: int
) -> pl.DataFrame | None:
    """Объединить SRM, A/A и bootstrap summaries в одну строку витрины."""
    srm = _read_if_exists(validation_dir / "srm.parquet")
    aa = _read_if_exists(validation_dir / "aa_summary.parquet")
    bootstrap = _read_if_exists(validation_dir / "bootstrap_summary.parquet")
    if srm is None or aa is None or bootstrap is None:
        return None
    srm_row = srm.row(0, named=True)
    aa_row = aa.row(0, named=True)
    bootstrap_row = bootstrap.row(0, named=True)
    return pl.DataFrame(
        [
            {
                "experiment_key": experiment_key,
                "observed_control": srm_row["observed_control"],
                "observed_treatment": srm_row["observed_treatment"],
                "expected_control": srm_row["expected_control"],
                "expected_treatment": srm_row["expected_treatment"],
                "srm_chi_square": srm_row["chi_square_statistic"],
                "srm_p_value": srm_row["p_value"],
                "srm_detected": srm_row["srm_detected"],
                "aa_simulations": aa_row["simulations"],
                "aa_false_positive_rate": aa_row["false_positive_rate"],
                "aa_mean_p_value": aa_row["mean_p_value"],
                "aa_ks_uniform_p_value": aa_row["ks_uniform_p_value"],
                "bootstrap_iterations": bootstrap_row["iterations"],
                "observed_uplift": bootstrap_row["observed_uplift"],
                "bootstrap_mean_uplift": bootstrap_row["mean_uplift"],
                "bootstrap_median_uplift": bootstrap_row["median_uplift"],
                "bootstrap_ci_lower": bootstrap_row["ci_lower"],
                "bootstrap_ci_upper": bootstrap_row["ci_upper"],
            }
        ]
    )


def _experiment_key(
    connection: Connection,
    name: str,
    start: datetime | None,
    expected_treatment_share: float,
) -> int:
    connection.execute(
        text(
            "INSERT INTO analytics.dim_experiment "
            "(experiment_name, experiment_start, expected_treatment_share) "
            "VALUES (:name, :start, :share) "
            "ON CONFLICT (experiment_name) DO UPDATE SET "
            "experiment_start = EXCLUDED.experiment_start, "
            "expected_treatment_share = EXCLUDED.expected_treatment_share"
        ),
        {"name": name, "start": start, "share": expected_treatment_share},
    )
    return int(
        connection.execute(
            text(
                "SELECT experiment_key FROM analytics.dim_experiment "
                "WHERE experiment_name = :name"
            ),
            {"name": name},
        ).scalar_one()
    )


def load_warehouse(
    engine: Engine,
    *,
    experiment_name: str,
    processed_dir: Path,
    clean_dir: Path,
    feature_config: FeatureMartConfig | None = None,
    expected_treatment_share: float = 0.5,
    scenario_name: str = "configured",
    apply_ddl: bool = True,
) -> WarehouseLoadResult:
    """Загрузить доступные канонические артефакты в одной транзакции."""
    feature_path = processed_dir / "customer_features.parquet"
    if not feature_path.exists():
        raise WarehouseInputError(
            f"Клиентская витрина не найдена: {feature_path}. Сначала материализуйте данные."
        )
    mart = pl.read_parquet(feature_path)
    row_counts: dict[str, int] = {}
    ddl_paths = sorted((PROJECT_ROOT / "sql" / "ddl").glob("*.sql"))
    view_paths = sorted((PROJECT_ROOT / "sql" / "marts").glob("*.sql"))
    with engine.begin() as connection:
        if apply_ddl:
            apply_sql_files(connection, [*ddl_paths, *view_paths])
        experiment_key = _experiment_key(
            connection,
            experiment_name,
            feature_config.experiment_start if feature_config else None,
            expected_treatment_share,
        )
        customers = customer_dimension(mart)
        row_counts["dim_customer"] = upsert_frame(
            connection, "dim_customer", customers, ["customer_id"]
        )
        row_counts["fact_experiment"] = replace_scope(
            connection,
            "fact_experiment",
            experiment_fact(mart, experiment_key),
            where_column="experiment_key",
            where_value=experiment_key,
        )
        if feature_config is not None:
            purchases_path = clean_dir / "purchases.parquet"
            if not purchases_path.exists():
                raise WarehouseInputError(
                    f"Не найден cleaned purchases: {purchases_path}"
                )
            if feature_config.products is not None:
                products_path = clean_dir / "products.parquet"
                if not products_path.exists():
                    raise WarehouseInputError(
                        f"Не найден cleaned products: {products_path}"
                    )
                products = normalize_products(
                    pl.read_parquet(products_path), feature_config
                )
                row_counts["dim_product"] = upsert_frame(
                    connection, "dim_product", products, ["product_id"]
                )
            purchases, dates = normalize_purchases(
                pl.read_parquet(purchases_path), feature_config
            )
            if feature_config.products is None and feature_config.purchases.product_id:
                derived_products = purchases.filter(
                    pl.col("product_id").is_not_null()
                ).select(
                    pl.col("product_id").unique(),
                    pl.lit(None, dtype=pl.String).alias("category"),
                )
                row_counts["dim_product"] = upsert_frame(
                    connection, "dim_product", derived_products, ["product_id"]
                )
            row_counts["dim_date"] = upsert_frame(
                connection, "dim_date", dates, ["date_key"]
            )
            row_counts["fact_purchases"] = replace_scope(
                connection, "fact_purchases", purchases
            )

        validation = validation_summary(
            processed_dir / "experiment_validation", experiment_key
        )
        if validation is not None:
            row_counts["experiment_validation_summary"] = upsert_frame(
                connection,
                "experiment_validation_summary",
                validation,
                ["experiment_key"],
            )
        artifacts: list[tuple[str, Path, dict[str, str], list[str]]] = [
            (
                "experiment_results",
                processed_dir / "experiment_results.parquet",
                {},
                ["experiment_key"],
            ),
            (
                "segment_experiment_results",
                processed_dir / "experiment_validation" / "segment_analysis.parquet",
                {},
                ["experiment_key", "dimension", "segment"],
            ),
            (
                "uplift_model_results",
                processed_dir / "uplift_modeling" / "uplift_model_comparison.parquet",
                {"model": "model_name"},
                ["experiment_key", "model_name"],
            ),
            (
                "uplift_deciles",
                processed_dir / "uplift_modeling" / "uplift_deciles.parquet",
                {"model": "model_name"},
                ["experiment_key", "model_name", "decile"],
            ),
        ]
        for table, path, rename, keys in artifacts:
            source = _read_if_exists(path)
            if source is None:
                continue
            frame = _with_experiment_key(source, experiment_key, rename)
            target_columns = _table_columns(table)
            frame = frame.select(
                column for column in target_columns if column in frame.columns
            )
            row_counts[table] = upsert_frame(connection, table, frame, keys)

        scores = _read_if_exists(
            processed_dir / "uplift_modeling" / "uplift_customer_scores.parquet"
        )
        if scores is not None:
            predictions = prediction_fact(scores, experiment_key)
            row_counts["uplift_predictions"] = replace_scope(
                connection,
                "uplift_predictions",
                predictions,
                where_column="experiment_key",
                where_value=experiment_key,
            )
        scenarios = _read_if_exists(
            processed_dir / "business_optimization" / "business_scenarios.parquet"
        )
        if scenarios is not None:
            scenarios = _with_experiment_key(
                scenarios, experiment_key, {"selected_model": "model_name"}
            ).with_columns(
                pl.concat_str(
                    pl.lit(scenario_name), pl.col("strategy"), separator=":"
                ).alias("scenario_name")
            )
            scenarios = scenarios.select(
                column
                for column in _table_columns("business_scenarios")
                if column in scenarios.columns
            )
            row_counts["business_scenarios"] = upsert_frame(
                connection,
                "business_scenarios",
                scenarios,
                ["experiment_key", "model_name", "scenario_name"],
            )
        threshold_scenarios = _read_if_exists(
            processed_dir / "business_optimization" / "threshold_optimization.parquet"
        )
        if threshold_scenarios is not None:
            threshold_scenarios = _with_experiment_key(
                threshold_scenarios,
                experiment_key,
                {"selected_model": "model_name"},
            ).with_columns(
                pl.concat_str(
                    pl.lit("optimized"),
                    pl.col("customers_targeted").cast(pl.String),
                    separator=":",
                ).alias("scenario_name")
            )
            threshold_scenarios = threshold_scenarios.select(
                column
                for column in _table_columns("business_scenarios")
                if column in threshold_scenarios.columns
            )
            loaded = upsert_frame(
                connection,
                "business_scenarios",
                threshold_scenarios,
                ["experiment_key", "model_name", "scenario_name"],
            )
            row_counts["business_scenarios"] = (
                row_counts.get("business_scenarios", 0) + loaded
            )
    return WarehouseLoadResult(experiment_key=experiment_key, row_counts=row_counts)


def _table_columns(table: str) -> tuple[str, ...]:
    """Разрешённые загрузочные поля таблиц с server-side timestamps."""
    columns = {
        "experiment_results": (
            "experiment_key",
            "control_size",
            "treatment_size",
            "control_conversions",
            "treatment_conversions",
            "control_cr",
            "treatment_cr",
            "absolute_uplift",
            "relative_uplift",
            "standard_error",
            "z_statistic",
            "p_value",
            "ci_lower",
            "ci_upper",
            "effect_size",
            "achieved_power",
            "required_total_size",
            "mde_absolute",
            "alpha",
            "statistically_significant",
        ),
        "segment_experiment_results": (
            "experiment_key",
            "dimension",
            "segment",
            "control_size",
            "treatment_size",
            "control_cr",
            "treatment_cr",
            "absolute_uplift",
            "relative_uplift",
            "ci_lower",
            "ci_upper",
            "p_value",
            "adjusted_p_value",
            "significant_adjusted",
        ),
        "uplift_model_results": (
            "experiment_key",
            "model_name",
            "train_customers",
            "holdout_customers",
            "auuc",
            "qini_coefficient",
            "overall_observed_uplift",
            "final_incremental_purchases",
            "random_state",
        ),
        "uplift_deciles": (
            "experiment_key",
            "model_name",
            "decile",
            "customers",
            "control_size",
            "treatment_size",
            "avg_predicted_uplift",
            "treatment_cr",
            "control_cr",
            "observed_uplift",
            "incremental_purchases",
            "cumulative_incremental_purchases",
        ),
        "business_scenarios": (
            "experiment_key",
            "model_name",
            "scenario_name",
            "strategy",
            "customers_total",
            "customers_targeted",
            "target_share",
            "targeting_threshold",
            "communication_cost",
            "profit_per_conversion",
            "marketing_cost",
            "incremental_conversions",
            "incremental_value",
            "incremental_profit",
            "roi",
            "scenario_analysis",
        ),
    }
    try:
        return columns[table]
    except KeyError as error:
        raise WarehouseInputError(
            f"Неизвестная аналитическая таблица: {table}"
        ) from error


def main() -> None:
    """Загрузить аналитические артефакты в PostgreSQL."""
    parser = argparse.ArgumentParser(description="Загрузка PostgreSQL marts.")
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--mapping", type=Path)
    parser.add_argument("--processed-dir", type=Path)
    parser.add_argument("--clean-dir", type=Path)
    parser.add_argument("--expected-treatment-share", type=float, default=0.5)
    parser.add_argument("--scenario-name", default="configured")
    parser.add_argument("--skip-ddl", action="store_true")
    args = parser.parse_args()
    settings = get_settings()
    try:
        result = load_warehouse(
            create_db_engine(settings),
            experiment_name=args.experiment_name,
            processed_dir=args.processed_dir or settings.processed_data_dir,
            clean_dir=args.clean_dir or settings.interim_data_dir / "clean",
            feature_config=load_feature_config(args.mapping) if args.mapping else None,
            expected_treatment_share=args.expected_treatment_share,
            scenario_name=args.scenario_name,
            apply_ddl=not args.skip_ddl,
        )
    except (WarehouseInputError, ValueError) as error:
        parser.error(str(error))
    print(
        f"PostgreSQL marts загружены: experiment_key={result.experiment_key}, "
        f"таблиц={len(result.row_counts)}"
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
