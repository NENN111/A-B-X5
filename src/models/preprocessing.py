"""Общая подготовка признаков для baseline uplift-моделей."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd
import polars as pl
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class UpliftDataError(ValueError):
    """Ошибка входной витрины uplift-модели."""


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    """Зафиксированное разделение признаков по типам."""

    columns: tuple[str, ...]
    numeric: tuple[str, ...]
    categorical: tuple[str, ...]
    temporal: tuple[str, ...]


def infer_feature_columns(
    frame: pl.DataFrame,
    *,
    treatment_column: str = "treatment",
    target_column: str = "target",
    id_columns: Sequence[str] = ("client_id",),
) -> list[str]:
    """Выбрать признаки, исключив идентификаторы и экспериментальные labels."""
    excluded = {*id_columns, treatment_column, target_column}
    columns = [column for column in frame.columns if column not in excluded]
    if not columns:
        raise UpliftDataError("После исключения labels не осталось признаков")
    return columns


def validate_uplift_frame(
    frame: pl.DataFrame,
    feature_columns: Sequence[str],
    *,
    treatment_column: str = "treatment",
    target_column: str = "target",
) -> None:
    """Проверить схему, бинарность labels и отсутствие прямой утечки."""
    if frame.height == 0:
        raise UpliftDataError("Витрина uplift-модели не должна быть пустой")
    if not feature_columns:
        raise UpliftDataError("Нужно передать хотя бы один признак")
    if len(feature_columns) != len(set(feature_columns)):
        raise UpliftDataError("Список признаков содержит дубликаты")
    forbidden = {treatment_column, target_column} & set(feature_columns)
    if forbidden:
        raise UpliftDataError(
            f"Labels нельзя использовать как признаки: {sorted(forbidden)}"
        )
    missing = ({treatment_column, target_column} | set(feature_columns)) - set(
        frame.columns
    )
    if missing:
        raise UpliftDataError(f"Отсутствуют колонки: {sorted(missing)}")
    invalid = frame.filter(
        pl.col(treatment_column).is_null()
        | pl.col(target_column).is_null()
        | ~pl.col(treatment_column).is_in([0, 1])
        | ~pl.col(target_column).is_in([0, 1])
    )
    if invalid.height:
        raise UpliftDataError(
            "Treatment и target должны быть бинарными 0/1 без пропусков"
        )


def infer_feature_spec(
    frame: pl.DataFrame, feature_columns: Sequence[str]
) -> FeatureSpec:
    """Зафиксировать поддерживаемые числовые, категориальные и временные поля."""
    numeric: list[str] = []
    categorical: list[str] = []
    temporal: list[str] = []
    schema = frame.schema
    for column in feature_columns:
        dtype = schema[column]
        if dtype.is_numeric():
            numeric.append(column)
        elif dtype.is_temporal():
            numeric.append(column)
            temporal.append(column)
        elif dtype in {pl.String, pl.Categorical, pl.Boolean} or dtype == pl.Null:
            categorical.append(column)
        else:
            raise UpliftDataError(
                f"Неподдерживаемый тип признака {column!r}: {dtype}. "
                "Преобразуйте List/Struct/Object до обучения."
            )
    return FeatureSpec(
        columns=tuple(feature_columns),
        numeric=tuple(numeric),
        categorical=tuple(categorical),
        temporal=tuple(temporal),
    )


def prepare_feature_matrix(frame: pl.DataFrame, spec: FeatureSpec) -> pd.DataFrame:
    """Преобразовать Polars-признаки в стабильную pandas-матрицу для sklearn."""
    missing = set(spec.columns) - set(frame.columns)
    if missing:
        raise UpliftDataError(f"Для prediction отсутствуют признаки: {sorted(missing)}")
    expressions: list[pl.Expr] = []
    temporal = set(spec.temporal)
    numeric = set(spec.numeric)
    for column in spec.columns:
        expression = pl.col(column)
        if column in temporal:
            expression = expression.cast(pl.Int64).cast(pl.Float64)
        elif column in numeric:
            expression = expression.cast(pl.Float64)
        else:
            expression = expression.cast(pl.String)
        expressions.append(expression.alias(column))
    return frame.select(expressions).to_pandas()


def build_preprocessor(
    spec: FeatureSpec,
    *,
    extra_numeric: Sequence[str] = (),
) -> ColumnTransformer:
    """Создать imputing/scaling/one-hot preprocessing без знания holdout."""
    transformers: list[tuple[str, Pipeline, list[str]]] = []
    numeric_columns = [*spec.numeric, *extra_numeric]
    if numeric_columns:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_columns,
            )
        )
    if spec.categorical:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(strategy="constant", fill_value="Неизвестно"),
                        ),
                        (
                            "one_hot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=True),
                        ),
                    ]
                ),
                list(spec.categorical),
            )
        )
    return ColumnTransformer(transformers, remainder="drop")


def require_both_outcomes(target: Sequence[int], context: str) -> None:
    """Проверить возможность обучения бинарной классификации."""
    if set(target) != {0, 1}:
        raise UpliftDataError(
            f"{context} должен содержать оба класса target; получено {sorted(set(target))}"
        )
