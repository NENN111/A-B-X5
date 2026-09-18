"""Контракт сопоставления фактических колонок с ролями feature mart."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


class FeatureConfigError(ValueError):
    """Ошибка неполной или противоречивой конфигурации признаков."""


@dataclass(frozen=True, slots=True)
class ClientColumns:
    """Роли колонок таблицы клиентов."""

    client_id: str
    age: str | None = None
    birth_date: str | None = None
    gender: str | None = None


@dataclass(frozen=True, slots=True)
class PurchaseColumns:
    """Роли колонок истории покупок."""

    client_id: str
    transaction_id: str
    transaction_datetime: str
    amount: str
    product_id: str | None = None
    regular_amount: str | None = None
    datetime_format: str | None = None


@dataclass(frozen=True, slots=True)
class ProductColumns:
    """Роли колонок справочника товаров."""

    product_id: str
    category: str


@dataclass(frozen=True, slots=True)
class ExperimentColumns:
    """Роли колонок назначения групп и результата эксперимента."""

    client_id: str
    treatment: str
    target: str


@dataclass(frozen=True, slots=True)
class FeatureMartConfig:
    """Полная конфигурация leakage-safe клиентской витрины."""

    experiment_start: datetime
    clients: ClientColumns
    purchases: PurchaseColumns
    products: ProductColumns | None = None
    experiment: ExperimentColumns | None = None
    lookback_days: int = 365
    strict_referential_integrity: bool = True

    def __post_init__(self) -> None:
        if self.lookback_days < 90:
            raise FeatureConfigError("lookback_days должен быть не меньше 90")
        if self.clients.age and self.clients.birth_date:
            raise FeatureConfigError(
                "Укажите только одну роль: clients.age или clients.birth_date"
            )
        if self.products is not None and self.purchases.product_id is None:
            raise FeatureConfigError(
                "Секция products требует mapping purchases.product_id"
            )


def _required_string(value: Any, path: str) -> str:
    """Проверить обязательное непустое строковое значение mapping."""
    if not isinstance(value, str) or not value.strip():
        raise FeatureConfigError(f"Заполните обязательное поле mapping: {path}")
    return value


def _optional_string(value: Any, path: str) -> str | None:
    """Проверить опциональное строковое значение mapping."""
    if value is None:
        return None
    return _required_string(value, path)


def load_feature_config(path: Path) -> FeatureMartConfig:
    """Загрузить и проверить JSON mapping фактических колонок."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FeatureConfigError(f"Файл mapping не найден: {path}") from error
    except json.JSONDecodeError as error:
        raise FeatureConfigError(f"Некорректный JSON в {path}: {error}") from error

    try:
        start_value = _required_string(
            payload.get("experiment_start"), "experiment_start"
        )
        experiment_start = datetime.fromisoformat(start_value)
    except ValueError as error:
        raise FeatureConfigError(
            "experiment_start должен быть ISO datetime, например 2024-01-01T00:00:00+00:00"
        ) from error

    clients_payload = payload.get("clients") or {}
    purchases_payload = payload.get("purchases") or {}
    products_payload = payload.get("products")
    experiment_payload = payload.get("experiment")

    clients = ClientColumns(
        client_id=_required_string(
            clients_payload.get("client_id"), "clients.client_id"
        ),
        age=_optional_string(clients_payload.get("age"), "clients.age"),
        birth_date=_optional_string(
            clients_payload.get("birth_date"), "clients.birth_date"
        ),
        gender=_optional_string(clients_payload.get("gender"), "clients.gender"),
    )
    purchases = PurchaseColumns(
        client_id=_required_string(
            purchases_payload.get("client_id"), "purchases.client_id"
        ),
        transaction_id=_required_string(
            purchases_payload.get("transaction_id"), "purchases.transaction_id"
        ),
        transaction_datetime=_required_string(
            purchases_payload.get("transaction_datetime"),
            "purchases.transaction_datetime",
        ),
        amount=_required_string(purchases_payload.get("amount"), "purchases.amount"),
        product_id=_optional_string(
            purchases_payload.get("product_id"), "purchases.product_id"
        ),
        regular_amount=_optional_string(
            purchases_payload.get("regular_amount"), "purchases.regular_amount"
        ),
        datetime_format=_optional_string(
            purchases_payload.get("datetime_format"), "purchases.datetime_format"
        ),
    )
    products = None
    if products_payload is not None:
        products = ProductColumns(
            product_id=_required_string(
                products_payload.get("product_id"), "products.product_id"
            ),
            category=_required_string(
                products_payload.get("category"), "products.category"
            ),
        )
    experiment = None
    if experiment_payload is not None:
        experiment = ExperimentColumns(
            client_id=_required_string(
                experiment_payload.get("client_id"), "experiment.client_id"
            ),
            treatment=_required_string(
                experiment_payload.get("treatment"), "experiment.treatment"
            ),
            target=_required_string(
                experiment_payload.get("target"), "experiment.target"
            ),
        )

    return FeatureMartConfig(
        experiment_start=experiment_start,
        clients=clients,
        purchases=purchases,
        products=products,
        experiment=experiment,
        lookback_days=int(payload.get("lookback_days", 365)),
        strict_referential_integrity=bool(
            payload.get("strict_referential_integrity", True)
        ),
    )
