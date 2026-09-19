"""Запустить полный воспроизводимый конвейер RetailHero до PostgreSQL-витрин."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sqlalchemy import text

from scripts.download_retailhero import download_dataset
from src.config.settings import get_settings
from src.data.database import create_db_engine

MAPPING = ROOT / "config" / "customer_feature_schema.retailhero.json"


@dataclass(frozen=True, slots=True)
class PipelineStep:
    name: str
    arguments: tuple[str, ...]


STEPS = (
    PipelineStep("Очистка данных", ("-m", "src.data.preprocessing")),
    PipelineStep(
        "Клиентская витрина",
        ("-m", "src.features.customer_features", "--mapping", str(MAPPING)),
    ),
    PipelineStep("A/B-анализ", ("-m", "src.experiments.ab_test")),
    PipelineStep("Валидация эксперимента", ("-m", "src.experiments.validation")),
    PipelineStep("Uplift-моделирование", ("-m", "src.models.model_evaluation")),
    PipelineStep(
        "Бизнес-оптимизация",
        (
            "-m",
            "src.business.optimization",
            "--communication-cost",
            "2.0",
            "--profit-per-conversion",
            "100.0",
            "--targeting-threshold",
            "0.02",
        ),
    ),
    PipelineStep(
        "Загрузка витрин PostgreSQL",
        (
            "-m",
            "src.data.warehouse",
            "--experiment-name",
            "retailhero_campaign",
            "--scenario-name",
            "base",
        ),
    ),
)


def wait_for_postgres(timeout: int = 90) -> None:
    """Дождаться PostgreSQL и выдать понятную ошибку при недоступности."""
    engine = create_db_engine(get_settings())
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with engine.connect() as connection:
                connection.execute(text("SELECT 1"))
            return
        except Exception as error:  # noqa: BLE001 - исходная ошибка полезнее обёртки
            last_error = error
            time.sleep(2)
    raise RuntimeError(f"PostgreSQL не стал доступен за {timeout} с: {last_error}")


def run_steps(*, skip_warehouse: bool = False) -> None:
    """Последовательно выполнить этапы и остановиться на первой ошибке."""
    selected = STEPS[:-1] if skip_warehouse else STEPS
    for index, step in enumerate(selected, 1):
        print(f"\n[{index}/{len(selected)}] {step.name}", flush=True)
        subprocess.run((sys.executable, *step.arguments), cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Полная сборка аналитического проекта.")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-warehouse", action="store_true")
    args = parser.parse_args()

    if not args.skip_download:
        print("[данные] Проверка и загрузка RetailHero", flush=True)
        download_dataset(get_settings().raw_data_dir)
    if not args.skip_warehouse:
        print("[postgres] Проверка подключения", flush=True)
        wait_for_postgres()
    run_steps(skip_warehouse=args.skip_warehouse)
    print("\nГотово: расчёты завершены, витрины доступны для Power BI.")


if __name__ == "__main__":
    main()
