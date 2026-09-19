"""Проверить готовность базы и PBIP после автоматического развёртывания."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sqlalchemy import text

from src.config.settings import PROJECT_ROOT, get_settings
from src.data.database import create_db_engine

REQUIRED_TABLES = (
    "dim_experiment",
    "fact_experiment",
    "experiment_results",
    "experiment_validation_summary",
    "segment_experiment_results",
    "uplift_model_results",
    "uplift_predictions",
    "uplift_deciles",
    "business_scenarios",
)


def verify_database() -> dict[str, int]:
    """Вернуть число строк и потребовать непустую отчётную модель."""
    engine = create_db_engine(get_settings())
    counts: dict[str, int] = {}
    with engine.connect() as connection:
        for table in REQUIRED_TABLES:
            count = connection.execute(
                text(f"SELECT count(*) FROM analytics.{table}")
            ).scalar_one()
            counts[table] = int(count)
    empty = [table for table, count in counts.items() if count == 0]
    if empty:
        raise RuntimeError(f"Пустые обязательные витрины: {empty}")
    return counts


def verify_pbip(root: Path = PROJECT_ROOT) -> Path:
    """Проверить точку входа PBIP и наличие пяти страниц."""
    dashboard = root / "powerbi" / "dashboard"
    pbip = dashboard / "RetailExperimentation.pbip"
    pages_root = dashboard / "RetailExperimentation.Report" / "definition" / "pages"
    metadata = json.loads((pages_root / "pages.json").read_text(encoding="utf-8"))
    if not pbip.exists() or len(metadata["pageOrder"]) != 5:
        raise RuntimeError("PBIP-проект неполон")
    return pbip


def main() -> None:
    counts = verify_database()
    pbip = verify_pbip()
    print("Проверка пройдена:")
    for table, count in counts.items():
        print(f"  analytics.{table}: {count:,}")
    print(f"  Power BI: {pbip}")


if __name__ == "__main__":
    main()
