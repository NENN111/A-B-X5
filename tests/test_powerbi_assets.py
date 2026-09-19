"""Контрактные проверки версионируемых артефактов Power BI."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
POWERBI = ROOT / "powerbi"
DASHBOARD = POWERBI / "dashboard"


def test_theme_is_valid_json_with_semantic_colors() -> None:
    theme = json.loads((POWERBI / "theme.json").read_text(encoding="utf-8"))

    assert theme["name"] == "Retail Experimentation Light"
    assert len(theme["dataColors"]) >= 6
    assert {"good", "neutral", "bad"} <= theme.keys()


def test_data_model_documents_every_warehouse_table() -> None:
    ddl = (ROOT / "sql" / "ddl" / "01_analytics.sql").read_text(encoding="utf-8")
    model = (POWERBI / "data_model.md").read_text(encoding="utf-8")
    table_names = {
        line.split("analytics.", maxsplit=1)[1].split()[0]
        for line in ddl.splitlines()
        if line.startswith("CREATE TABLE IF NOT EXISTS analytics.")
    }

    assert table_names
    assert all(f"`{table}`" in model for table in table_names)


def test_required_dax_measures_are_documented() -> None:
    dax = (POWERBI / "dax_measures.md").read_text(encoding="utf-8")
    required = {
        "Users",
        "Control Users",
        "Treatment Users",
        "Conversions",
        "Control CR",
        "Treatment CR",
        "Absolute Uplift",
        "Relative Uplift",
        "Incremental Conversions",
        "Campaign Cost",
        "Incremental Profit",
        "ROI",
        "Target Share",
    }

    assert all(f"{measure} =" in dax for measure in required)
    assert "P-value = SELECTEDVALUE" in dax
    assert "P-value = CALCULATE" not in dax


def test_report_spec_contains_five_required_pages() -> None:
    report = (POWERBI / "report_spec.md").read_text(encoding="utf-8")
    pages = (
        "Executive Overview",
        "Experiment Diagnostics",
        "Customer Segments",
        "Uplift Modeling",
        "Business Impact",
    )

    assert all(f"## {index}. {page}" in report for index, page in enumerate(pages, 1))


def test_generated_pbip_has_five_pages_in_expected_order() -> None:
    pages_root = DASHBOARD / "RetailExperimentation.Report" / "definition" / "pages"
    metadata = json.loads((pages_root / "pages.json").read_text(encoding="utf-8"))
    display_names = [
        json.loads((pages_root / page / "page.json").read_text(encoding="utf-8"))["displayName"]
        for page in metadata["pageOrder"]
    ]

    assert display_names == [
        "Executive Overview",
        "Experiment Diagnostics",
        "Customer Segments",
        "Uplift Modeling",
        "Business Impact",
    ]


def test_generated_model_uses_docker_postgres_without_credentials() -> None:
    model_root = DASHBOARD / "RetailExperimentation.SemanticModel" / "definition"
    model_text = "\n".join(
        path.read_text(encoding="utf-8") for path in model_root.rglob("*.tmdl")
    )

    assert "localhost:5432" in model_text
    assert "retail_experiments" in model_text
    assert "retail_local_only" not in model_text
    assert "Password=" not in model_text
