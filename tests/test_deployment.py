"""Проверки сценариев воспроизводимого развёртывания."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.download_retailhero import FILES, DatasetFile, _download
from scripts.run_pipeline import STEPS
from scripts.verify_deployment import verify_pbip


def test_dataset_manifest_contains_all_retailhero_sources() -> None:
    assert {item.name for item in FILES} == {
        "clients.csv.gz",
        "products.csv.gz",
        "purchases.csv.gz",
        "uplift_train.csv.gz",
    }
    assert all(item.size > 0 for item in FILES)


def test_downloader_reuses_complete_file(tmp_path: Path) -> None:
    item = DatasetFile("sample.csv.gz", 3)
    existing = tmp_path / item.name
    existing.write_bytes(b"abc")

    assert _download(item, tmp_path) == existing
    assert existing.read_bytes() == b"abc"


def test_pipeline_ends_with_warehouse_load() -> None:
    assert len(STEPS) == 7
    assert STEPS[-1].arguments[:2] == ("-m", "src.data.warehouse")
    assert "--mapping" not in STEPS[-1].arguments


def test_verify_pbip_requires_five_pages(tmp_path: Path) -> None:
    dashboard = tmp_path / "powerbi" / "dashboard"
    pages = dashboard / "RetailExperimentation.Report" / "definition" / "pages"
    pages.mkdir(parents=True)
    (dashboard / "RetailExperimentation.pbip").write_text("{}", encoding="utf-8")
    (pages / "pages.json").write_text(
        json.dumps({"pageOrder": ["a", "b", "c", "d", "e"]}),
        encoding="utf-8",
    )

    assert verify_pbip(tmp_path).name == "RetailExperimentation.pbip"
