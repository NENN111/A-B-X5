"""Тесты обнаружения исходных файлов."""

from __future__ import annotations

import gzip
from pathlib import Path

import pytest

from src.data.loader import MissingRawDataError, RawDataLoader


@pytest.mark.parametrize("suffix", [".csv", ".csv.gz", ".parquet"])
def test_discover_accepts_supported_formats(tmp_path: Path, suffix: str) -> None:
    for entity in ("clients", "products", "purchases", "uplift_train"):
        (tmp_path / f"x5_{entity}{suffix}").touch()

    discovered = RawDataLoader(tmp_path).discover()

    assert set(discovered) == {
        "clients",
        "products",
        "purchases",
        "uplift_train",
    }


def test_discover_reports_missing_entities(tmp_path: Path) -> None:
    (tmp_path / "clients.csv").touch()

    with pytest.raises(MissingRawDataError, match="products"):
        RawDataLoader(tmp_path).discover()


def test_discover_rejects_ambiguous_entity_files(tmp_path: Path) -> None:
    for entity in ("clients", "products", "purchases", "uplift_train"):
        (tmp_path / f"{entity}.csv").touch()
    (tmp_path / "clients_backup.csv").touch()

    with pytest.raises(MissingRawDataError, match="несколько файлов"):
        RawDataLoader(tmp_path).discover()


def test_scan_reads_compressed_csv(tmp_path: Path) -> None:
    path = tmp_path / "clients.csv.gz"
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        stream.write("client_id,value\n1,10\n2,20\n")

    frame = RawDataLoader(tmp_path).scan(path).collect()

    assert frame.shape == (2, 2)
    assert frame["client_id"].to_list() == [1, 2]
