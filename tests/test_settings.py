"""Тесты настроек путей и подключения."""

from __future__ import annotations

from src.config.settings import PROJECT_ROOT, get_settings


def test_default_paths_are_anchored_to_project_root() -> None:
    get_settings.cache_clear()
    settings = get_settings()

    assert settings.data_dir == PROJECT_ROOT / "data"
    assert settings.raw_data_dir == PROJECT_ROOT / "data" / "raw"
    assert not settings.database_url.endswith("@")
