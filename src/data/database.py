"""Фабрика подключений к PostgreSQL."""

from __future__ import annotations

from sqlalchemy import Engine, create_engine, text

from src.config.settings import Settings, get_settings


def create_db_engine(settings: Settings | None = None) -> Engine:
    """Создать SQLAlchemy engine с пулом подключений из настроек окружения."""
    config = settings or get_settings()
    return create_engine(config.database_url, pool_pre_ping=True)


def database_is_ready(engine: Engine) -> bool:
    """Проверить подключение к базе данных минимальным запросом."""
    with engine.connect() as connection:
        return connection.execute(text("SELECT 1")).scalar_one() == 1
