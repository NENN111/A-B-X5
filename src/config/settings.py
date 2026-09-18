"""Настройки проекта из переменных окружения с безопасными локальными значениями."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from urllib.parse import quote_plus

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _project_path(value: str) -> Path:
    """Преобразовать настроенный путь относительно корня репозитория."""
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


@dataclass(frozen=True, slots=True)
class Settings:
    """Настройки приложения, загружаемые из переменных окружения."""

    app_env: str
    log_level: str
    random_state: int
    data_dir: Path
    raw_data_dir: Path
    interim_data_dir: Path
    processed_data_dir: Path
    postgres_host: str
    postgres_port: int
    postgres_db: str
    postgres_user: str
    postgres_password: str

    @property
    def database_url(self) -> str:
        """Вернуть URL SQLAlchemy для драйвера psycopg v3."""
        return (
            f"postgresql+psycopg://{quote_plus(self.postgres_user)}:"
            f"{quote_plus(self.postgres_password)}@{self.postgres_host}:"
            f"{self.postgres_port}/{quote_plus(self.postgres_db)}"
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Однократно загрузить `.env` и вернуть неизменяемые настройки."""
    load_dotenv(PROJECT_ROOT / ".env")
    data_dir = _project_path(os.getenv("DATA_DIR", "data"))
    return Settings(
        app_env=os.getenv("APP_ENV", "development"),
        log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
        random_state=int(os.getenv("RANDOM_STATE", "42")),
        data_dir=data_dir,
        raw_data_dir=_project_path(os.getenv("RAW_DATA_DIR", str(data_dir / "raw"))),
        interim_data_dir=_project_path(
            os.getenv("INTERIM_DATA_DIR", str(data_dir / "interim"))
        ),
        processed_data_dir=_project_path(
            os.getenv("PROCESSED_DATA_DIR", str(data_dir / "processed"))
        ),
        postgres_host=os.getenv("POSTGRES_HOST", "localhost"),
        postgres_port=int(os.getenv("POSTGRES_PORT", "5432")),
        postgres_db=os.getenv("POSTGRES_DB", "retail_experiments"),
        postgres_user=os.getenv("POSTGRES_USER", "retail"),
        postgres_password=os.getenv("POSTGRES_PASSWORD", "retail_local_only"),
    )
