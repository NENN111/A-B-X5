# Развёртывание проекта

## Рекомендуемый способ — Windows и Docker Desktop

1. Установите Git, Docker Desktop и Power BI Desktop.
2. Запустите Docker Desktop и дождитесь статуса **Engine running**.
3. Клонируйте репозиторий и выполните из его корня:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup_windows.ps1
```

Сценарий можно запускать повторно. Чтобы не открывать Power BI автоматически,
добавьте параметр `-NoOpenPowerBI`.

## Что создаётся локально

- `data/raw` — исходные CSV.GZ RetailHero;
- `data/interim` — очищенные Parquet и отчёты качества;
- `data/processed` — feature mart, статистика, uplift-score и бизнес-сценарии;
- Docker volume `postgres_data` — PostgreSQL-витрины;
- `.env` — локальные настройки, исключённые из Git.

Эти артефакты не загружаются в репозиторий. PBIP хранит только модель, DAX,
визуалы и адрес локального PostgreSQL; пароль в файлы дашборда не встраивается.

## Ручной запуск через Docker

```powershell
Copy-Item .env.example .env
docker compose up -d --wait postgres
docker compose --profile tools build pipeline
docker compose --profile tools run --rm pipeline python scripts/run_pipeline.py
docker compose --profile tools run --rm pipeline python scripts/verify_deployment.py
```

После успешной проверки откройте
`powerbi/dashboard/RetailExperimentation.pbip` и обновите данные.

## Локальный запуск Python

Поддерживается Python 3.11–3.12:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
python scripts/download_retailhero.py
python scripts/run_pipeline.py --skip-download
python scripts/verify_deployment.py
```

Перед локальным запуском поднимите PostgreSQL командой
`docker compose up -d --wait postgres`.

## Типовые ошибки

- **Docker Engine не запущен** — откройте Docker Desktop и дождитесь полной загрузки.
- **Порт 5432 занят** — остановите другой PostgreSQL либо измените
  `POSTGRES_PORT` и адрес источника в Power BI.
- **Power BI не подключается** — используйте тип аутентификации «База данных»,
  сервер `localhost:5432`, базу `retail_experiments` и значения
  `POSTGRES_USER`/`POSTGRES_PASSWORD` из `.env`.
- **Загрузка данных оборвалась** — снова запустите setup; временный `.part`-файл
  будет удалён, уже загруженные корректные файлы будут пропущены.

## Очистка окружения

Остановить сервисы без удаления базы:

```powershell
docker compose down
```

Удаление Docker volume с базой выполняйте только осознанно:

```powershell
docker compose down --volumes
```
