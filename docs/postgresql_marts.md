# PostgreSQL: аналитические витрины

## Назначение

PostgreSQL хранит канонические результаты Python-расчётов и предоставляет узкие представления для BI. Сложные статистические тесты, bootstrap, Qini и оптимизация порога не пересчитываются в SQL или DAX.

Фактическая база в репозитории не заполнена: исходные X5 RetailHero файлы отсутствуют. DDL и загрузчик проверены unit-тестами; интеграционный запуск PostgreSQL требует работающий Docker Engine или внешний PostgreSQL 16.

## Схемы и зерно

DDL находится в `sql/ddl/01_analytics.sql`.

| Таблица | Зерно |
|---|---|
| `dim_customer` | один клиент |
| `dim_product` | один товар |
| `dim_date` | один календарный день |
| `dim_experiment` | один эксперимент |
| `fact_purchases` | одна строка покупки из cleaned source |
| `fact_experiment` | клиент × эксперимент |
| `experiment_results` | эксперимент |
| `experiment_validation_summary` | эксперимент |
| `segment_experiment_results` | эксперимент × измерение × сегмент |
| `uplift_model_results` | эксперимент × модель |
| `uplift_predictions` | эксперимент × клиент × модель |
| `uplift_deciles` | эксперимент × модель × дециль |
| `business_scenarios` | эксперимент × модель × сценарий |

Primary keys фиксируют зерно. Foreign keys связывают факты с клиентами, товарами, датами и экспериментами. CHECK constraints защищают бинарные labels, вероятности, alpha, scores и календарные атрибуты. Индексы добавлены для клиентской истории, экспериментальных групп, predicted uplift и бизнес-сценариев.

## Raw mapping

Загрузчик не предполагает имена исходных полей. Для `fact_purchases` и `dim_product` используется тот же JSON mapping, что и для feature mart. Если отдельного справочника товаров нет, но `product_id` присутствует в покупках, `dim_product` строится из уникальных идентификаторов с пустой категорией.

Клиентские, экспериментальные, модельные и бизнес-таблицы загружаются из канонических Parquet-артефактов проекта.

## Идемпотентность и транзакции

- DDL использует `IF NOT EXISTS` и `CREATE OR REPLACE VIEW`.
- Измерения и агрегированные результаты загружаются через параметризованный `INSERT ... ON CONFLICT`.
- `fact_experiment`, predictions и purchases заменяются как контролируемые snapshots внутри одной транзакции.
- SQL identifiers проходят allowlist-проверку; значения передаются bind-параметрами.
- При ошибке транзакция откатывается целиком.

Snapshot `fact_purchases` сохраняет повторяющиеся item-level строки и не требует выдуманного natural key, которого нет в подтверждённой схеме источника.

## Запуск

Запустите PostgreSQL:

```bash
docker compose up -d postgres
docker compose ps
```

После материализации Python-артефактов:

```bash
python -m src.data.warehouse \
  --experiment-name retailhero_campaign \
  --mapping customer_feature_schema.json \
  --expected-treatment-share 0.5 \
  --scenario-name base
```

Без `--mapping` загружаются клиентская витрина и аналитические результаты, но не purchase/product facts. Экономический `scenario-name` входит в ключ и предотвращает неявное смешение разных assumptions.

При создании нового Docker volume файлы `00_init.sql` и `01_analytics.sql` выполняются автоматически. Для существующей базы CLI применяет идемпотентный DDL перед загрузкой. Флаг `--skip-ddl` предназначен для среды с отдельным migration process.

## Представления и контроль качества

`sql/marts/01_power_bi_views.sql` создаёт:

- `vw_experiment_overview`;
- `vw_segment_performance`;
- `vw_uplift_model_performance`;
- `vw_business_strategy_comparison`.

`sql/analysis/01_quality_checks.sql` проверяет orphan keys, бинарность labels, согласованность uplift и обязательную маркировку scenario analysis. После успешной загрузки все строки запроса должны иметь `violations = 0`.

Пример запуска:

```bash
psql "$DATABASE_URL" -f sql/analysis/01_quality_checks.sql
```

## Тестовая стратегия

- DDL проверяется на обязательные таблицы, composite keys, foreign keys, CHECK constraints и индексы.
- Нормализация клиентов, эксперимента, товаров, покупок и дат проверяется на небольших ручных fixtures.
- Дата покупки должна давать правильный `date_key`, ISO weekday и weekend flag.
- Wide scores S/T-Learner преобразуются в одну строку на клиент × модель.
- SRM, A/A и bootstrap summaries объединяются только по явному mapping полей.
- Генератор upsert проверяется на bind-параметры и отклонение опасных identifiers.
- SQL-файлы проверяются на последовательное выполнение отдельных statements.

## Ограничения

- DDL не заменяет управление миграциями для production-среды; изменения существующих колонок потребуют Alembic или аналогичного инструмента.
- Полная интеграционная проверка ограничений требует работающего PostgreSQL. Unit-тесты не эмулируют особенности PostgreSQL через SQLite.
- Денежные поля сохраняют сценарный характер и не становятся фактической прибылью после загрузки в базу.
