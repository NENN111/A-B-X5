# Power BI

Каталог содержит воспроизводимую спецификацию отчёта поверх PostgreSQL-схемы
`analytics`. Файл `.pbix` не хранится в Git: это бинарный артефакт, неудобный
для code review. Модель, меры и оформление описаны текстом и повторяются в
Power BI Desktop.

## Состав

- [`data_model.md`](data_model.md) — подключение, таблицы, связи и обновление;
- [`dax_measures.md`](dax_measures.md) — calculated tables и готовые меры;
- [`report_spec.md`](report_spec.md) — пять страниц и правила интерпретации;
- [`theme.json`](theme.json) — светлая минималистичная тема.

## Сборка

1. Загрузить результаты Python в PostgreSQL.
2. Создать параметры `Server`, `Database`, `Schema` и подключить таблицы.
3. Настроить связи до добавления мер.
4. Создать служебные таблицы и меры из `dax_measures.md`.
5. Импортировать тему и собрать страницы по `report_spec.md`.
6. Сверить users и conversions с PostgreSQL.

Тяжёлая статистика не переносится в DAX. P-value, CI, power, MDE, SRM,
bootstrap, AUUC и Qini поступают готовыми из Python/PostgreSQL.

## Готовый отчёт

Точка входа — `dashboard/RetailExperimentation.pbip`. Проект содержит
семантическую модель TMDL, DAX-меры и пять собранных страниц. Он подключается к
`localhost:5432`, базе `retail_experiments`, схеме `analytics`.

Учётные данные намеренно не сохраняются в PBIP. При первом обновлении выберите
аутентификацию «База данных» и используйте значения `POSTGRES_USER` и
`POSTGRES_PASSWORD` из корневого `.env`. Полная инструкция находится в
`docs/deployment.md`.
