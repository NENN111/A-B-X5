# Исходные данные RetailHero

При стандартном развёртывании файлы скачиваются автоматически командой
`python scripts/download_retailhero.py` с публичного зеркала набора
[`pytorch-lifestream/retailhero-uplift`](https://huggingface.co/datasets/pytorch-lifestream/retailhero-uplift).
Страница исходного соревнования: [ODS X5 RetailHero](https://ods.ai/competitions/x5-retailhero-uplift-modeling/data).

Поместите оригинальные файлы X5 RetailHero Uplift в этот каталог. Загрузчик ожидает ровно по одному файлу каждой сущности в формате CSV, CSV.GZ или Parquet:

- `clients.csv` / `clients.csv.gz` / `clients.parquet`;
- `products.csv` / `products.csv.gz` / `products.parquet`;
- `purchases.csv` / `purchases.csv.gz` / `purchases.parquet`;
- `uplift_train.csv` / `uplift_train.csv.gz` / `uplift_train.parquet`.

Допускаются префиксы и суффиксы в именах файлов, если название соответствующей сущности входит в имя и совпадение остаётся однозначным. Raw-файлы намеренно исключены из Git.

Проект не предполагает названия колонок до чтения источников. После размещения данных выполните из корня репозитория:

```bash
python -m src.data.loader
```

Команда выведет фактические названия и типы колонок, не загружая таблицы целиком в память. Если файлы из вашей поставки называются иначе, сохраните оригиналы и сначала задокументируйте соответствие сущностей. Не переименовывайте колонки источника без явного описания преобразования.
