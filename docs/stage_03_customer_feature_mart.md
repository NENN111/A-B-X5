# Stage 3 — клиентская витрина признаков

## Статус данных

Исходные файлы X5 RetailHero пока отсутствуют. Код витрины реализован и проверен на unit fixtures, но фактическая витрина и численные результаты не создаются. Названия исходных колонок задаются только после запуска Stage 2 и изучения `schema_inventory.json`.

## Защита от target leakage

`experiment_start` — обязательная граница. Для признаков используются только покупки из полуинтервала:

```text
[experiment_start - lookback_days; experiment_start)
```

Покупки ровно в момент старта и после него исключаются. Treatment и target присоединяются только после завершения всех агрегаций. Timezone даты старта должна совпадать с timezone временной колонки покупок; неявное преобразование запрещено.

## Настройка фактических колонок

1. Выполните Stage 2 и откройте `data/interim/quality/raw/schema_inventory.json`.
2. Скопируйте `src/config/customer_feature_schema.example.json` в локальный файл, например `customer_feature_schema.json`.
3. Заполните только фактические имена колонок и точную дату старта эксперимента.
4. Не коммитьте mapping, если он содержит закрытую информацию.

Обязательные роли:

- `clients.client_id`;
- `purchases.client_id`;
- `purchases.transaction_id`;
- `purchases.transaction_datetime`;
- `purchases.amount`.

Опциональные роли включают возраст или дату рождения, пол, товар, категорию, регулярную стоимость и experiment labels. Секция `products` требует `purchases.product_id`; сам `product_id` может использоваться без категорий для расчёта `unique_products`.

## Запуск

```bash
python -m src.features.customer_features \
  --mapping customer_feature_schema.json
```

По умолчанию читаются Parquet-файлы Stage 2 из `data/interim/clean`, а результат записывается в `data/processed/customer_features.parquet`. Рядом создаётся `customer_features.metadata.json` с mapping, границами окна, схемой и аудитом исключённых строк.

## Признаки

При наличии необходимых ролей рассчитываются:

- `age`, `gender`;
- `total_spend`, `avg_check`, `median_check`, `max_check`;
- `transactions_count`, число транзакций на 30 дней;
- `purchases_30d`, `purchases_60d`, `purchases_90d`;
- `days_since_last_purchase`;
- `unique_products`, `unique_categories`, `favorite_category`;
- `avg_discount`, `discounted_purchase_share`;
- `recency_score`, `frequency_score`, `monetary_score`, `rfm_segment`;
- квинтили трат и частоты;
- `treatment` и `target` как отдельные labels, если настроена секция experiment.

Чек сначала агрегируется по паре клиент–транзакция, поэтому item-level строки не завышают `transactions_count`. Любимые категории определяются по pre-treatment тратам; при равенстве применяется стабильная сортировка по названию категории.

## RFM

Scores принимают значения 1–5 и строятся через percentile rank:

- меньшая recency лучше;
- большая frequency лучше;
- большая monetary лучше.

При единственном непустом наблюдении присваивается нейтральный score 3. Для клиента без покупок recency и `rfm_segment` остаются null, а счётчики и траты заполняются нулями.

## Контроли качества

Pipeline останавливается, если:

- в mapping отсутствует обязательная роль;
- настроенная колонка не найдена;
- ключ клиента, товара или experiment assignment не уникален;
- обязательные поля покупки содержат null;
- сумма покупки не числовая;
- timezone cutoff и покупок несовместимы;
- при строгом режиме покупка ссылается на неизвестного клиента или товар.

Автоматическое исправление бизнес-аномалий не выполняется без фактического исследования данных.
