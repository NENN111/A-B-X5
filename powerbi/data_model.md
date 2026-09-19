# Модель данных Power BI

## Подключение

Для портфолио-версии используйте режим **Import**: агрегированные таблицы
компактны, а взаимодействия и What-if расчёты работают быстрее. Создайте
параметры Power Query `Server`, `Database` и `Schema` без учётных данных.

| Параметр | Тип | Локальный пример |
|---|---|---|
| `Server` | Text | `localhost:5432` |
| `Database` | Text | `retail_analytics` |
| `Schema` | Text | `analytics` |

Базовый запрос для каждой таблицы:

```powerquery
let
    Source = PostgreSQL.Database(Server, Database),
    Analytics = Source{[Schema = Schema, Item = "fact_experiment"]}[Data]
in
    Analytics
```

Заменяйте только `Item`. Пароль задаётся в настройках источника Power BI и не
хранится в репозитории. Бизнес-логику в Power Query не дублируйте.

## Таблицы и зерно

| Таблица | Зерно | Назначение |
|---|---|---|
| `dim_experiment` | эксперимент | общий фильтр отчёта |
| `dim_customer` | клиент | профиль клиентов |
| `dim_date` | день | календарь покупок |
| `dim_product` | товар | категории покупок |
| `fact_experiment` | эксперимент × клиент | группы и конверсии |
| `fact_purchases` | строка покупки | история до эксперимента |
| `experiment_results` | эксперимент | готовый A/B inference |
| `experiment_validation_summary` | эксперимент | SRM, A/A и bootstrap summary |
| `segment_experiment_results` | эксперимент × измерение × сегмент | HTE с BH-поправкой |
| `uplift_model_results` | эксперимент × модель | AUUC и Qini |
| `uplift_predictions` | эксперимент × модель × клиент | What-if таргетинг |
| `uplift_deciles` | эксперимент × модель × дециль | uplift-графики |
| `business_scenarios` | эксперимент × модель × сценарий | сценарии Python |

SQL views предназначены для быстрой проверки. Не загружайте их одновременно с
базовыми таблицами: показатели задублируются, а модель станет неоднозначной.

## Связи

Все связи активные, с фильтрацией **Single** от измерения к факту.

```text
dim_customer  1 ─── * fact_experiment * ─── 1 dim_experiment
      │                                            │
      ├─────── 1 ─── * fact_purchases * ─── 1 dim_date
      │                       │                    │
      │                       * ─── 1 dim_product  ├── 1:* experiment_results
      │                                            ├── 1:* validation summary
      └─────── 1 ─── * uplift_predictions         └── 1:* segment results

Model 1 ─── * uplift_predictions
      ├──── 1 ─── * uplift_model_results
      ├──── 1 ─── * uplift_deciles
      └──── 1 ─── * business_scenarios
```

| Измерение `[поле]` | Факт `[поле]` | Связь |
|---|---|---|
| `dim_experiment[experiment_key]` | `experiment_key` во всех фактах эксперимента | 1:* |
| `dim_customer[customer_id]` | `fact_experiment[customer_id]` | 1:* |
| `dim_customer[customer_id]` | `fact_purchases[customer_id]` | 1:* |
| `dim_customer[customer_id]` | `uplift_predictions[customer_id]` | 1:* |
| `dim_date[date_key]` | `fact_purchases[date_key]` | 1:* |
| `dim_product[product_id]` | `fact_purchases[product_id]` | 1:* |
| `Model[model_name]` | `model_name` в uplift-таблицах и business scenarios | 1:* |

`Model` создаётся DAX-кодом из `dax_measures.md`. Не связывайте факты друг с
другом и не включайте двунаправленную фильтрацию: это создаёт несколько путей
между клиентом, экспериментом и моделью.

## Настройки модели

- Пометьте `dim_date` как таблицу дат по `full_date`.
- Скройте ключи, `loaded_at`, `calculated_at` и `scored_at`.
- Не суммируйте rates, p-value, пороги и model scores.
- Rates/uplift/power/ROI: `0.0%;-0.0%;0.0%`; p-value: `0.0000`; counts: целые.
- `month_name` сортируется по `calendar_month`, дециль — по возрастанию.
- Требуйте single select эксперимента, а на страницах uplift/economics — модели.

## What-if параметры

Disconnected-таблицы создаются DAX-кодом и не имеют связей:

- стоимость коммуникации: 0–20 с шагом 0,5;
- прибыль с конверсии: 0–1000 с шагом 10;
- targeting threshold: −20%–50% с шагом 1 п.п.

Это интерфейсные диапазоны, а не бизнес-факты. Перед публикацией их нужно
согласовать с владельцем P&L.

## Обновление и контроль

Порядок: Python pipeline → warehouse loader → SQL quality checks → Power BI
refresh. После обновления проверьте:

1. выбран ровно один эксперимент;
2. `[Users] = control_size + treatment_size`;
3. `[Conversions] = control_conversions + treatment_conversions`;
4. A/B показатели совпадают с `experiment_results`;
5. scored clients не превышают размер аудитории;
6. business scenarios имеют `scenario_analysis = TRUE`.

## Ограничения данных

PostgreSQL хранит сводки A/A и bootstrap, но не отдельные симуляции. Поэтому
отчёт показывает false-positive rate и bootstrap CI, но не гистограммы этих
распределений. `uplift_deciles` даёт децильную аппроксимацию Qini; точечная
кривая Python пока не загружается в БД. Эти визуалы нельзя имитировать на
искусственных данных.

