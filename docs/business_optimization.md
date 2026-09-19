# Uplift-таргетинг и бизнес-экономика

## Назначение

Модуль переводит predicted uplift в практическое решение о коммуникации и сравнивает его со стратегией Target All. Денежные показатели являются **сценарным анализом**, пока пользователь не передал подтверждённые значения стоимости контакта и прибыли с дополнительной конверсии.

Репозиторий не содержит рассчитанных бизнес-результатов: для этого необходимы реальные X5 RetailHero данные и явно заданные экономические параметры.

## Выбор модели

Если модель не указана явно, используется S- или T-Learner с максимальным Qini coefficient на holdout. Выбор не выполняется по full-data score и не использует target из обучающей выборки.

К бизнес-слою передаются три поля выбранной модели:

```text
probability_control
probability_treatment
predicted_uplift = probability_treatment - probability_control
```

Контракт проверяет диапазон вероятностей, конечность значений и согласованность разности.

## Модельные типы клиентов

`src/business/targeting.py` добавляет интерпретацию:

- `Persuadables` — uplift выше положительного порога;
- `Sleeping Dogs` — uplift ниже отрицательного порога;
- `Sure Things` — эффект около нуля, но средняя прогнозная вероятность outcome высокая;
- `Lost Causes` — эффект около нуля и средняя прогнозная вероятность низкая.

Это приближённые модельные классы. Они не означают, что оба индивидуальных counterfactual outcomes наблюдались. Поле `customer_type_is_model_interpretation = true` сохраняет это ограничение непосредственно в данных.

Практическое решение не зависит от названия типа и использует строгое правило:

```text
is_targeted = predicted_uplift > targeting_threshold
```

## Unit economics

Для любой выбранной аудитории рассчитываются:

```text
expected incremental conversions = sum(predicted_uplift)
marketing cost = customers targeted × communication cost
incremental value = expected incremental conversions × profit per conversion
incremental profit = incremental value - marketing cost
ROI = incremental profit / marketing cost
```

При нулевой стоимости коммуникации ROI сохраняется как `null`, а не как бесконечность. Отрицательный predicted uplift не обрезается: Target All должен учитывать возможный вред коммуникации.

Сравниваются две стратегии:

- `target_all` — коммуникация со всей аудиторией;
- `uplift_targeting` — только клиенты выше заданного порога.

Каждая строка содержит `scenario_analysis = true`, стоимость контакта и прибыль с конверсии. Таким образом, экономические предположения не теряются при передаче результата в BI.

## Оптимизация порога

`optimize_targeting_threshold` проверяет все достижимые размеры аудитории на границах уникальных predicted uplift, включая вариант не таргетировать никого. Лучшим считается порог с максимальной expected incremental profit; при равной прибыли выбирается меньшая аудитория.

Оптимум является модельно-сценарным. Он зависит одновременно от качества uplift-score, стоимости коммуникации и прибыли с конверсии и не должен интерпретироваться как гарантированный фактический финансовый результат.

## Запуск

Экономические параметры обязательны и задаются явно:

```bash
python -m src.business.optimization \
  --communication-cost 2.0 \
  --profit-per-conversion 100.0 \
  --targeting-threshold 0.02
```

Дополнительные параметры:

- `--model s_learner|t_learner` — ручной выбор вместо лучшего holdout Qini;
- `--effect-threshold` — граница Persuadables/Sleeping Dogs;
- `--outcome-probability-threshold` — разделение Sure Things и Lost Causes.

Результаты сохраняются в `data/processed/business_optimization/`:

- `customer_targeting.parquet`;
- `business_scenarios.parquet`;
- `threshold_optimization.parquet`.

## Тестовая стратегия

- Типы клиентов проверяются на четырёх ручных случаях, включая положительный, отрицательный и околонулевой эффект.
- Строгое условие `>` проверяется отдельно: клиент с uplift, равным threshold, не включается.
- Target All и Uplift Targeting сверяются с ручными расчётами conversions, cost, profit и ROI.
- Нулевая стоимость обязана давать `ROI = null`, а не `inf` или деление на ноль.
- Оптимизатор должен найти известный максимум прибыли и сохранить вариант пустой аудитории.
- Автовыбор модели проверяется по holdout Qini.
- Интеграционный тест проходит путь model scores → типы → решение → economics → три Parquet-файла.

## Ограничения

- Сумма predicted uplift является ожидаемой, а не гарантированной величиной дополнительных покупок.
- In-sample full-data scores подходят для формирования операционной аудитории, но не для повторной оценки качества модели.
- Порог следует дополнительно проверять на стабильность во времени и чувствительность к экономическим параметрам.
- При ограничениях бюджета, частоты контактов или каналов потребуется отдельная constrained optimization.
