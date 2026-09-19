# Спецификация отчёта Power BI

## Общая компоновка

Холст 16:9, светлый фон, сетка 12 колонок. Сверху — название и дата обновления,
слева — единая панель фильтров. На всех страницах синхронен single-select
`experiment_name`; на страницах 4–5 также `model_name`. Tooltips показывают
размер выборки и контекст.

## 1. Executive Overview

- KPI: Control Users, Treatment Users, Control CR, Treatment CR, Absolute
  Uplift, Relative Uplift, P-value и Incremental Conversions.
- Карточка `Experiment Status` с семантическим цветом.
- Clustered column: Control CR против Treatment CR.
- Error bar: Absolute Uplift с `CI Lower` и `CI Upper`.
- Блок качества: Achieved Power, MDE и SRM status.
- Сноска: причинный вывод допустим только при корректной рандомизации;
  отсутствие значимости не доказывает отсутствие эффекта.

## 2. Experiment Diagnostics

- Cards: sample size, Achieved Power, MDE, SRM p-value, A/A false positive
  rate, число A/A и bootstrap итераций.
- Status visual для SRM: `srm_detected = FALSE` — норма.
- Интервал bootstrap: observed uplift, mean/median и CI.
- Таблица контрольных показателей с исходным p-value A/B-теста.
- Предупреждение: PostgreSQL содержит summary, а не строки A/A и bootstrap;
  гистограммы распределений до расширения warehouse не строятся.

## 3. Customer Segments

- Slicer `dimension`, bar chart `absolute_uplift` по `segment`.
- Error bars по `ci_lower` и `ci_upper`.
- Scatter: размер сегмента по X, absolute uplift по Y, цвет —
  `significant_adjusted`.
- Matrix: segment, размеры групп, CR, uplift, raw и adjusted p-value.
- Значимость обозначается только по `significant_adjusted`, после поправки
  Benjamini–Hochberg.

## 4. Uplift Modeling

- Cards: AUUC, Qini Coefficient, overall observed uplift, holdout customers.
- Сравнительная таблица S-Learner и T-Learner.
- Combo chart по децилям: avg predicted uplift и observed uplift.
- Line chart: cumulative incremental purchases и `Qini Random Baseline`.
- Histogram строится из `uplift_predictions`; исходный score не округляется.
- Подпись: дециль 1 содержит максимальный predicted uplift; score не является
  известным индивидуальным counterfactual.

## 5. Business Impact

- What-if slicers: Communication Cost, Profit per Conversion и Threshold.
- KPI: Customers Targeted, Target Share, Campaign Cost, Expected Incremental
  Conversions, Incremental Profit и ROI.
- Сравнение Target All и Uplift Targeting по cost, conversions, profit и ROI.
- Line chart по threshold для сохранённых Python-сценариев. Если загружен один
  сценарий, визуал скрывается и используются динамические DAX-карточки.
- Всегда видимый бейдж «Сценарный расчёт»; валюта и assumptions — в subtitle.

## Взаимодействия и защита от неверного чтения

- Выбор сегмента не фильтрует model/economics: это разные уровни анализа.
- Модель фильтрует predictions, model results, deciles и scenarios через
  `Model`, но не меняет A/B-результат.
- Для KPI Executive Overview отключите cross-highlighting нижних визуалов.
- При множественном выборе экспериментов или моделей `SELECTEDVALUE` KPI
  должны быть пустыми, а не усреднёнными.
- P-value не интерпретируется как размер эффекта или вероятность истинности H0.

## Визуальный стандарт и доступность

- Использовать `theme.json`; текст тёмно-синий, фон почти белый.
- Красный/зелёный только для статуса и всегда с текстом или значком.
- Основной текст от 11 pt, заголовки visual от 12 pt.
- Не более шести KPI в ряду; числа выравниваются вправо.
- Проценты — один знак, p-value — четыре, деньги — два.
- Для каждого visual задать alt text и логичный tab order.

## Приёмка

Refresh проходит без локальных путей и паролей, связи соответствуют модели,
контрольные суммы сходятся с SQL, фильтры и What-if работают независимо от
канонических A/B KPI, а ограничения данных видимы пользователю.

