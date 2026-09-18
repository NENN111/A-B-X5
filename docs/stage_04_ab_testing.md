# Stage 4 — A/B Testing Engine

## Статус данных

Статистический движок реализован и проверен на эталонных примерах. Фактический результат эксперимента не рассчитан, потому что клиентская feature mart ещё не материализована из реальных X5 RetailHero данных.

## Основная гипотеза

Для Conversion Rate используется two-proportion z-test:

```text
H0: CR_treatment = CR_control
H1: CR_treatment != CR_control
```

По умолчанию применяются двусторонняя альтернатива и `alpha = 0.05`. Pooled standard error используется только под H0 для z-statistic; доверительный интервал абсолютной разницы использует unpooled standard error.

## Выходные показатели

`ExperimentResult` содержит:

- размеры групп и число конверсий;
- Control CR и Treatment CR;
- absolute uplift и relative uplift;
- unpooled standard error разницы;
- pooled null standard error;
- z-statistic и p-value;
- 95% Wald CI абсолютного uplift;
- Wilson CI для CR каждой группы;
- signed Cohen’s h;
- achieved power;
- required sample size для заданной мощности;
- MDE как минимальный положительный absolute uplift;
- решение о статистической значимости;
- предупреждения о неопределённых или недостижимых расчётах.

Relative uplift возвращается как `null`, если Control CR равна нулю. Это предотвращает бесконечные и вводящие в заблуждение проценты.

## Power analysis

Power и sample size рассчитываются через `statsmodels.NormalIndPower` и arcsine effect size двух долей. MDE находится численно через `scipy.optimize.brentq`.

Важно:

- плановый sample size должен рассчитываться до эксперимента по бизнес-значимому ожидаемому эффекту;
- achieved power и required sample size по уже наблюдаемому эффекту являются диагностикой, а не новым доказательством значимости;
- текущий MDE описывает минимальный положительный uplift при фактических размерах групп и baseline control.

## Запуск

После материализации Stage 3:

```bash
python -m src.experiments.ab_test
```

По умолчанию читается `data/processed/customer_features.parquet`, а результат сохраняется в `data/processed/experiment_results.parquet`.

Параметры:

```bash
python -m src.experiments.ab_test \
  --alpha 0.05 \
  --target-power 0.8 \
  --alternative two-sided
```

## Стратегия тестирования и её польза

### Эталонные тесты

z-statistic, p-value и Wald CI сравниваются с независимой реализацией `statsmodels`. Это защищает от ошибки в знаке эффекта, pooled variance и хвостах распределения.

### Ручные примеры

CR, absolute uplift, relative uplift и Cohen’s h проверяются на простых counts. Такие тесты остаются понятными без знания реализации библиотеки.

### Инварианты

При перестановке treatment и control двусторонний p-value сохраняется, а знак uplift и z-statistic меняется. Нарушение свойства указывало бы на асимметричную формулу.

### Confidence interval и решение

На эталонном значимом примере CI исключает ноль. Это проверяет согласованность направления эффекта, интервала и statistical decision.

### Edge cases

Проверяются пустые/некорректные counts, небинарные значения, отсутствующая группа, 0% baseline и одинаковые rates. Польза — контролируемые `None` или исключения вместо `NaN`, infinity и тихо неверных KPI.

### Power, sample size и MDE

Тесты подтверждают, что power растёт с выборкой, MDE уменьшается, а рассчитанный sample size действительно обеспечивает целевую мощность. Это ловит ошибки allocation ratio и направления effect size.

### Интеграция

Отдельный тест проходит путь Polars frame → агрегирование групп → статистика → Parquet. Он обнаруживает ошибки контракта между Stage 3 и будущими PostgreSQL-витринами.

## Ограничения

- Wald CI разницы рассчитан для large-sample inference; при малых группах или редких событиях потребуется score/exact sensitivity analysis.
- Последовательное наблюдение за p-value без alpha-spending не поддерживается.
- SRM, A/A simulation и bootstrap относятся к Stage 5.
- Множественные сравнения сегментов будут корректироваться отдельно.
