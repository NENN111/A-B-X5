# Uplift-моделирование

## Цель

Модели оценивают условный индивидуальный эффект коммуникации:

```text
uplift(X) = P(Y=1 | T=1, X) - P(Y=1 | T=0, X)
```

Это разность двух модельных вероятностей, а не наблюдаемый индивидуальный counterfactual. Для одного клиента одновременно наблюдается только один фактический outcome.

Фактические метрики в репозитории не приведены, поскольку реальная клиентская витрина X5 RetailHero не материализована.

## Реализованные модели

### S-Learner

`src/models/s_learner.py` обучает одну `LogisticRegression` на pre-treatment признаках и treatment. При scoring одна и та же строка клиента оценивается дважды: с `T=0` и `T=1`.

### T-Learner

`src/models/t_learner.py` обучает две независимые `LogisticRegression`: отдельно на control и treatment. Разность их прогнозов образует predicted uplift.

Логистическая регрессия выбрана как интерпретируемый baseline. CatBoost и LightGBM не добавлены: этих библиотек нет в зависимостях проекта, а без реальных данных невозможно обосновать усложнение модели честным holdout-сравнением.

## Признаки и защита от leakage

По умолчанию используются все поля клиентской feature mart, кроме `client_id`, `treatment` и `target`. Витрина построена только на данных до начала эксперимента. Дополнительно modeling layer запрещает включать treatment или target в `feature_columns`.

Числовые признаки заполняются медианой и стандартизируются. Категориальные признаки заполняются значением `Неизвестно` и кодируются `OneHotEncoder(handle_unknown="ignore")`. Временные признаки переводятся в числовое представление. Все preprocessing-компоненты обучаются только на соответствующей обучающей выборке.

List, Struct и Object намеренно не принимаются: такие поля требуют явного осмысленного преобразования.

## Train, holdout и итоговый scoring

Разделение воспроизводимо по `random_state` и стратифицировано по четырём комбинациям `treatment × target`. Это сохраняет экспериментальные группы и оба класса outcome. Пересечение train и holdout исключено.

Процесс разделён на два назначения:

1. S- и T-Learner обучаются на train, а Qini/AUUC считаются только на holdout.
2. После оценки модели переобучаются на всей доступной витрине и формируют `uplift_customer_scores.parquet` для каждого клиента.

Full-data scores нельзя использовать для отчёта о качестве на тех же строках. Источником честных метрик остаётся только `uplift_predictions_holdout.parquet`.

## Uplift-метрики

Обычные accuracy и ROC-AUC не измеряют качество ранжирования по причинному эффекту. Реализованы:

- uplift по децилям;
- cumulative uplift;
- uplift curve;
- Qini curve и Qini coefficient;
- AUUC;
- cumulative incremental purchases.

Кривая строится по убыванию predicted uplift. Для cumulative gain используется transformed outcome с inverse propensity weighting:

```text
Z_i = Y_i T_i / p(T=1) - Y_i (1-T_i) / (1-p(T=1))
```

`cumulative_incremental_purchases` равен накопленной сумме `Z_i`. Qini coefficient — площадь между cumulative gain модели и прямой случайного таргетинга. AUUC — площадь под cumulative uplift curve. Эти оценки корректно интерпретируются при рандомизированном treatment и корректной propensity.

Таблица `uplift_deciles` содержит номер дециля, число клиентов, средний predicted uplift, размеры групп, CR, observed uplift, incremental и cumulative incremental purchases. Если в малом бине отсутствует одна группа, причинная оценка этого бина сохраняется как `null`, а не подменяется нулём.

## Запуск

После материализации клиентской feature mart:

```bash
python -m src.models.model_evaluation \
  --test-size 0.3 \
  --bins 10 \
  --random-state 42
```

Результаты сохраняются в `data/processed/uplift_modeling/`:

- `uplift_predictions_holdout.parquet` — только честные holdout-прогнозы;
- `uplift_customer_scores.parquet` — scoring каждого клиента после final fit;
- `uplift_model_comparison.parquet` — AUUC и Qini двух learner-подходов;
- `uplift_curves.parquet`;
- `uplift_deciles.parquet`.

## Тестовая стратегия

- Контракты S/T-Learner проверяются на диапазон вероятностей, форму результата, направление известного положительного эффекта и воспроизводимость.
- Stratified split проверяется на отсутствие пересечений, сохранение всех четырёх `treatment × target` страт и одинаковый результат при одинаковом seed.
- На детерминированном примере правильное ранжирование обязано иметь Qini и AUUC выше обратного. Это проверяет именно uplift-ranking, а не predictive accuracy.
- Проверяются начало и конец кривой, сохранение числа клиентов в децилях и итоговый incremental effect с известным ручным ответом.
- Leakage, отсутствующие признаки, неправильные labels и нечисловые predictions должны завершаться понятными исключениями.
- Интеграционный тест проходит путь feature frame → split → две модели → holdout evaluation → full-data scoring → Parquet.

## Ограничения

- Метрики требуют рандомизированного treatment; при наблюдательных данных нужна отдельная propensity-модель и анализ overlap.
- Один holdout даёт более шумную оценку, чем cross-fitting или repeated evaluation.
- Logistic S-Learner без явных взаимодействий является намеренно простым baseline.
- Qini и uplift по малым бинам нестабильны; рядом с эффектом всегда нужно анализировать размеры control и treatment.
