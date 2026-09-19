# DAX-меры

Формулы рассчитаны на связи из `data_model.md` и фильтр одного эксперимента.
Статистические тесты в DAX не пересчитываются.

## Служебные таблицы

```DAX
Model =
DISTINCT (
    UNION (
        SELECTCOLUMNS ( uplift_model_results, "model_name", uplift_model_results[model_name] ),
        SELECTCOLUMNS ( uplift_predictions, "model_name", uplift_predictions[model_name] ),
        SELECTCOLUMNS ( uplift_deciles, "model_name", uplift_deciles[model_name] ),
        SELECTCOLUMNS ( business_scenarios, "model_name", business_scenarios[model_name] )
    )
)

Communication Cost = GENERATESERIES ( 0, 20, 0.5 )
Profit per Conversion = GENERATESERIES ( 0, 1000, 10 )
Targeting Threshold = GENERATESERIES ( -0.20, 0.50, 0.01 )
```

Переименуйте `Value` в каждой What-if таблице по имени самой таблицы.

## KPI эксперимента

```DAX
Users = DISTINCTCOUNT ( fact_experiment[customer_id] )

Control Users =
CALCULATE ( [Users], KEEPFILTERS ( fact_experiment[treatment] = 0 ) )

Treatment Users =
CALCULATE ( [Users], KEEPFILTERS ( fact_experiment[treatment] = 1 ) )

Conversions = SUM ( fact_experiment[target] )

Control Conversions =
CALCULATE ( [Conversions], KEEPFILTERS ( fact_experiment[treatment] = 0 ) )

Treatment Conversions =
CALCULATE ( [Conversions], KEEPFILTERS ( fact_experiment[treatment] = 1 ) )

Control CR = DIVIDE ( [Control Conversions], [Control Users] )
Treatment CR = DIVIDE ( [Treatment Conversions], [Treatment Users] )
Absolute Uplift = [Treatment CR] - [Control CR]
Relative Uplift = DIVIDE ( [Absolute Uplift], [Control CR] )
Incremental Conversions = [Absolute Uplift] * [Treatment Users]
```

Последняя мера оценивает эффект фактической treatment-группы. Для модельного
таргетинга используется отдельная мера ниже.

## Готовые статистические результаты

```DAX
P-value = SELECTEDVALUE ( experiment_results[p_value] )
CI Lower = SELECTEDVALUE ( experiment_results[ci_lower] )
CI Upper = SELECTEDVALUE ( experiment_results[ci_upper] )
Achieved Power = SELECTEDVALUE ( experiment_results[achieved_power] )
MDE = SELECTEDVALUE ( experiment_results[mde_absolute] )

Statistical Significance =
IF (
    NOT ISBLANK ( [P-value] ),
    IF ( SELECTEDVALUE ( experiment_results[statistically_significant] ), "Да", "Нет" )
)

Experiment Status =
VAR HasSRM = SELECTEDVALUE ( experiment_validation_summary[srm_detected] )
VAR IsSignificant = SELECTEDVALUE ( experiment_results[statistically_significant] )
RETURN
    SWITCH (
        TRUE (),
        ISBLANK ( [P-value] ), "Нет расчёта",
        HasSRM = TRUE (), "Ошибка рандомизации: SRM",
        IsSignificant = TRUE () && [Absolute Uplift] > 0, "Значимый положительный эффект",
        IsSignificant = TRUE () && [Absolute Uplift] < 0, "Значимый отрицательный эффект",
        "Статистически незначимо"
    )
```

## Uplift-модель

```DAX
AUUC = SELECTEDVALUE ( uplift_model_results[auuc] )
Qini Coefficient = SELECTEDVALUE ( uplift_model_results[qini_coefficient] )
Scored Customers = DISTINCTCOUNT ( uplift_predictions[customer_id] )
Average Predicted Uplift = AVERAGE ( uplift_predictions[predicted_uplift] )

Qini Random Baseline =
VAR CurrentDecile = MAX ( uplift_deciles[decile] )
VAR LastDecile = MAXX ( ALLSELECTED ( uplift_deciles[decile] ), uplift_deciles[decile] )
VAR FinalGain =
    MAXX (
        FILTER ( ALLSELECTED ( uplift_deciles ), uplift_deciles[decile] = LastDecile ),
        uplift_deciles[cumulative_incremental_purchases]
    )
RETURN
    FinalGain * DIVIDE ( CurrentDecile, LastDecile )
```

Линия `cumulative_incremental_purchases` вместе с baseline образует децильное
представление Qini. AUUC и Qini coefficient поступают из точного Python-расчёта.

## Динамический What-if таргетинг

```DAX
Selected Communication Cost =
SELECTEDVALUE ( 'Communication Cost'[Communication Cost], 2 )

Selected Profit per Conversion =
SELECTEDVALUE ( 'Profit per Conversion'[Profit per Conversion], 100 )

Selected Targeting Threshold =
SELECTEDVALUE ( 'Targeting Threshold'[Targeting Threshold], 0.02 )

Customers Targeted =
VAR Threshold = [Selected Targeting Threshold]
RETURN
    CALCULATE (
        DISTINCTCOUNT ( uplift_predictions[customer_id] ),
        KEEPFILTERS ( uplift_predictions[predicted_uplift] > Threshold )
    )

Target Share = DIVIDE ( [Customers Targeted], [Scored Customers] )

Expected Incremental Conversions =
VAR Threshold = [Selected Targeting Threshold]
RETURN
    CALCULATE (
        SUM ( uplift_predictions[predicted_uplift] ),
        KEEPFILTERS ( uplift_predictions[predicted_uplift] > Threshold )
    )

Campaign Cost = [Customers Targeted] * [Selected Communication Cost]
Incremental Value = [Expected Incremental Conversions] * [Selected Profit per Conversion]
Incremental Profit = [Incremental Value] - [Campaign Cost]
ROI = DIVIDE ( [Incremental Profit], [Campaign Cost] )
```

Это scenario analysis, а не фактическая бухгалтерская прибыль.

## Сравнение с Target All

```DAX
Target All Customers = [Scored Customers]
Target All Incremental Conversions = SUM ( uplift_predictions[predicted_uplift] )
Target All Campaign Cost = [Target All Customers] * [Selected Communication Cost]

Target All Incremental Profit =
    [Target All Incremental Conversions] * [Selected Profit per Conversion]
        - [Target All Campaign Cost]

Target All ROI = DIVIDE ( [Target All Incremental Profit], [Target All Campaign Cost] )
Profit Delta vs Target All = [Incremental Profit] - [Target All Incremental Profit]
Cost Saving vs Target All = [Target All Campaign Cost] - [Campaign Cost]
```

Красный/зелёный применяйте только к статусу и дублируйте текстом. Отрицательные
uplift и ROI отображайте со знаком, не заменяя нулём.
