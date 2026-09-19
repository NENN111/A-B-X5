-- Запрос должен вернуть нули во всех строках после успешной загрузки.
SELECT 'orphan_purchase_customer' AS check_name, count(*) AS violations
FROM analytics.fact_purchases AS f
LEFT JOIN analytics.dim_customer AS d USING (customer_id)
WHERE d.customer_id IS NULL
UNION ALL
SELECT 'orphan_purchase_product', count(*)
FROM analytics.fact_purchases AS f
LEFT JOIN analytics.dim_product AS d USING (product_id)
WHERE f.product_id IS NOT NULL AND d.product_id IS NULL
UNION ALL
SELECT 'orphan_purchase_date', count(*)
FROM analytics.fact_purchases AS f
LEFT JOIN analytics.dim_date AS d USING (date_key)
WHERE d.date_key IS NULL
UNION ALL
SELECT 'invalid_experiment_binary', count(*)
FROM analytics.fact_experiment
WHERE treatment NOT IN (0, 1) OR target NOT IN (0, 1)
UNION ALL
SELECT 'uplift_probability_mismatch', count(*)
FROM analytics.uplift_predictions
WHERE abs((probability_treatment - probability_control) - predicted_uplift) > 1e-9
UNION ALL
SELECT 'non_scenario_business_row', count(*)
FROM analytics.business_scenarios
WHERE scenario_analysis IS NOT TRUE;
