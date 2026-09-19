-- Узкие представления для BI: сложная статистика уже рассчитана в Python.

CREATE OR REPLACE VIEW analytics.vw_experiment_overview AS
SELECT
    e.experiment_key,
    e.experiment_name,
    e.experiment_start,
    r.control_size,
    r.treatment_size,
    r.control_cr,
    r.treatment_cr,
    r.absolute_uplift,
    r.relative_uplift,
    r.p_value,
    r.ci_lower,
    r.ci_upper,
    r.achieved_power,
    r.mde_absolute,
    r.statistically_significant,
    v.srm_detected,
    v.srm_p_value,
    v.aa_false_positive_rate,
    v.bootstrap_ci_lower,
    v.bootstrap_ci_upper
FROM analytics.dim_experiment AS e
LEFT JOIN analytics.experiment_results AS r USING (experiment_key)
LEFT JOIN analytics.experiment_validation_summary AS v USING (experiment_key);

CREATE OR REPLACE VIEW analytics.vw_segment_performance AS
SELECT e.experiment_name, s.*
FROM analytics.segment_experiment_results AS s
JOIN analytics.dim_experiment AS e USING (experiment_key);

CREATE OR REPLACE VIEW analytics.vw_uplift_model_performance AS
SELECT e.experiment_name, m.*
FROM analytics.uplift_model_results AS m
JOIN analytics.dim_experiment AS e USING (experiment_key);

CREATE OR REPLACE VIEW analytics.vw_business_strategy_comparison AS
SELECT e.experiment_name, b.*
FROM analytics.business_scenarios AS b
JOIN analytics.dim_experiment AS e USING (experiment_key);
