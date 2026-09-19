"""Собрать версионируемый PBIP/PBIR-проект Retail Experimentation."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

REPORT_SCHEMA = (
    "https://developer.microsoft.com/json-schemas/fabric/item/report/"
    "definition/report/3.1.0/schema.json"
)
PAGE_SCHEMA = (
    "https://developer.microsoft.com/json-schemas/fabric/item/report/"
    "definition/page/2.1.0/schema.json"
)
VISUAL_SCHEMA = (
    "https://developer.microsoft.com/json-schemas/fabric/item/report/"
    "definition/visualContainer/2.9.0/schema.json"
)


TABLES: dict[str, list[tuple[str, str]]] = {
    "dim_experiment": [
        ("experiment_key", "int64"),
        ("experiment_name", "string"),
        ("experiment_start", "dateTime"),
        ("expected_treatment_share", "double"),
        ("description", "string"),
    ],
    "fact_experiment": [
        ("experiment_key", "int64"),
        ("customer_id", "string"),
        ("treatment", "int64"),
        ("target", "int64"),
    ],
    "experiment_results": [
        ("experiment_key", "int64"),
        ("control_size", "int64"),
        ("treatment_size", "int64"),
        ("control_conversions", "int64"),
        ("treatment_conversions", "int64"),
        ("control_cr", "double"),
        ("treatment_cr", "double"),
        ("absolute_uplift", "double"),
        ("relative_uplift", "double"),
        ("p_value", "double"),
        ("ci_lower", "double"),
        ("ci_upper", "double"),
        ("achieved_power", "double"),
        ("mde_absolute", "double"),
        ("statistically_significant", "boolean"),
    ],
    "experiment_validation_summary": [
        ("experiment_key", "int64"),
        ("srm_p_value", "double"),
        ("srm_detected", "boolean"),
        ("aa_simulations", "int64"),
        ("aa_false_positive_rate", "double"),
        ("bootstrap_iterations", "int64"),
        ("observed_uplift", "double"),
        ("bootstrap_mean_uplift", "double"),
        ("bootstrap_median_uplift", "double"),
        ("bootstrap_ci_lower", "double"),
        ("bootstrap_ci_upper", "double"),
    ],
    "segment_experiment_results": [
        ("experiment_key", "int64"),
        ("dimension", "string"),
        ("segment", "string"),
        ("control_size", "int64"),
        ("treatment_size", "int64"),
        ("control_cr", "double"),
        ("treatment_cr", "double"),
        ("absolute_uplift", "double"),
        ("relative_uplift", "double"),
        ("ci_lower", "double"),
        ("ci_upper", "double"),
        ("p_value", "double"),
        ("adjusted_p_value", "double"),
        ("significant_adjusted", "boolean"),
    ],
    "uplift_model_results": [
        ("experiment_key", "int64"),
        ("model_name", "string"),
        ("train_customers", "int64"),
        ("holdout_customers", "int64"),
        ("auuc", "double"),
        ("qini_coefficient", "double"),
        ("overall_observed_uplift", "double"),
        ("final_incremental_purchases", "double"),
    ],
    "uplift_predictions": [
        ("experiment_key", "int64"),
        ("customer_id", "string"),
        ("model_name", "string"),
        ("probability_control", "double"),
        ("probability_treatment", "double"),
        ("predicted_uplift", "double"),
    ],
    "uplift_deciles": [
        ("experiment_key", "int64"),
        ("model_name", "string"),
        ("decile", "int64"),
        ("customers", "int64"),
        ("control_size", "int64"),
        ("treatment_size", "int64"),
        ("avg_predicted_uplift", "double"),
        ("treatment_cr", "double"),
        ("control_cr", "double"),
        ("observed_uplift", "double"),
        ("incremental_purchases", "double"),
        ("cumulative_incremental_purchases", "double"),
    ],
    "business_scenarios": [
        ("experiment_key", "int64"),
        ("model_name", "string"),
        ("scenario_name", "string"),
        ("strategy", "string"),
        ("customers_total", "int64"),
        ("customers_targeted", "int64"),
        ("target_share", "double"),
        ("targeting_threshold", "double"),
        ("communication_cost", "decimal"),
        ("profit_per_conversion", "decimal"),
        ("marketing_cost", "decimal"),
        ("incremental_conversions", "double"),
        ("incremental_profit", "decimal"),
        ("roi", "double"),
        ("scenario_analysis", "boolean"),
    ],
}


MEASURES = [
    ("Users", "DISTINCTCOUNT(fact_experiment[customer_id])", "#,##0"),
    ("Control Users", "CALCULATE([Users], fact_experiment[treatment] = 0)", "#,##0"),
    ("Treatment Users", "CALCULATE([Users], fact_experiment[treatment] = 1)", "#,##0"),
    ("Conversions", "SUM(fact_experiment[target])", "#,##0"),
    ("Control CR", "DIVIDE(CALCULATE([Conversions], fact_experiment[treatment] = 0), [Control Users])", "0.0%"),
    ("Treatment CR", "DIVIDE(CALCULATE([Conversions], fact_experiment[treatment] = 1), [Treatment Users])", "0.0%"),
    ("Conversion Rate", "DIVIDE([Conversions], [Users])", "0.0%"),
    ("Absolute Uplift", "[Treatment CR] - [Control CR]", "+0.00%;-0.00%;0.00%"),
    ("Relative Uplift", "DIVIDE([Absolute Uplift], [Control CR])", "+0.0%;-0.0%;0.0%"),
    ("P-value", "SELECTEDVALUE(experiment_results[p_value])", "0.0000"),
    ("CI Lower", "SELECTEDVALUE(experiment_results[ci_lower])", "+0.00%;-0.00%;0.00%"),
    ("CI Upper", "SELECTEDVALUE(experiment_results[ci_upper])", "+0.00%;-0.00%;0.00%"),
    ("Achieved Power", "SELECTEDVALUE(experiment_results[achieved_power])", "0.0%"),
    ("MDE", "SELECTEDVALUE(experiment_results[mde_absolute])", "0.00%"),
    ("SRM P-value", "SELECTEDVALUE(experiment_validation_summary[srm_p_value])", "0.0000"),
    ("A/A FPR", "SELECTEDVALUE(experiment_validation_summary[aa_false_positive_rate])", "0.0%"),
    ("Bootstrap Mean", "SELECTEDVALUE(experiment_validation_summary[bootstrap_mean_uplift])", "+0.00%;-0.00%;0.00%"),
    ("Bootstrap CI Lower", "SELECTEDVALUE(experiment_validation_summary[bootstrap_ci_lower])", "+0.00%;-0.00%;0.00%"),
    ("Bootstrap CI Upper", "SELECTEDVALUE(experiment_validation_summary[bootstrap_ci_upper])", "+0.00%;-0.00%;0.00%"),
    ("Segment Uplift", "MAX(segment_experiment_results[absolute_uplift])", "+0.00%;-0.00%;0.00%"),
    ("Segment Users", "SUM(segment_experiment_results[control_size]) + SUM(segment_experiment_results[treatment_size])", "#,##0"),
    ("Model AUUC", "MAX(uplift_model_results[auuc])", "0.0000"),
    ("Qini Coefficient", "MAX(uplift_model_results[qini_coefficient])", "#,##0.00"),
    ("Holdout Customers", "MAX(uplift_model_results[holdout_customers])", "#,##0"),
    ("Scored Customers", "DISTINCTCOUNT(uplift_predictions[customer_id])", "#,##0"),
    ("Average Predicted Uplift", "AVERAGE(uplift_predictions[predicted_uplift])", "+0.00%;-0.00%;0.00%"),
    ("Observed Decile Uplift", "MAX(uplift_deciles[observed_uplift])", "+0.00%;-0.00%;0.00%"),
    ("Predicted Decile Uplift", "MAX(uplift_deciles[avg_predicted_uplift])", "+0.00%;-0.00%;0.00%"),
    ("Cumulative Incremental", "MAX(uplift_deciles[cumulative_incremental_purchases])", "#,##0.0"),
    ("Selected Communication Cost", "SELECTEDVALUE('Communication Cost'[Communication Cost], 2)", "#,##0.00"),
    ("Selected Profit per Conversion", "SELECTEDVALUE('Profit per Conversion'[Profit per Conversion], 100)", "#,##0.00"),
    ("Selected Targeting Threshold", "SELECTEDVALUE('Targeting Threshold'[Targeting Threshold], 0.02)", "0.0%"),
    ("Customers Targeted", "VAR T=[Selected Targeting Threshold] RETURN CALCULATE(DISTINCTCOUNT(uplift_predictions[customer_id]), KEEPFILTERS(uplift_predictions[predicted_uplift] > T))", "#,##0"),
    ("Target Share", "DIVIDE([Customers Targeted], [Scored Customers])", "0.0%"),
    ("Expected Incremental Conversions", "VAR T=[Selected Targeting Threshold] RETURN CALCULATE(SUM(uplift_predictions[predicted_uplift]), KEEPFILTERS(uplift_predictions[predicted_uplift] > T))", "#,##0.0"),
    ("Campaign Cost", "[Customers Targeted] * [Selected Communication Cost]", "#,##0.00"),
    ("Incremental Profit", "[Expected Incremental Conversions] * [Selected Profit per Conversion] - [Campaign Cost]", "#,##0.00;(#,##0.00)"),
    ("ROI", "DIVIDE([Incremental Profit], [Campaign Cost])", "+0.0%;-0.0%;0.0%"),
    ("Scenario Profit", "MAX(business_scenarios[incremental_profit])", "#,##0.00;(#,##0.00)"),
    ("Scenario Cost", "MAX(business_scenarios[marketing_cost])", "#,##0.00"),
]


def _write(path: Path, content: str | dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, dict):
        text = json.dumps(content, ensure_ascii=False, indent=2) + "\n"
    else:
        text = content.rstrip() + "\n"
    path.write_text(text, encoding="utf-8")


def _id(value: str) -> str:
    return hashlib.sha1(value.encode()).hexdigest()[:20]


def _column(table: str, name: str) -> dict[str, Any]:
    return {
        "field": {"Column": {"Expression": {"SourceRef": {"Entity": table}}, "Property": name}},
        "queryRef": f"{table}.{name}",
        "nativeQueryRef": name,
    }


def _measure(name: str) -> dict[str, Any]:
    return {
        "field": {"Measure": {"Expression": {"SourceRef": {"Entity": "Metrics"}}, "Property": name}},
        "queryRef": f"Metrics.{name}",
        "nativeQueryRef": name,
    }


def _position(x: int, y: int, width: int, height: int, order: int) -> dict[str, int]:
    return {"x": x, "y": y, "z": order, "height": height, "width": width, "tabOrder": order}


def _textbox(name: str, text: str, x: int, y: int, width: int, height: int, order: int, size: int = 24) -> dict[str, Any]:
    return {
        "$schema": VISUAL_SCHEMA,
        "name": name,
        "position": _position(x, y, width, height, order),
        "visual": {
            "visualType": "textbox",
            "objects": {"general": [{"properties": {"paragraphs": [{"textRuns": [{"value": text, "textStyle": {"fontFamily": "Segoe UI Semibold", "fontSize": f"{size}px", "color": "#0F172A"}}], "horizontalTextAlignment": "left"}]}}]},
        },
    }


def _slicer(name: str, table: str, column: str, label: str, x: int, y: int, order: int) -> dict[str, Any]:
    return {
        "$schema": VISUAL_SCHEMA,
        "name": name,
        "position": _position(x, y, 200, 72, order),
        "visual": {
            "visualType": "slicer",
            "query": {"queryState": {"Values": {"projections": [_column(table, column)]}}},
            "objects": {
                "data": [{"properties": {"mode": {"expr": {"Literal": {"Value": "'Dropdown'"}}}}}],
                "header": [{"properties": {"show": {"expr": {"Literal": {"Value": "true"}}}, "text": {"expr": {"Literal": {"Value": f"'{label}'"}}}}}],
            },
        },
    }


def _card(name: str, measures: list[str], x: int, y: int, width: int, height: int, order: int) -> dict[str, Any]:
    return {
        "$schema": VISUAL_SCHEMA,
        "name": name,
        "position": _position(x, y, width, height, order),
        "visual": {"visualType": "cardVisual", "query": {"queryState": {"Data": {"projections": [_measure(item) for item in measures]}}}},
    }


def _chart(name: str, visual_type: str, category: tuple[str, str], measures: list[str], title: str, x: int, y: int, width: int, height: int, order: int) -> dict[str, Any]:
    return {
        "$schema": VISUAL_SCHEMA,
        "name": name,
        "position": _position(x, y, width, height, order),
        "visual": {
            "visualType": visual_type,
            "query": {"queryState": {"Category": {"projections": [_column(*category)]}, "Y": {"projections": [_measure(item) for item in measures]}}},
            "visualContainerObjects": {"title": [{"properties": {"show": {"expr": {"Literal": {"Value": "true"}}}, "text": {"expr": {"Literal": {"Value": f"'{title}'"}}}}}]},
        },
    }


def _table(name: str, fields: list[tuple[str, str] | str], title: str, x: int, y: int, width: int, height: int, order: int) -> dict[str, Any]:
    projections = [_measure(item) if isinstance(item, str) else _column(*item) for item in fields]
    return {
        "$schema": VISUAL_SCHEMA,
        "name": name,
        "position": _position(x, y, width, height, order),
        "visual": {
            "visualType": "tableEx",
            "query": {"queryState": {"Values": {"projections": projections}}},
            "objects": {"columnHeaders": [{"properties": {"columnAdjustment": {"expr": {"Literal": {"Value": "'growToFit'"}}}, "autoSizeColumnWidth": {"expr": {"Literal": {"Value": "true"}}}}}]},
            "visualContainerObjects": {"title": [{"properties": {"show": {"expr": {"Literal": {"Value": "true"}}}, "text": {"expr": {"Literal": {"Value": f"'{title}'"}}}}}]},
        },
    }


def _source_table(name: str, columns: list[tuple[str, str]]) -> str:
    lines = [f"table {name}", ""]
    for column, dtype in columns:
        lines.extend([
            f"\tcolumn {column}",
            f"\t\tdataType: {dtype}",
            "\t\tsummarizeBy: none",
            f"\t\tsourceColumn: {column}",
            "",
        ])
    lines.extend([
        f"\tpartition {name} = m",
        "\t\tmode: import",
        "\t\tsource =",
        "\t\t\t\tlet",
        "\t\t\t\t\tSource = PostgreSQL.Database(Server, Database),",
        f"\t\t\t\t\tData = Source{{[Schema=\"analytics\", Item=\"{name}\"]}}[Data]",
        "\t\t\t\tin",
        "\t\t\t\t\tData",
    ])
    return "\n".join(lines)


def _calculated_table(name: str, values: str, source_column: str, dtype: str = "double") -> str:
    return "\n".join([
        f"table '{name}'",
        "",
        f"\tcolumn '{name}'",
        f"\t\tdataType: {dtype}",
        "\t\tsummarizeBy: none",
        f"\t\tsourceColumn: {source_column}",
        "",
        f"\tpartition '{name}' = calculated",
        "\t\tmode: import",
        f"\t\tsource = {values}",
    ])


def _build_model(model_dir: Path, server: str, database: str) -> None:
    _write(model_dir / "definition.pbism", {
        "$schema": "https://developer.microsoft.com/json-schemas/fabric/item/semanticModel/definitionProperties/1.0.0/schema.json",
        "version": "4.2",
        "settings": {"qnaEnabled": True},
    })
    definition = model_dir / "definition"
    _write(definition / "database.tmdl", "database RetailExperimentation\n\tcompatibilityLevel: 1702\n\tcompatibilityMode: powerBI\n\tlanguage: 1049")
    refs = [*TABLES, "Model", "Metrics", "Communication Cost", "Profit per Conversion", "Targeting Threshold"]
    _write(definition / "model.tmdl", "\n".join([
        "model Model",
        "\tculture: ru-RU",
        "\tdefaultPowerBIDataSourceVersion: powerBI_V3",
        "\tsourceQueryCulture: ru-RU",
        "\tdiscourageImplicitMeasures",
        "",
        *[f"ref table '{name}'" if " " in name else f"ref table {name}" for name in refs],
    ]))
    _write(definition / "expressions.tmdl", "\n".join([
        f'expression Server = "{server}" meta [IsParameterQuery=true, Type="Text", IsParameterQueryRequired=true]',
        f'expression Database = "{database}" meta [IsParameterQuery=true, Type="Text", IsParameterQueryRequired=true]',
    ]))
    for table, columns in TABLES.items():
        _write(definition / "tables" / f"{table}.tmdl", _source_table(table, columns))
    model_source = "DISTINCT(UNION(SELECTCOLUMNS(uplift_model_results, \"model_name\", uplift_model_results[model_name]), SELECTCOLUMNS(uplift_predictions, \"model_name\", uplift_predictions[model_name]), SELECTCOLUMNS(uplift_deciles, \"model_name\", uplift_deciles[model_name])))"
    _write(definition / "tables" / "Model.tmdl", _calculated_table("Model", model_source, "[model_name]", "string").replace("column 'Model'", "column model_name"))
    metric_lines = ["table Metrics", ""]
    for name, dax, fmt in MEASURES:
        metric_lines.extend([f"\t/// {name}", f"\tmeasure '{name}' = {dax}", f"\t\tformatString: {fmt}", ""])
    metric_lines.extend(["\tcolumn Dummy", "\t\tdataType: int64", "\t\tisHidden", "\t\tsourceColumn: [Dummy]", "", "\tpartition Metrics = calculated", "\t\tmode: import", "\t\tsource = ROW(\"Dummy\", 1)"])
    _write(definition / "tables" / "Metrics.tmdl", "\n".join(metric_lines))
    _write(definition / "tables" / "Communication Cost.tmdl", _calculated_table("Communication Cost", "GENERATESERIES(0, 20, 0.5)", "[Value]"))
    _write(definition / "tables" / "Profit per Conversion.tmdl", _calculated_table("Profit per Conversion", "GENERATESERIES(0, 1000, 10)", "[Value]"))
    _write(definition / "tables" / "Targeting Threshold.tmdl", _calculated_table("Targeting Threshold", "GENERATESERIES(-0.20, 0.50, 0.01)", "[Value]"))
    relationships = []
    for table in TABLES:
        if table != "dim_experiment":
            relationships.extend([f"relationship '{table}-experiment'", f"\tfromColumn: {table}.experiment_key", "\ttoColumn: dim_experiment.experiment_key", ""])
    for table in ("uplift_model_results", "uplift_predictions", "uplift_deciles", "business_scenarios"):
        relationships.extend([f"relationship '{table}-model'", f"\tfromColumn: {table}.model_name", "\ttoColumn: Model.model_name", ""])
    _write(definition / "relationships.tmdl", "\n".join(relationships))


def _build_report(report_dir: Path, model_name: str) -> None:
    _write(report_dir / "definition.pbir", {
        "$schema": "https://developer.microsoft.com/json-schemas/fabric/item/report/definitionProperties/2.0.0/schema.json",
        "version": "4.0",
        "datasetReference": {"byPath": {"path": f"../{model_name}.SemanticModel"}},
    })
    definition = report_dir / "definition"
    _write(definition / "version.json", {"$schema": "https://developer.microsoft.com/json-schemas/fabric/item/report/definition/versionMetadata/1.0.0/schema.json", "version": "2.0.0"})
    _write(definition / "report.json", {
        "$schema": REPORT_SCHEMA,
        "themeCollection": {"baseTheme": {"name": "CY24SU10", "reportVersionAtImport": {"visual": "2.6.0", "report": "3.1.0", "page": "2.1.0"}, "type": "SharedResources"}},
        "settings": {"useStylableVisualContainerHeader": True},
    })

    pages: list[tuple[str, str, list[dict[str, Any]]]] = []
    exp = lambda key, order=2: _slicer(_id(key+"exp"), "dim_experiment", "experiment_name", "Эксперимент", 1040, 8, order)
    model = lambda key, order=3: _slicer(_id(key+"model"), "Model", "model_name", "Модель", 824, 8, order)

    pages.append(("Executive Overview", "exec", [
        _textbox(_id("exec-title"), "Executive Overview", 24, 16, 700, 48, 1), exp("exec"),
        _card(_id("exec-users"), ["Control Users", "Treatment Users", "Conversions"], 24, 96, 600, 112, 3),
        _card(_id("exec-effect"), ["Control CR", "Treatment CR", "Absolute Uplift", "Relative Uplift", "P-value"], 640, 96, 600, 112, 4),
        _chart(_id("exec-cr"), "clusteredColumnChart", ("fact_experiment", "treatment"), ["Conversion Rate"], "Конверсия: control (0) vs treatment (1)", 24, 232, 600, 440, 5),
        _table(_id("exec-ci"), ["Absolute Uplift", "CI Lower", "CI Upper", "Achieved Power", "MDE", ("experiment_results", "statistically_significant")], "Статистический вывод", 640, 232, 600, 440, 6),
    ]))
    pages.append(("Experiment Diagnostics", "diagnostics", [
        _textbox(_id("diag-title"), "Experiment Diagnostics", 24, 16, 700, 48, 1), exp("diag"),
        _card(_id("diag-main"), ["Achieved Power", "MDE", "SRM P-value", "A/A FPR"], 24, 96, 1216, 112, 3),
        _table(_id("diag-validation"), [("experiment_validation_summary", "srm_detected"), ("experiment_validation_summary", "aa_simulations"), ("experiment_validation_summary", "bootstrap_iterations"), "Bootstrap Mean", "Bootstrap CI Lower", "Bootstrap CI Upper"], "Контроль качества эксперимента", 24, 232, 1216, 440, 4),
    ]))
    pages.append(("Customer Segments", "segments", [
        _textbox(_id("seg-title"), "Customer Segments", 24, 16, 600, 48, 1), exp("seg"),
        _slicer(_id("seg-dimension"), "segment_experiment_results", "dimension", "Измерение", 824, 8, 3),
        _card(_id("seg-kpi"), ["Segment Users", "Segment Uplift"], 24, 96, 400, 112, 4),
        _chart(_id("seg-chart"), "barChart", ("segment_experiment_results", "segment"), ["Segment Uplift"], "Absolute uplift по сегментам", 24, 232, 600, 440, 5),
        _table(_id("seg-table"), [("segment_experiment_results", "segment"), ("segment_experiment_results", "control_size"), ("segment_experiment_results", "treatment_size"), ("segment_experiment_results", "control_cr"), ("segment_experiment_results", "treatment_cr"), ("segment_experiment_results", "adjusted_p_value"), ("segment_experiment_results", "significant_adjusted")], "Сегменты с BH-поправкой", 640, 232, 600, 440, 6),
    ]))
    pages.append(("Uplift Modeling", "uplift", [
        _textbox(_id("uplift-title"), "Uplift Modeling", 24, 16, 600, 48, 1), exp("uplift"), model("uplift"),
        _card(_id("uplift-kpi"), ["Model AUUC", "Qini Coefficient", "Holdout Customers", "Average Predicted Uplift"], 24, 96, 1216, 112, 4),
        _chart(_id("uplift-deciles"), "clusteredColumnChart", ("uplift_deciles", "decile"), ["Predicted Decile Uplift", "Observed Decile Uplift"], "Predicted и observed uplift по децилям", 24, 232, 600, 440, 5),
        _chart(_id("uplift-qini"), "lineChart", ("uplift_deciles", "decile"), ["Cumulative Incremental"], "Cumulative incremental conversions", 640, 232, 600, 440, 6),
    ]))
    pages.append(("Business Impact", "business", [
        _textbox(_id("biz-title"), "Business Impact · сценарный расчёт", 24, 16, 650, 48, 1), exp("biz"), model("biz"),
        _slicer(_id("biz-threshold"), "Targeting Threshold", "Targeting Threshold", "Порог uplift", 608, 8, 4),
        _card(_id("biz-kpi"), ["Customers Targeted", "Target Share", "Campaign Cost", "Expected Incremental Conversions", "Incremental Profit", "ROI"], 24, 96, 1216, 128, 5),
        _chart(_id("biz-scenario"), "clusteredColumnChart", ("business_scenarios", "strategy"), ["Scenario Profit", "Scenario Cost"], "Target All vs Uplift Targeting", 24, 248, 600, 424, 6),
        _table(_id("biz-assumptions"), [("business_scenarios", "scenario_name"), ("business_scenarios", "strategy"), ("business_scenarios", "targeting_threshold"), ("business_scenarios", "communication_cost"), ("business_scenarios", "profit_per_conversion"), ("business_scenarios", "target_share"), ("business_scenarios", "roi")], "Сохранённые Python-сценарии", 640, 248, 600, 424, 7),
    ]))

    page_order: list[str] = []
    for display, key, visuals in pages:
        page_name = "ReportSection" + _id("page-" + key)
        page_order.append(page_name)
        page_path = definition / "pages" / page_name
        _write(page_path / "page.json", {"$schema": PAGE_SCHEMA, "name": page_name, "displayName": display, "displayOption": "FitToPage", "height": 720, "width": 1280})
        for visual in visuals:
            _write(page_path / "visuals" / visual["name"] / "visual.json", visual)
    _write(definition / "pages" / "pages.json", {"$schema": "https://developer.microsoft.com/json-schemas/fabric/item/report/definition/pagesMetadata/1.1.0/schema.json", "pageOrder": page_order, "activePageName": page_order[0]})


def build(output_dir: Path, server: str, database: str) -> Path:
    """Пересобрать PBIP в контролируемом каталоге."""
    output_dir = output_dir.resolve()
    expected_parent = (Path(__file__).resolve().parents[1] / "powerbi").resolve()
    if expected_parent not in output_dir.parents:
        raise ValueError("PBIP можно генерировать только внутри powerbi/")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    name = "RetailExperimentation"
    _build_model(output_dir / f"{name}.SemanticModel", server, database)
    _build_report(output_dir / f"{name}.Report", name)
    _write(output_dir / f"{name}.pbip", {
        "$schema": "https://developer.microsoft.com/json-schemas/fabric/pbip/pbipProperties/1.0.0/schema.json",
        "version": "1.0",
        "artifacts": [{"report": {"path": f"{name}.Report"}}],
        "settings": {"enableAutoRecovery": True},
    })
    _write(output_dir / ".gitignore", "**/.pbi/localSettings.json\n**/.pbi/cache.abf")
    return output_dir / f"{name}.pbip"


def main() -> None:
    parser = argparse.ArgumentParser(description="Сборка Power BI Project.")
    parser.add_argument("--output-dir", type=Path, default=Path("powerbi/dashboard"))
    parser.add_argument("--server", default="localhost:5432")
    parser.add_argument("--database", default="retail_experiments")
    args = parser.parse_args()
    print(build(args.output_dir, args.server, args.database))


if __name__ == "__main__":
    main()
