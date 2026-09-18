"""Интеграционный тест полного слоя валидации эксперимента."""

from pathlib import Path

import polars as pl

from src.experiments.validation import materialize_validation, validate_experiment


def test_full_validation_pipeline_materializes_all_outputs(tmp_path: Path) -> None:
    size = 40
    frame = pl.DataFrame(
        {
            "age": [22, 31, 43, 57] * (size // 4),
            "gender": ["M", "F"] * (size // 2),
            "rfm_segment": ["R1F1M1", "R5F5M5"] * (size // 2),
            "spending_quantile": [1, 2, 4, 5] * (size // 4),
            "purchase_frequency_quantile": [1, 2, 4, 5] * (size // 4),
            "recency_score": [1, 2, 4, 5] * (size // 4),
            "treatment": [0] * (size // 2) + [1] * (size // 2),
            "target": [0, 1, 0, 0, 1] * (size // 5),
        }
    ).lazy()

    result = validate_experiment(
        frame,
        aa_simulations=20,
        bootstrap_iterations=30,
        random_state=5,
    )
    paths = materialize_validation(result, tmp_path / "validation")

    assert set(paths) == {
        "srm",
        "aa_summary",
        "aa_simulations",
        "bootstrap_summary",
        "bootstrap_distribution",
        "segments",
    }
    assert all(path.exists() for path in paths.values())
    assert pl.read_parquet(paths["aa_simulations"]).height == 20
    assert pl.read_parquet(paths["bootstrap_distribution"]).height == 30
    assert pl.read_parquet(paths["segments"]).height > 0
