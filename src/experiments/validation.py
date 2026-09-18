"""Оркестрация полного слоя валидации A/B-эксперимента."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import polars as pl

from src.config.settings import get_settings
from src.experiments.aa_test import AATestResult, run_aa_from_frame
from src.experiments.ab_test import ExperimentInputError, analyze_experiment_frame
from src.experiments.bootstrap import BootstrapResult, bootstrap_uplift_frame
from src.experiments.segment_analysis import analyze_segments
from src.experiments.srm import SRMResult, check_srm_frame


@dataclass(frozen=True, slots=True)
class ValidationResult:
    """Результаты всех диагностик Stage 5."""

    srm: SRMResult
    aa: AATestResult
    bootstrap: BootstrapResult
    segments: pl.DataFrame


def validate_experiment(
    frame: pl.LazyFrame,
    *,
    treatment_column: str = "treatment",
    target_column: str = "target",
    expected_treatment_share: float = 0.5,
    alpha: float = 0.05,
    aa_simulations: int = 1_000,
    bootstrap_iterations: int = 10_000,
    random_state: int | None = 42,
) -> ValidationResult:
    """Проверить контракт данных и выполнить все диагностики Stage 5."""
    analyze_experiment_frame(
        frame,
        treatment_column=treatment_column,
        target_column=target_column,
        alpha=alpha,
    )
    return ValidationResult(
        srm=check_srm_frame(
            frame,
            treatment_column=treatment_column,
            expected_treatment_share=expected_treatment_share,
            alpha=alpha,
        ),
        aa=run_aa_from_frame(
            frame,
            treatment_column=treatment_column,
            target_column=target_column,
            simulations=aa_simulations,
            alpha=alpha,
            random_state=random_state,
        ),
        bootstrap=bootstrap_uplift_frame(
            frame,
            treatment_column=treatment_column,
            target_column=target_column,
            iterations=bootstrap_iterations,
            confidence_level=1 - alpha,
            random_state=random_state,
        ),
        segments=analyze_segments(
            frame,
            treatment_column=treatment_column,
            target_column=target_column,
            alpha=alpha,
        ),
    )


def _write_parquet_atomic(frame: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{uuid4().hex}.tmp.parquet")
    try:
        frame.write_parquet(temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def materialize_validation(
    result: ValidationResult, output_dir: Path
) -> dict[str, Path]:
    """Атомарно сохранить сводки и распределения для BI и notebook."""
    paths = {
        "srm": output_dir / "srm.parquet",
        "aa_summary": output_dir / "aa_summary.parquet",
        "aa_simulations": output_dir / "aa_simulations.parquet",
        "bootstrap_summary": output_dir / "bootstrap_summary.parquet",
        "bootstrap_distribution": output_dir / "bootstrap_distribution.parquet",
        "segments": output_dir / "segment_analysis.parquet",
    }
    _write_parquet_atomic(pl.DataFrame([result.srm.to_record()]), paths["srm"])
    _write_parquet_atomic(
        pl.DataFrame([result.aa.summary.to_record()]), paths["aa_summary"]
    )
    _write_parquet_atomic(result.aa.simulations, paths["aa_simulations"])
    _write_parquet_atomic(
        pl.DataFrame([result.bootstrap.summary.to_record()]),
        paths["bootstrap_summary"],
    )
    _write_parquet_atomic(
        result.bootstrap.distribution, paths["bootstrap_distribution"]
    )
    _write_parquet_atomic(result.segments, paths["segments"])
    return paths


def main() -> None:
    """Запустить Stage 5 из командной строки."""
    parser = argparse.ArgumentParser(description="Валидация A/B-эксперимента.")
    parser.add_argument("--input-path", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--treatment-column", default="treatment")
    parser.add_argument("--target-column", default="target")
    parser.add_argument("--expected-treatment-share", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--aa-simulations", type=int, default=1_000)
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()
    settings = get_settings()
    input_path = (
        args.input_path or settings.processed_data_dir / "customer_features.parquet"
    )
    output_dir = (
        args.output_dir or settings.processed_data_dir / "experiment_validation"
    )
    if not input_path.exists():
        parser.error(
            f"Feature mart не найдена: {input_path}. Сначала выполните Stage 3 на реальных данных."
        )
    try:
        result = validate_experiment(
            pl.scan_parquet(input_path),
            treatment_column=args.treatment_column,
            target_column=args.target_column,
            expected_treatment_share=args.expected_treatment_share,
            alpha=args.alpha,
            aa_simulations=args.aa_simulations,
            bootstrap_iterations=args.bootstrap_iterations,
            random_state=args.random_state,
        )
    except (ExperimentInputError, ValueError) as error:
        parser.error(str(error))
    paths = materialize_validation(result, output_dir)
    print(
        f"Stage 5 завершён: SRM={result.srm.srm_detected}, "
        f"A/A FPR={result.aa.summary.false_positive_rate:.4f}, "
        f"артефакты={len(paths)}, каталог={output_dir}"
    )


if __name__ == "__main__":
    main()
