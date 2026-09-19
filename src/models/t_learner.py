"""T-Learner с отдельными outcome-моделями экспериментальных групп."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import polars as pl
from sklearn.base import BaseEstimator, clone
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.models.preprocessing import (
    build_preprocessor,
    infer_feature_spec,
    prepare_feature_matrix,
    require_both_outcomes,
    validate_uplift_frame,
)


class TLearner:
    """Две модели для P(Y|T=0,X) и P(Y|T=1,X)."""

    def __init__(
        self,
        *,
        estimator: BaseEstimator | None = None,
        random_state: int = 42,
    ) -> None:
        self.estimator = estimator
        self.random_state = random_state

    def fit(
        self,
        frame: pl.DataFrame,
        feature_columns: Sequence[str],
        *,
        treatment_column: str = "treatment",
        target_column: str = "target",
    ) -> TLearner:
        """Обучить отдельные модели на control и treatment клиентах."""
        validate_uplift_frame(
            frame,
            feature_columns,
            treatment_column=treatment_column,
            target_column=target_column,
        )
        self.feature_spec_ = infer_feature_spec(frame, feature_columns)
        estimator = self.estimator or LogisticRegression(
            max_iter=1_000,
            random_state=self.random_state,
        )
        self.control_pipeline_ = Pipeline(
            [
                ("preprocessor", build_preprocessor(self.feature_spec_)),
                ("estimator", clone(estimator)),
            ]
        )
        self.treatment_pipeline_ = Pipeline(
            [
                ("preprocessor", build_preprocessor(self.feature_spec_)),
                ("estimator", clone(estimator)),
            ]
        )
        for group, pipeline in (
            (0, self.control_pipeline_),
            (1, self.treatment_pipeline_),
        ):
            subset = frame.filter(pl.col(treatment_column) == group)
            target = subset[target_column].to_numpy()
            require_both_outcomes(target, f"Группа treatment={group} T-Learner")
            pipeline.fit(prepare_feature_matrix(subset, self.feature_spec_), target)
        return self

    def predict_potential_outcomes(self, frame: pl.DataFrame) -> pl.DataFrame:
        """Получить вероятности двух потенциальных outcomes и uplift."""
        if not hasattr(self, "control_pipeline_"):
            raise RuntimeError("Сначала обучите T-Learner методом fit")
        features = prepare_feature_matrix(frame, self.feature_spec_)
        probability_control = np.asarray(
            self.control_pipeline_.predict_proba(features)
        )[:, 1]
        probability_treatment = np.asarray(
            self.treatment_pipeline_.predict_proba(features)
        )[:, 1]
        return pl.DataFrame(
            {
                "probability_control": probability_control,
                "probability_treatment": probability_treatment,
                "predicted_uplift": probability_treatment - probability_control,
            }
        )
