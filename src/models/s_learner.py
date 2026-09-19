"""S-Learner для оценки индивидуального условного uplift."""

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


class SLearner:
    """Одна модель outcome с treatment как дополнительным признаком."""

    treatment_feature = "__treatment__"

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
    ) -> SLearner:
        """Обучить outcome-модель на признаках и фактическом treatment."""
        if self.treatment_feature in feature_columns:
            raise ValueError(f"Имя {self.treatment_feature!r} зарезервировано")
        validate_uplift_frame(
            frame,
            feature_columns,
            treatment_column=treatment_column,
            target_column=target_column,
        )
        target = frame[target_column].to_numpy()
        require_both_outcomes(target, "Обучающая выборка S-Learner")
        self.feature_spec_ = infer_feature_spec(frame, feature_columns)
        features = prepare_feature_matrix(frame, self.feature_spec_)
        features[self.treatment_feature] = frame[treatment_column].to_numpy()
        estimator = self.estimator or LogisticRegression(
            max_iter=1_000,
            random_state=self.random_state,
        )
        self.pipeline_ = Pipeline(
            [
                (
                    "preprocessor",
                    build_preprocessor(
                        self.feature_spec_, extra_numeric=[self.treatment_feature]
                    ),
                ),
                ("estimator", clone(estimator)),
            ]
        )
        self.pipeline_.fit(features, target)
        return self

    def predict_potential_outcomes(self, frame: pl.DataFrame) -> pl.DataFrame:
        """Получить P(Y|T=0,X), P(Y|T=1,X) и их разность."""
        if not hasattr(self, "pipeline_"):
            raise RuntimeError("Сначала обучите S-Learner методом fit")
        features = prepare_feature_matrix(frame, self.feature_spec_)
        control = features.copy()
        treatment = features.copy()
        control[self.treatment_feature] = 0
        treatment[self.treatment_feature] = 1
        probability_control = np.asarray(self.pipeline_.predict_proba(control))[:, 1]
        probability_treatment = np.asarray(self.pipeline_.predict_proba(treatment))[
            :, 1
        ]
        return pl.DataFrame(
            {
                "probability_control": probability_control,
                "probability_treatment": probability_treatment,
                "predicted_uplift": probability_treatment - probability_control,
            }
        )
