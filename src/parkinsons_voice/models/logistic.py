"""Regularized logistic regression evaluated with participant-grouped nested CV.

For each outer fold, the imputer, scaler, regularization strength, and classifier are fitted
using only the other folds' participants. The regularization strength is the one with the
best participant-level AUROC over inner folds that are also split by participant.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from parkinsons_voice.evaluation.participant import aggregate_participants, participant_metrics

C_GRID: tuple[float, ...] = (0.001, 0.01, 0.1, 1.0, 10.0)  # searched from strongest penalty
INNER_SPLITS = 4
SEED = 0


@dataclass(frozen=True)
class FoldResult:
    fold: int
    chosen_c: float
    inner_auroc: dict[float, float]
    pipeline: Pipeline


def build_pipeline(c: float) -> Pipeline:
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(C=c, class_weight="balanced", max_iter=5000),
    )


def _inner_auroc(train: pd.DataFrame, feature_names: list[str], c: float) -> float:
    """Participant AUROC of out-of-fold scores pooled over the inner folds."""

    splitter = StratifiedGroupKFold(INNER_SPLITS, shuffle=True, random_state=SEED)
    scores = np.empty(len(train))
    for fit_idx, val_idx in splitter.split(train, train["label"], train["person_id"]):
        fit, val = train.iloc[fit_idx], train.iloc[val_idx]
        model = build_pipeline(c).fit(fit[feature_names], fit["label"])
        scores[val_idx] = model.predict_proba(val[feature_names])[:, 1]
    participants = aggregate_participants(train.assign(score=scores))
    return participant_metrics(participants)["auroc"]


def nested_cv_predict(
    recordings: pd.DataFrame, feature_names: list[str]
) -> tuple[pd.DataFrame, list[FoldResult]]:
    """Score every recording with a model that never saw its participant.

    ``recordings`` needs ``person_id``, ``label``, ``fold``, and the feature columns. Returns the
    recordings with a ``score`` column (the model's PD-class output) and one result per fold.
    """

    if (recordings.groupby("person_id")["fold"].nunique() != 1).any():
        raise ValueError("a participant appears in more than one fold")

    scored = recordings.copy()
    scored["score"] = np.nan
    results = []
    for fold in sorted(recordings["fold"].unique()):
        held_out = recordings["fold"] == fold
        train = recordings[~held_out]
        inner = {c: _inner_auroc(train, feature_names, c) for c in C_GRID}
        chosen_c = max(C_GRID, key=lambda c: inner[c])  # ties go to the stronger penalty
        pipeline = build_pipeline(chosen_c).fit(train[feature_names], train["label"])
        scored.loc[held_out, "score"] = pipeline.predict_proba(
            recordings.loc[held_out, feature_names]
        )[:, 1]
        results.append(FoldResult(int(fold), chosen_c, inner, pipeline))
    return scored, results
