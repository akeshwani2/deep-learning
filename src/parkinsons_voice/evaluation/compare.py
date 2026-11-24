"""Calibration measures and paired participant-bootstrap comparisons between models."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

CALIBRATION_BINS = 5


def brier_score(participants: pd.DataFrame) -> float:
    return float(((participants["score"] - participants["label"]) ** 2).mean())


def reliability_table(participants: pd.DataFrame, bins: int = CALIBRATION_BINS) -> pd.DataFrame:
    """Mean score, observed PD fraction, and count for each equal-width score bin in use."""

    edges = np.linspace(0, 1, bins + 1)
    index = np.clip(np.digitize(participants["score"], edges[1:-1]), 0, bins - 1)
    table = (
        participants.assign(bin=index)
        .groupby("bin")
        .agg(mean_score=("score", "mean"), pd_fraction=("label", "mean"), count=("label", "size"))
    )
    return table.reset_index()


def expected_calibration_error(participants: pd.DataFrame, bins: int = CALIBRATION_BINS) -> float:
    table = reliability_table(participants, bins)
    gap = (table["mean_score"] - table["pd_fraction"]).abs()
    return float((gap * table["count"]).sum() / table["count"].sum())


def paired_auroc_difference(
    first: pd.DataFrame, second: pd.DataFrame, *, replicates: int = 2000, seed: int = 0
) -> dict[str, float]:
    """AUROC(first) - AUROC(second) with a 95% interval from one shared participant resample.

    Both tables must cover the same participants. Each replicate draws participants within
    class once and applies that draw to both models, which keeps the comparison paired.
    """

    merged = first.merge(second, on=["person_id", "label"], suffixes=("_first", "_second"))
    if not len(merged) == len(first) == len(second):
        raise ValueError("the two models do not cover the same participants")

    def difference(table: pd.DataFrame) -> float:
        return roc_auc_score(table["label"], table["score_first"]) - roc_auc_score(
            table["label"], table["score_second"]
        )

    rng = np.random.default_rng(seed)
    by_class = [group for _, group in merged.groupby("label")]
    draws = [
        difference(
            pd.concat([group.iloc[rng.integers(0, len(group), len(group))] for group in by_class])
        )
        for _ in range(replicates)
    ]
    return {
        "difference": float(difference(merged)),
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
    }
