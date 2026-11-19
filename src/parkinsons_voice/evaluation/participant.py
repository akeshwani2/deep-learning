"""Participant-level aggregation, metrics, and bootstrap intervals.

Models score recordings; every reported number is computed after averaging those scores
within a participant, so repeated recordings of one person count once.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

THRESHOLD = 0.5  # fixed before any held-out score was seen


def aggregate_participants(recordings: pd.DataFrame) -> pd.DataFrame:
    """Average ``score`` per ``person_id``; fail if a participant has two labels."""

    grouped = recordings.groupby("person_id")
    if (grouped["label"].nunique() != 1).any():
        raise ValueError("a participant has more than one label")
    return grouped.agg(
        label=("label", "first"),
        score=("score", "mean"),
        recording_count=("score", "size"),
    ).reset_index()


def participant_metrics(participants: pd.DataFrame, threshold: float = THRESHOLD) -> dict:
    """AUROC plus threshold metrics over one row per participant."""

    label = participants["label"].to_numpy()
    predicted = participants["score"].to_numpy() >= threshold
    sensitivity = float(predicted[label == 1].mean())
    specificity = float((~predicted[label == 0]).mean())
    return {
        "auroc": float(roc_auc_score(label, participants["score"])),
        "balanced_accuracy": (sensitivity + specificity) / 2,
        "sensitivity": sensitivity,
        "specificity": specificity,
    }


def bootstrap_intervals(
    participants: pd.DataFrame,
    *,
    replicates: int = 2000,
    seed: int = 0,
    threshold: float = THRESHOLD,
) -> dict[str, tuple[float, float]]:
    """95% percentile intervals from resampling participants within each class.

    Resampling within class keeps both classes in every replicate, so none is discarded.
    """

    rng = np.random.default_rng(seed)
    by_class = [group for _, group in participants.groupby("label")]
    draws = []
    for _ in range(replicates):
        sample = pd.concat(
            [group.iloc[rng.integers(0, len(group), len(group))] for group in by_class]
        )
        draws.append(participant_metrics(sample, threshold))
    table = pd.DataFrame(draws)
    return {
        name: (float(table[name].quantile(0.025)), float(table[name].quantile(0.975)))
        for name in table.columns
    }
