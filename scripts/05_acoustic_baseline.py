"""Acoustic baseline: regularized logistic regression -> artifacts/acoustic_baseline/.

Each person is scored by a model trained on the other four folds; imputation, scaling, and
the regularization strength are fitted inside those training folds only. Clip scores are
averaged per person before any metric is computed.
"""

import json
from pathlib import Path

import pandas as pd

from parkinsons_voice.evaluation.participant import (
    THRESHOLD,
    aggregate_participants,
    bootstrap_intervals,
    participant_metrics,
)
from parkinsons_voice.features.acoustic import FEATURE_NAMES
from parkinsons_voice.models.logistic import C_GRID, nested_cv_predict

FEATURES = Path("data/derived/acoustic_features.csv")
OUT_DIR = Path("artifacts/acoustic_baseline")

clips = pd.read_csv(FEATURES)
scored, folds = nested_cv_predict(clips, list(FEATURE_NAMES))
people = aggregate_participants(scored)
metrics = participant_metrics(people)
intervals = bootstrap_intervals(people)

OUT_DIR.mkdir(parents=True, exist_ok=True)
scored[["clean_path", "person_id", "label", "fold", "score"]].to_csv(
    OUT_DIR / "clip_predictions.csv", index=False
)
people.to_csv(OUT_DIR / "participant_predictions.csv", index=False)
summary = {
    "model": "acoustic_logistic",
    "participants": len(people),
    "clips": len(scored),
    "threshold": THRESHOLD,
    "c_grid": C_GRID,
    "chosen_c_by_fold": {f.fold: f.chosen_c for f in folds},
    "metrics": metrics,
    "ci95": intervals,
}
(OUT_DIR / "metrics.json").write_text(json.dumps(summary, indent=2))

print(f"{len(people)} people, {len(scored)} clips -> {OUT_DIR}")
print("C chosen per fold:", summary["chosen_c_by_fold"])
for name, value in metrics.items():
    low, high = intervals[name]
    print(f"{name:18s} {value:.3f}  (95% CI {low:.3f}-{high:.3f})")
