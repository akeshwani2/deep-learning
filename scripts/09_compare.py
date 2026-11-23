"""Compare the three models on the same held-out people -> artifacts/comparison/.

Reads each model's person-level predictions (scripts 05, 07, 08) and the two background-only
controls (scripts 06, 08). Writes one metrics table, paired AUROC differences with bootstrap
intervals, calibration measures, and ROC and reliability figures.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve

from parkinsons_voice.evaluation.compare import (
    brier_score,
    expected_calibration_error,
    paired_auroc_difference,
    reliability_table,
)
from parkinsons_voice.evaluation.participant import (
    THRESHOLD,
    bootstrap_intervals,
    participant_metrics,
)

ARTIFACTS = Path("artifacts")
OUT_DIR = ARTIFACTS / "comparison"
MODELS = {
    "Acoustic features + logistic regression": "acoustic_baseline",
    "Log-Mel CNN": "cnn",
    "Frozen wav2vec 2.0 + logistic regression": "wav2vec",
}
BACKGROUND_WAV2VEC = ARTIFACTS / "wav2vec/background_only/participant_predictions.csv"
ROBUSTNESS = ARTIFACTS / "acoustic_baseline/robustness.json"


def load(directory: str) -> pd.DataFrame:
    return pd.read_csv(ARTIFACTS / directory / "participant_predictions.csv")


people = {name: load(directory) for name, directory in MODELS.items()}
ids = [set(table["person_id"]) for table in people.values()]
if any(group != ids[0] for group in ids):
    raise SystemExit("the models were not evaluated on the same people")

OUT_DIR.mkdir(parents=True, exist_ok=True)
results = {}
for name, table in people.items():
    predicted = (table["score"] >= THRESHOLD).astype(int)
    tn, fp, fn, tp = confusion_matrix(table["label"], predicted).ravel()
    results[name] = {
        "participants": len(table),
        "pd": int(table["label"].sum()),
        "clips": int(table["recording_count"].sum()),
        "metrics": participant_metrics(table),
        "ci95": bootstrap_intervals(table),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "brier": brier_score(table),
        "ece": expected_calibration_error(table),
    }

names = list(people)
differences = {
    f"{first} minus {second}": paired_auroc_difference(people[first], people[second])
    for i, first in enumerate(names)
    for second in names[i + 1 :]
}

background = pd.read_csv(BACKGROUND_WAV2VEC)
controls = {
    "Background only, wav2vec 2.0 + logistic regression": {
        "participants": len(background),
        "metrics": participant_metrics(background),
        "ci95": bootstrap_intervals(background),
    },
    "Background only, MFCC + logistic regression": {
        key: json.loads(ROBUSTNESS.read_text())["checks"]["background_noise_only"][key]
        for key in ("participants", "metrics", "ci95")
    },
}
# The voice models on the people who have a background-only score, for a like-for-like read.
covered = set(background["person_id"])
same_people = {
    name: participant_metrics(table[table["person_id"].isin(covered)])["auroc"]
    for name, table in people.items()
}

report = {
    "threshold": THRESHOLD,
    "models": results,
    "paired_auroc_differences": differences,
    "background_only_controls": controls,
    "voice_model_auroc_on_background_covered_people": same_people,
}
(OUT_DIR / "comparison.json").write_text(json.dumps(report, indent=2))


def interval(entry: dict, metric: str) -> str:
    low, high = entry["ci95"][metric]
    return f"{entry['metrics'][metric]:.3f} ({low:.3f}-{high:.3f})"


lines = [
    "| Model | People | AUROC (95% CI) | Balanced accuracy | Sensitivity | Specificity "
    "| Precision | F1 | TN/FP/FN/TP | Brier | ECE |",
    "|---|---|---|---|---|---|---|---|---|---|---|",
]
for name, entry in results.items():
    m, c = entry["metrics"], entry["confusion"]
    lines.append(
        f"| {name} | {entry['participants']} | {interval(entry, 'auroc')} "
        f"| {interval(entry, 'balanced_accuracy')} | {m['sensitivity']:.2f} "
        f"| {m['specificity']:.2f} | {m['precision']:.2f} | {m['f1']:.2f} "
        f"| {c['tn']}/{c['fp']}/{c['fn']}/{c['tp']} | {entry['brier']:.3f} | {entry['ece']:.3f} |"
    )
for name, entry in controls.items():
    lines.append(
        f"| {name} | {entry['participants']} | {interval(entry, 'auroc')} "
        f"| {interval(entry, 'balanced_accuracy')} | | | | | | | |"
    )
lines += ["", "| Paired AUROC difference | Estimate (95% CI) |", "|---|---|"]
for name, d in differences.items():
    lines.append(
        f"| {name} | {d['difference']:+.3f} ({d['ci95_low']:+.3f} to {d['ci95_high']:+.3f}) |"
    )
(OUT_DIR / "comparison.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
print("voice-model AUROC on the people with a background score:", same_people)

figure, (roc_axis, reliability_axis) = plt.subplots(1, 2, figsize=(10, 4.5))
for name, table in people.items():
    fpr, tpr, _ = roc_curve(table["label"], table["score"])
    roc_axis.plot(fpr, tpr, label=f"{name} ({results[name]['metrics']['auroc']:.2f})")
    bins = reliability_table(table)
    reliability_axis.plot(bins["mean_score"], bins["pd_fraction"], marker="o", label=name)
fpr, tpr, _ = roc_curve(background["label"], background["score"])
auroc = controls["Background only, wav2vec 2.0 + logistic regression"]["metrics"]["auroc"]
roc_axis.plot(fpr, tpr, "k--", label=f"Background only, wav2vec 2.0 ({auroc:.2f})")
roc_axis.plot([0, 1], [0, 1], color="0.8", linewidth=1)
roc_axis.set(xlabel="False positive rate", ylabel="True positive rate")
roc_axis.set_title("Person-level ROC, held-out folds (AUROC)")
roc_axis.legend(fontsize=7, loc="lower right")
reliability_axis.plot([0, 1], [0, 1], color="0.8", linewidth=1)
reliability_axis.set(xlabel="Mean model score in bin", ylabel="Fraction with PD in bin")
reliability_axis.set_title("Reliability (5 bins, exploratory)")
figure.tight_layout()
figure.savefig(OUT_DIR / "roc_reliability.png", dpi=150)
print(f"-> {OUT_DIR}")
