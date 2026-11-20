"""Robustness checks for the acoustic baseline -> artifacts/acoustic_baseline/robustness.json.

The PD and healthy recordings were not all made under the same conditions, so a high score
can come from the recording setup instead of the voice. Each check reruns the same nested CV
(or re-reads its held-out scores) on a restricted view of the data:

- originally-16 kHz recordings only, so no clip was downsampled from 44.1 kHz;
- each feature family on its own (Praat voice measures vs MFCC summaries);
- each sex, using the held-out scores of the full model;
- labels shuffled between participants, which must fall to chance;
- the non-speech parts of each raw recording only, which contain no voice, so any score
  above chance there comes from the room, microphone, or recording procedure.
"""

import json
from pathlib import Path

import librosa
import numpy as np
import pandas as pd

from parkinsons_voice.evaluation.participant import (
    aggregate_participants,
    bootstrap_intervals,
    participant_metrics,
)
from parkinsons_voice.features.acoustic import FEATURE_NAMES
from parkinsons_voice.models.logistic import nested_cv_predict

FEATURES = Path("data/derived/acoustic_features.csv")
OUT = Path("artifacts/acoustic_baseline/robustness.json")
NATIVE_SR = 16000
PERMUTATIONS = 20
VOWELS = Path("data/derived/vowels.csv")
SR = 16000
TOP_DB = 30  # same silence threshold as 03_clean_vowels.py
MARGIN = SR // 10  # also drop 0.1 s around each voiced stretch
MIN_BACKGROUND = SR // 4
N_MFCC = 13
BACKGROUND = [f"background_mfcc{i}" for i in range(N_MFCC)]

ALL = list(FEATURE_NAMES)
MFCC = [name for name in ALL if name.startswith("mfcc")]
PRAAT = [name for name in ALL if not name.startswith("mfcc")]


def summarize(scored: pd.DataFrame) -> dict:
    people = aggregate_participants(scored)
    return {
        "participants": len(people),
        "pd": int(people["label"].sum()),
        "healthy": int((people["label"] == 0).sum()),
        "clips": len(scored),
        "metrics": participant_metrics(people),
        "ci95": bootstrap_intervals(people),
    }


def shuffled_label_aurocs(clips: pd.DataFrame) -> list[float]:
    """AUROC after reassigning whole participants' labels at random."""

    rng = np.random.default_rng(0)
    labels = clips.groupby("person_id")["label"].first()
    aurocs = []
    for _ in range(PERMUTATIONS):
        shuffled = pd.Series(rng.permutation(labels.to_numpy()), index=labels.index)
        permuted = clips.assign(label=clips["person_id"].map(shuffled))
        scored, _ = nested_cv_predict(permuted, ALL)
        aurocs.append(participant_metrics(aggregate_participants(scored))["auroc"])
    return aurocs


def background_features(vowels: pd.DataFrame) -> pd.DataFrame:
    """MFCC means of each raw recording with every voiced stretch removed."""

    rows = []
    for row in vowels.itertuples():
        samples, _ = librosa.load(row.path, sr=SR)
        background = np.ones(len(samples), dtype=bool)
        for start, end in librosa.effects.split(samples, top_db=TOP_DB):
            background[max(0, start - MARGIN) : end + MARGIN] = False
        if background.sum() < MIN_BACKGROUND:
            continue
        quiet = samples[background] / np.abs(samples).max()
        mfcc = librosa.feature.mfcc(y=quiet, sr=SR, n_mfcc=N_MFCC).mean(axis=1)
        rows.append(
            {"person_id": row.person_id, "label": row.label, "fold": row.fold}
            | dict(zip(BACKGROUND, mfcc, strict=True))
        )
    return pd.DataFrame(rows)


clips = pd.read_csv(FEATURES)
full, _ = nested_cv_predict(clips, ALL)
native = clips[clips["source_sr"] == NATIVE_SR]

checks = {
    "all_recordings": summarize(full),
    "praat_features_only": summarize(nested_cv_predict(clips, PRAAT)[0]),
    "mfcc_features_only": summarize(nested_cv_predict(clips, MFCC)[0]),
    "native_16k_only": summarize(nested_cv_predict(native, ALL)[0]),
    "native_16k_praat_only": summarize(nested_cv_predict(native, PRAAT)[0]),
    "background_noise_only": summarize(
        nested_cv_predict(background_features(pd.read_csv(VOWELS)), BACKGROUND)[0]
    ),
}
for sex, group in full.groupby("sex"):
    checks[f"sex_{sex}"] = summarize(group)

shuffled = shuffled_label_aurocs(clips)
report = {
    "checks": checks,
    "shuffled_labels": {
        "permutations": PERMUTATIONS,
        "auroc_mean": float(np.mean(shuffled)),
        "auroc_max": float(np.max(shuffled)),
    },
    "clips_by_label_and_source_sr": {
        f"label{label}_sr{sr}": int(count)
        for (label, sr), count in clips.groupby(["label", "source_sr"]).size().items()
    },
}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(report, indent=2))

print(f"{'check':22s}   n  PD  HC  AUROC (95% CI)       bacc  sens  spec  prec  F1")
for name, check in checks.items():
    m, (low, high) = check["metrics"], check["ci95"]["auroc"]
    print(
        f"{name:22s} {check['participants']:3d} {check['pd']:3d} {check['healthy']:3d}  "
        f"{m['auroc']:.3f} ({low:.3f}-{high:.3f})  {m['balanced_accuracy']:.2f}  "
        f"{m['sensitivity']:.2f}  {m['specificity']:.2f}  {m['precision']:.2f}  {m['f1']:.2f}"
    )
print(
    f"shuffled labels x{PERMUTATIONS}: mean AUROC {report['shuffled_labels']['auroc_mean']:.3f}, "
    f"max {report['shuffled_labels']['auroc_max']:.3f}"
)
print("clips by label and original sample rate:", report["clips_by_label_and_source_sr"])
print(f"-> {OUT}")
