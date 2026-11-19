"""Extract acoustic features for every cleaned vowel clip -> data/derived/acoustic_features.csv.

One row per clip: pitch, jitter, shimmer, HNR, and MFCC summaries, plus n_missing (how many
of them could not be measured). Each clip is processed on its own, so nothing leaks across folds.
"""

from pathlib import Path

import pandas as pd
import soundfile as sf

from parkinsons_voice.features.acoustic import FEATURE_NAMES, extract_acoustic_features

VOWELS = Path("data/derived/vowels.csv")
OUT = Path("data/derived/acoustic_features.csv")
KEYS = ["clean_path", "person_id", "label", "fold", "task", "rep", "sex", "source_sr"]

clips = pd.read_csv(VOWELS)
rows = []
for clip in clips.itertuples():
    samples, sample_rate = sf.read(clip.clean_path)
    rows.append(extract_acoustic_features(samples, sample_rate))

out = pd.concat([clips[KEYS], pd.DataFrame(rows)], axis=1)
out.to_csv(OUT, index=False)
print(f"{len(out)} clips x {len(FEATURE_NAMES)} features -> {OUT}")
print(f"clips with a missing feature: {(out.n_missing > 0).sum()}")
print(out[list(FEATURE_NAMES)].isna().sum().loc[lambda s: s > 0].rename("missing"))
