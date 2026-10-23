"""Clean the sustained-vowel recordings -> data/derived/vowels_16k/*.wav + data/derived/vowels.csv.

Per recording, with no statistics shared across recordings (so nothing can leak between folds):
1. load as mono 16 kHz. All PD files from 2016 are 44.1 kHz and every healthy file is 16 kHz,
   so resampling everything stops a model from telling the groups apart by sample rate;
2. trim silence at the start and end;
3. keep the middle 3 s, the steady part of the vowel (recordings shorter than 3 s are dropped);
4. scale so the loudest point is 0.9, since microphone distance and gain weren't controlled.
"""

from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf

MANIFEST = Path("data/derived/manifest.csv")
FOLDS = Path("data/derived/folds.csv")
OUT_DIR = Path("data/derived/vowels_16k")
OUT = Path("data/derived/vowels.csv")
SR = 16000
LENGTH = 3 * SR

df = pd.read_csv(MANIFEST)
folds = pd.read_csv(FOLDS)[["person_id", "fold"]]
df = df[df.is_vowel].merge(folds, on="person_id", validate="many_to_one")
OUT_DIR.mkdir(parents=True, exist_ok=True)

kept = []
for row in df.itertuples():
    y, _ = librosa.load(row.path, sr=SR, mono=True)
    y, _ = librosa.effects.trim(y, top_db=30)
    if len(y) < LENGTH:
        print(f"dropped (only {len(y) / SR:.2f}s of voice): {row.path}")
        continue
    start = (len(y) - LENGTH) // 2
    y = y[start : start + LENGTH]
    y = 0.9 * y / np.abs(y).max()
    clean_path = OUT_DIR / Path(row.path).name
    sf.write(clean_path, y, SR, subtype="FLOAT")
    kept.append({**row._asdict(), "clean_path": str(clean_path)})

out = pd.DataFrame(kept).drop(columns=["Index", "is_vowel"])
assert out.clean_path.is_unique
out.to_csv(OUT, index=False)
print(f"{len(out)} of {len(df)} vowel recordings kept, {out.person_id.nunique()} people -> {OUT}")
