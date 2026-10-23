"""Assign each person to one of 5 cross-validation folds -> data/derived/folds.csv.

Folds are made over people, not recordings, so no one is in both training and testing.
Each fold keeps roughly the same healthy/PD balance.
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold

MANIFEST = Path("data/derived/manifest.csv")
OUT = Path("data/derived/folds.csv")
SEED = 0

people = pd.read_csv(MANIFEST).groupby("person_id", as_index=False)["label"].first()
people["fold"] = -1
for fold, (_, test_idx) in enumerate(StratifiedKFold(5, shuffle=True, random_state=SEED).split(people, people.label)):
    people.loc[test_idx, "fold"] = fold
assert (people.fold >= 0).all()

people.to_csv(OUT, index=False)
print(people.groupby(["fold", "label"]).size().unstack().rename(columns={0: "healthy", 1: "pd"}))
