"""Build data/derived/manifest.csv: one row per recording, with a person ID and label.

The young healthy group is skipped: they are much younger than the PD group, so a model
could learn age instead of disease. Their file names also don't identify the speaker.
"""

import re
from pathlib import Path

import pandas as pd
import soundfile as sf

RAW = Path("data/raw/italian_pvs/italian_parkinson")
OUT = Path("data/derived/manifest.csv")
GROUPS = {"22 Elderly Healthy Control": 0, "28 People with Parkinson's disease": 1}

# e.g. VA1 NMIICNOO 52 M 100220171136 .wav -> task, rep, scrambled initials, birth year, sex, date+time
NAME = re.compile(r"^(VA|VE|VI|VO|VU|B|D|FB|PR)(\d)([A-Za-z_]{7,8})(\d{2})([MF])(\d+)\.wav$")

rows = []
for group, label in GROUPS.items():
    for path in sorted((RAW / group).rglob("*.wav")):
        m = NAME.match(path.name)
        assert m, f"unparsed file name: {path}"
        task, rep, initials, birth_year, sex, _ = m.groups()
        # One of Antonia G's files says born '63; her other 15 files say '52.
        if path.name == "VI2AGNUTGOL63F100220171052.wav":
            birth_year = "52"
        # The same person can appear in two folders with the initials scrambled in a
        # different order (and one file drops the "_"), so the ID uses the sorted letters.
        person_id = "".join(sorted(initials.lower().replace("_", ""))) + birth_year + sex
        info = sf.info(path)
        rows.append({
            "path": str(path),
            "person_id": person_id,
            "label": label,
            "task": task,
            "is_vowel": task.startswith("V"),
            "rep": int(rep),
            "sex": sex,
            "birth_year": int(birth_year),
            "source_sr": info.samplerate,
            "duration_s": round(info.duration, 2),
        })

df = pd.DataFrame(rows)
assert df.groupby("person_id")["label"].nunique().max() == 1, "a person has two labels"
people = df.groupby("label")["person_id"].nunique()
assert people[0] == 22 and people[1] == 25, f"unexpected person counts: {people.to_dict()}"

OUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT, index=False)
print(f"{len(df)} recordings, {df.person_id.nunique()} people -> {OUT}")
print(df.groupby(["label", "source_sr"]).person_id.nunique().rename("people"))
