"""Download the Italian Parkinson's Voice and Speech dataset from Kaggle -> data/raw/italian_pvs/.

Source: https://www.kaggle.com/datasets/ieeeai/italian-parkinsons-voice-and-speech (CC BY 4.0),
mirrored from IEEE Dataport (Dimauro & Girardi 2019, DOI 10.21227/aw6b-tg17). No login needed.

Fixes applied while extracting:
- the zip holds a second identical copy of the data and a .cache folder; both are skipped;
- the "Nicolò C" folder name is garbled in the zip, so it is written as "Nicolo C";
- that folder also holds a copy of Nicola M's recordings (code NMIICNOO); those are skipped.
"""

import shutil
import urllib.request
import zipfile
from pathlib import Path

URL = "https://www.kaggle.com/api/v1/datasets/download/ieeeai/italian-parkinsons-voice-and-speech"
OUT = Path("data/raw/italian_pvs")
ZIP = Path("data/raw/italian_pvs.zip")

assert not OUT.exists(), f"{OUT} already exists; delete it to re-download"
ZIP.parent.mkdir(parents=True, exist_ok=True)
with urllib.request.urlopen(URL) as response, open(ZIP, "wb") as f:
    shutil.copyfileobj(response, f)

with zipfile.ZipFile(ZIP) as z:
    for info in z.infolist():
        parts = info.filename.split("/")
        if info.is_dir() or parts[0] != "italian_parkinson":
            continue
        if parts[1] == "Italian_Parkinson_s_Voice_and_speech":
            continue
        is_nicolo = len(parts) > 3 and parts[3].startswith("Nicol") and parts[3] != "Nicola M"
        if parts[2] == "17-28" and is_nicolo:
            if "NMIICNOO" in parts[-1]:
                continue
            parts[3] = "Nicolo C"
        out = OUT.joinpath(*parts)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(z.read(info))
ZIP.unlink()

wavs = list(OUT.rglob("*.wav"))
assert len(wavs) == 831, f"expected 831 wav files, got {len(wavs)}"
print(f"{len(wavs)} wav files -> {OUT}")
