# Parkinson's-Associated Voice Patterns

This repository implements a participant-level research comparison of
Parkinson's-associated voice features. It is not a diagnostic or screening
product.

The active operating mode is the feature-only fallback using UCI dataset 489.
The mode decision and its scope are recorded in
[`docs/decision-log.md`](docs/decision-log.md).

## Setup

```bash
UV_CACHE_DIR=/tmp/parkinsons-voice-uv-cache uv sync --frozen --all-groups
```

## Freeze and verify the Mode B cohort

Place the verified UCI CSV at `data/raw/uci489/` or pass an explicit path:

```bash
uv run parkinsons-voice freeze-cohort \
  --source data/raw/uci489/ReplicatedAcousticFeatures-ParkinsonDatabase.csv \
  --output artifacts/m1-cohort
uv run parkinsons-voice verify-cohort --artifact-dir artifacts/m1-cohort
```

Source data, row-level manifests, and participant allocations are ignored by
Git. Only aggregate, non-identifying evidence belongs in tracked documentation.

## Checks

```bash
scripts/check.sh
```

## Italian PVS raw-audio pipeline

Raw sustained-vowel audio, as an alternative to the UCI 489 feature-only data. Run from the repo root, in order:

```sh
uv run python scripts/00_download.py      # Kaggle download + fixes -> data/raw/italian_pvs/
uv run python scripts/01_manifest.py      # one row per recording, with person ID -> data/derived/manifest.csv
uv run python scripts/02_folds.py         # 5 person-level CV folds -> data/derived/folds.csv
uv run python scripts/03_clean_vowels.py  # 3 s, 16 kHz vowel clips -> data/derived/vowels_16k/, vowels.csv
```

`data/` is git-ignored.

### Dataset notes

Italian Parkinson's Voice and Speech (Dimauro & Girardi 2019, CC BY 4.0), via [Kaggle](https://www.kaggle.com/datasets/ieeeai/italian-parkinsons-voice-and-speech).

- **47 people:** 22 elderly healthy, 25 PD. The 15 young healthy controls are left out, so the model can't separate the groups by age.
- **The PD group has 28 folders but 25 people.** Vito S, Nicola S and Roberto R were each recorded twice (2016 and 2017). Person IDs come from the code in each file name, not the folder.
- **Sample rate differs by group.** All 2016 PD recordings (10 people) are 44.1 kHz; everything else is 16 kHz. Everything is resampled to 16 kHz, but results should also be checked on the 16 kHz recordings only.
- **Vowels:** a/e/i/o/u, each said twice per session. One clip, under 3 s after trimming silence, is dropped, leaving 494.
