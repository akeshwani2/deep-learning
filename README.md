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
