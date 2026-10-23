# M1 Cohort Freeze and Validation Report

- **Status:** Pass
- **Frozen:** 2026-10-22
- **Operating mode:** Mode B — UCI feature-only fallback
- **Configuration:** `configs/data/uci489.json`
- **Fold algorithm:** `sha256_stratified_round_robin_v1`
- **Seed:** 2026

This report contains aggregate evidence only. The source table, source-row
locators, pseudonymous participant allocations, and row-level feature artifacts
remain in ignored local storage.

## Source verification

| Check | Result |
|---|---:|
| Official archive SHA-256 | `d82f569bacd50b9f49f6d59e9155a3c318c08e50b137b09423ea5eeed2c92ef5` |
| CSV SHA-256 | `78e41352e0dcab340d5ab3f45ce6b40b805de4c6105ecbecce35b082ef800f91` |
| Rows | 240 |
| Columns | 48 |
| Participants | 80 |
| Rows per participant | 3 for every participant |
| Control participants / rows | 40 / 120 |
| PD participants / rows | 40 / 120 |
| Duplicate sample keys | 0 |
| Within-participant label conflicts | 0 |
| Missing feature cells | 0 |
| Infinite or nonnumeric feature cells | 0 |

The durable local CSV reproduced the checksums and aggregate facts recorded in
the data manifest. The [official UCI documentation](https://archive.ics.uci.edu/dataset/489/parkinson+dataset+with+replicated+acoustic+features)
defines `Status` as 0 for healthy/control and 1 for PD; the adapter enforces only
those values. The freeze command rejects any different source checksum.

## Canonical manifest schema

| Field | Type/constraint | Release handling |
|---|---|---|
| `sample_key` | Unique opaque string | Shareable in controlled project artifacts |
| `participant_key` | Non-empty pseudonymous grouping key | Allocation remains ignored |
| `label` | Binary integer for eligible rows | Aggregate release only by default |
| `label_source` | Non-empty provenance string | Shareable |
| `source_kind` | `audio` or `feature_row` | Shareable |
| `source_locator` | Approved local/source-row locator | Removed from shareable manifest |
| `task_type` | Source-defined task | Shareable |
| `duration_seconds` | Nonnegative number or null | Null in Mode B |
| `voiced_seconds` | Nonnegative number or null | Null in Mode B |
| `eligible` | Boolean | Shareable |
| `exclusion_code` | Stable code for ineligible rows | Shareable as aggregates |
| `metadata_available` | JSON availability flags, not raw metadata | Shareable |
| `source_fingerprint` | Lowercase SHA-256 | Shareable |

Validation fails on duplicate sample keys, missing participant groups, unknown
labels/source kinds/exclusion codes, negative durations, malformed fingerprints,
or inconsistent participant labels. Source identifiers are deterministically
mapped to pseudonymous keys; this is a grouping safeguard, not a claim of
anonymity.

## UCI adapter and predictors

| Source field | Canonical use | Predictor? |
|---|---|---|
| `ID` | Pseudonymous `participant_key` | No |
| `ID` + `Recording` | Unique opaque `sample_key` | No |
| `Status` | Binary `label` | No |
| `Gender` | Metadata-availability flag only | No |
| Acoustic columns | Model feature table | Yes |

The frozen allow-list contains 44 acoustic predictors: four jitter measures,
five shimmer measures, five HNR measures, RPDE, DFA, PPE, GNE, MFCC0–MFCC12,
and Delta0–Delta12. `ID`, `Recording`, `Status`, and `Gender` are explicitly
forbidden as predictors. Missing predictors are allowed at ingestion only so a
later fold-local imputer can handle them without leakage; this cohort has none.

## Exclusion reconciliation

All 240 released rows are eligible. Each stable reason count is zero:

| Exclusion code | Count |
|---|---:|
| `MISSING_PARTICIPANT_ID` | 0 |
| `MISSING_OR_UNCERTAIN_LABEL` | 0 |
| `WRONG_TASK` | 0 |
| `DECODE_FAILURE` | 0 |
| `INSUFFICIENT_AUDIO` | 0 |
| `INSUFFICIENT_VOICED_CONTENT` | 0 |
| `DUPLICATE_SAMPLE` | 0 |
| `FEATURE_EXTRACTION_FAILURE` | 0 |

## Frozen participant folds

Five outer folds are persisted once and must be reused by both required models.
Every outer test fold contains 8 control and 8 PD participants, corresponding to
24 rows per class. Thus each outer training set contains 64 participants and 192
rows.

Within each outer training set, four inner folds are persisted. Every inner
validation fold contains 8 control and 8 PD participants. Automated validation
confirms complete coverage, class presence, pairwise participant disjointness,
outer-test exclusion from inner folds, and correct inheritance of participant
allocations by samples or derived segments.

## Frozen fingerprints

| Artifact | SHA-256 fingerprint |
|---|---|
| Source CSV | `78e41352e0dcab340d5ab3f45ce6b40b805de4c6105ecbecce35b082ef800f91` |
| Cohort configuration | `508213f4f59c6efe6eb48a240c58580c29b2f4d1edb1a1f03a58fb70d29d1917` |
| Cohort implementation | `e0d3b0e0578d8acf58487f216c9bff1f896884e85df1a8ba522955459af4b3b1` |
| Canonical manifest | `7e387e27c850e724579daad4989e7808cfcfea769f608758a1a377fbcea3406d` |
| Eligible feature table | `72119a15bc212587c081793952bebc528e32bf6ca3d4d6df000e8359236217a7` |
| Outer folds | `dd333fc735be72033b15820381f73ec825c2b3c7eadc22082f2bf2db5fae7f4e` |
| Inner folds | `32e0a6981226554973945a023af9b55afd18d922231f88a8eaf6aa1d76e029cc` |

Existing cached cohort artifacts are reusable only when all required
fingerprints match. Changed source data, configuration, cohort implementation,
manifests, features, or folds cause a fatal provenance error.

## Scope frozen with the cohort

The required comparison is regularized logistic regression versus a small MLP
using the same 44 predictors and the same outer folds. This project will make no
claim about raw-audio models, spectrogram CNNs, pretrained speech embeddings,
clinical diagnosis, screening, or clinical validation.

## Reproduction and test evidence

```bash
uv run --frozen parkinsons-voice freeze-cohort \
  --source data/raw/uci489/ReplicatedAcousticFeatures-ParkinsonDatabase.csv \
  --output artifacts/m1-cohort
uv run --frozen parkinsons-voice verify-cohort \
  --artifact-dir artifacts/m1-cohort
scripts/check.sh
```

The freeze and independent reload-verification commands passed under Python
3.12.13, NumPy 2.1.3, and pandas 2.2.3. The automated suite covers valid and
deliberately failing manifest, adapter, provenance, cache, overlap, inheritance,
and nested-fold cases.
