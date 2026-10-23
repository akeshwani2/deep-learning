# Shareable Data Manifest

This manifest records source and governance metadata only. It contains no source
rows, participant identifiers, restricted locators, or participant allocations.

## Active source: UCI dataset 489

| Field | Recorded value |
|---|---|
| Operating mode | Mode B — feature-only fallback |
| Dataset | Parkinson Dataset with replicated acoustic features |
| Repository identifier | UCI Machine Learning Repository dataset 489 |
| Persistent identifier | <https://doi.org/10.24432/C5701F> |
| Official source | <https://archive.ics.uci.edu/dataset/489/parkinson+dataset+with+replicated+acoustic+features> |
| Metadata review date | 2026-10-22 |
| Dataset version | No explicit version is exposed by the catalog; pinned by DOI and retrieval checksum |
| Data retrieval date | 2026-10-22 (verified durable local copy; no source data committed) |
| Access class | Public download |
| License | Creative Commons Attribution 4.0 International (CC BY 4.0) |
| License text | <https://creativecommons.org/licenses/by/4.0/> |
| Verified source file | `ReplicatedAcousticFeatures-ParkinsonDatabase.csv` |
| Verified contents | 240 feature rows, 80 participants, three replications per participant, 40 control and 40 PD participants |
| Archive SHA-256 | `d82f569bacd50b9f49f6d59e9155a3c318c08e50b137b09423ea5eeed2c92ef5` |
| CSV SHA-256 | `78e41352e0dcab340d5ab3f45ce6b40b805de4c6105ecbecce35b082ef800f91` |
| Local source location | `${PARKINSONS_VOICE_DATA_DIR}/raw/uci489/` (ignored; never commit) |
| Local derived location | `${PARKINSONS_VOICE_ARTIFACT_DIR}/` (ignored; never commit participant allocations) |

The archive was downloaded from the official UCI static-file endpoint. The
durable ignored source copy reproduced the recorded checksum. Aggregate
verification found 240 data rows, 48 columns, 80 unique participant IDs, exactly
three rows per participant, 40 participants with label 0, 40 with label 1, and no
within-participant label inconsistency. The official catalog defines `Status=0`
as healthy/control and `Status=1` as PD. Any later mismatch stops ingestion and
requires a manifest update; it must not be silently accepted.

The frozen cohort and allocation fingerprints are recorded in the
[M1 validation report](m1-cohort-report.md#frozen-fingerprints).

## Permitted use and redistribution policy

The official UCI catalog states that the dataset is licensed under CC BY 4.0,
which permits use, sharing, and adaptation with appropriate credit. Repository
code and aggregate results may be shared. If source data or an adaptation is ever
redistributed, it must preserve attribution, link the license, and identify
changes. The default project policy is to link to UCI rather than republish the
source table.

Required attribution:

> Prez, C. (2016). Parkinson Dataset with replicated acoustic features
> [Dataset]. UCI Machine Learning Repository.
> <https://doi.org/10.24432/C5701F>.

## Storage and release controls

- Source and derived row-level data live only below ignored local data/artifact
  roots configured through `.env`; machine-specific absolute paths are forbidden.
- Participant fold assignments, row-level predictions, caches, and serialized
  models remain ignored unless a later release audit explicitly establishes that
  an artifact is safe and necessary to publish.
- Credentials and tokens are never needed for the active UCI path and must never
  be committed if mPower access is later reconsidered.
- Shareable outputs are limited to configuration, schemas, code, aggregate
  counts, metrics, permitted models, and non-identifying figures.
- All reports must describe association/classification and must not present the
  model as diagnosis, screening, treatment guidance, or clinical validation.
