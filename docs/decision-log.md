# Project Decision Log

This log records decisions that affect the study's scope, validity, privacy, or
reproducibility. It must not contain participant identifiers, credentials,
restricted source locators, or controlled data.

Project decision owners: **Poorav Rawat and Arhaan Keshwani (joint)**.

Future entries should record the date, status, owners, decision, rationale,
assumptions, consequences, and any follow-up work. A superseded decision remains
in the log and links to its replacement.

## DL-001: Project framing and candidate operating modes

- **Date:** 2026-10-22
- **Status:** Accepted for project framing; operating-mode selection remains pending
- **Owners:** Poorav Rawat and Arhaan Keshwani (joint)
- **Related work:** T0.1, T1.1, T1.2, T1.3; FR-DATA-001 through
  FR-DATA-006

### Decision

This project is a research comparison of Parkinson's-associated voice patterns.
It is **not** a diagnostic, screening, treatment, or clinical-validation product.
Outputs must use association and classification language, report uncertainty and
limitations, and must not be presented as medical advice or as evidence that a
person has or does not have Parkinson's disease.

Exactly one of these candidate modes will be selected before model development:

1. **Mode A — primary raw-audio study.** Use approved mPower sustained-vowel
   audio only if access, participant identifiers, usable target labels, permitted
   analysis, and decode feasibility satisfy the decision gate. Compare a
   regularized acoustic-feature model, a compact log-Mel CNN, and a frozen speech
   encoder with a regularized linear head on the same participant-disjoint cohort
   and split.
2. **Mode B — feature-only fallback.** If Mode A is unavailable or infeasible by
   the deadline established in T1.2, use UCI dataset 489. Compare regularized
   logistic regression with a small MLP using the released acoustic features and
   participant-grouped outer folds. Make no claims about raw audio, spectrogram
   CNNs, or pretrained speech representations.

This entry does not select a mode. The final choice will be recorded as a new
decision after T1.1 and T1.2 establish access status, permitted use, feasibility
criteria, and the decision deadline.

### Rationale

Keeping both paths explicit preserves a feasible study while preventing results
from different datasets or representations from being treated as a controlled
architecture comparison. Deferring the final mode avoids claiming data access or
permissions that have not yet been documented.

### Current assumptions

- The repository contains no evidence that mPower approval is complete, so its
  access status is treated as **unverified**, not denied or approved.
- Both candidate datasets contain a participant grouping key suitable for
  participant-disjoint evaluation; this must be verified during ingestion.
- No trained model, measured performance, or clinical validation currently
  exists.
- Dataset licenses, controlled-access conditions, institutional guidance, and
  participant protections take precedence over the planned analysis or demo.

### Consequences and follow-up

- All code, reports, and any optional demo must preserve the non-diagnostic scope.
- Controlled data, direct identifiers, restricted locators, credentials, and
  participant allocation artifacts must remain outside version control.
- T1.1 must record a non-sensitive mPower access-status update.
- T1.2 must define a dated decision deadline and measurable Mode A feasibility
  criteria.
- T1.3 must record the chosen mode, rationale, access status, data version or
  retrieval date, and permitted uses before model development begins.

## DL-002: Select the feature-only operating mode

- **Date:** 2026-10-22
- **Status:** Accepted; supersedes the pending mode selection in DL-001
- **Owners:** Poorav Rawat and Arhaan Keshwani (joint)
- **Selected mode:** Mode B — feature-only fallback
- **Related work:** T1.1, T1.2, T1.3; FR-DATA-001 through FR-DATA-006;
  FR-SPLIT-006; FR-MODEL-004
- **Governance record:** [Data access and governance status](data-access-status.md)
- **Data record:** [Shareable data manifest](data-manifest.md)

### Decision

Proceed with UCI dataset 489 in Mode B. Limit the required comparison to
regularized logistic regression and a small MLP over the same released acoustic
features and participant-grouped outer folds. Do not implement or report
raw-audio, spectrogram-CNN, or pretrained-speech-embedding comparisons as results
of this study.

### Rationale

The repository contains no evidence that the team has completed the individual
mPower access, certification/validation, intended-use, Conditions for Use, and
institutional requirements. The project cannot lawfully or reproducibly claim
that controlled data are available. With no external delivery date documented,
2026-10-22 is the conservative Mode A deadline. The official UCI catalog offers
the fallback dataset under CC BY 4.0, allowing the project to proceed on a
documented, lawful path without waiting for controlled access.

### Data version and permitted use

- **Source:** UCI Machine Learning Repository dataset 489
- **Persistent identifier:** <https://doi.org/10.24432/C5701F>
- **Metadata review date:** 2026-10-22
- **Source-data retrieval:** Official archive verification copy retrieved to
  temporary storage on 2026-10-22; archive and CSV checksums are recorded in the
  data manifest and must be reproduced for the durable T3B.1 source copy
- **License:** CC BY 4.0; use, sharing, and adaptation are permitted with
  attribution and the applicable license notices

### Consequences

- The Mode A branches (T3A, T4A, and Mode A model/evaluation work) are out of
  scope unless a later accepted entry explicitly supersedes this decision.
- Mode B must use participant-grouped outer cross-validation and fold-local model
  selection; repeated rows must never cross participant folds.
- Claims are limited to a comparison of feature-based models on this dataset.
- All non-diagnostic and responsible-communication constraints from DL-001 remain
  in force.

## DL-003: Switch to raw audio from the Italian PVS dataset

- **Date:** 2026-11-22 (recorded retroactively; the switch is described in the
  midpoint report)
- **Status:** Recorded; supersedes the scope limits of DL-002
- **Owners:** Poorav Rawat and Arhaan Keshwani (joint)
- **Related work:** T1.3, Phase 3A, Phase 4A, T5A.1 to T5A.4

### Decision

Run the three-model raw-audio comparison (acoustic features with logistic
regression, a compact log-Mel CNN, and a linear head on frozen wav2vec 2.0
embeddings) on the Italian Parkinson's Voice and Speech dataset (Dimauro and
Girardi 2019, CC BY 4.0). UCI 489 is retained only as the frozen M1 cohort and is
not used for any reported model result.

### Rationale

mPower access was not available in time and UCI 489 releases features without
audio, so it cannot support the CNN or the speech encoder. Italian PVS has raw
audio, participant identity in the file names, and a permissive license.

### Consequences

- The cohort is 47 people (22 elderly healthy, 25 PD), so there is no separate
  test partition. Evaluation is 5-fold participant-grouped cross-validation with
  all tuning inside each training fold.
- The healthy and PD groups were recorded under different conditions. DL-005
  records what that does to the claims.

## DL-004: Model settings and evaluation protocol

- **Date:** 2026-11-22
- **Status:** Recorded
- **Owners:** Poorav Rawat and Arhaan Keshwani (joint)
- **Related work:** T5.3, T6.1, design section 14 items 4 to 10

### Decision

- **Folds:** 5 outer participant folds from `scripts/02_folds.py` (seed 0).
- **Acoustic features:** Praat pitch (75 to 500 Hz), four jitter and five shimmer
  measures, harmonics-to-noise ratio; 13 MFCC means and standard deviations.
- **Linear models:** median imputer, standardizer, class-balanced logistic
  regression; C from {0.001, 0.01, 0.1, 1, 10} by participant AUROC over 4 inner
  participant folds, ties to the stronger penalty.
- **CNN:** 64-band log-Mel (25 ms window, 10 ms hop, 50 to 8000 Hz), per-clip
  standardization, three conv blocks (16, 32, 64 channels), global average
  pooling, dropout 0.3, AdamW (learning rate 0.001, weight decay 0.0001), batch
  32, at most 40 epochs, patience 8 on validation participant AUROC with one
  inner fold as validation. Training-only gain (+-6 dB) and noise (20 to 40 dB
  SNR). These values were fixed before the held-out scores were computed and
  were not changed afterwards.
- **Encoder:** `facebook/wav2vec2-base` at revision
  `0b5b8e868dd84f03fd87d01f9c4ff0f080fecfe8` (Apache-2.0), frozen, mean of the
  last hidden state over all frames.
- **Aggregation and threshold:** arithmetic mean of clip scores per participant;
  threshold 0.5, fixed in advance.
- **Uncertainty:** 2000 participant bootstrap replicates, resampled within class,
  percentile 95% intervals, seed 0. Model differences use one shared resample.
- **Calibration:** Brier score and 5-bin expected calibration error on held-out
  participant scores. No recalibration. Outputs are called model scores, not
  probabilities.

## DL-005: Recording-condition confound and demo decision

- **Date:** 2026-11-22
- **Status:** Proposed by the implementation; needs owner confirmation
- **Owners:** Poorav Rawat and Arhaan Keshwani (joint)
- **Related work:** T6.6, T7.3, T7.4, T8.1

### Decision

1. Report every model next to a background-only control: the same classifier
   given only the non-speech part of each raw recording.
2. Do not claim that any model detects Parkinson's-associated voice patterns in
   this dataset. The supported claim is that the groups are separable from these
   recordings and that separability from recording conditions alone is at least
   as large.
3. Do not ship the optional demo (no-go for T8.1).

### Rationale

Background-only AUROC is 0.973 (MFCC) and 0.957 (wav2vec 2.0) on the 42 people
with enough non-speech audio. No voice model exceeds that by a margin its
interval supports. A demo would present a score that mostly reflects the
recording setup as if it reflected a voice, which the project's non-diagnostic
rules do not allow.
