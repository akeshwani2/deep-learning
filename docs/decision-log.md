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
