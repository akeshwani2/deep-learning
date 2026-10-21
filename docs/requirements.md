# Project Requirements

## 1. Purpose

This document translates the project proposal, *Parkinson's-Associated Voice Patterns: A Participant-Level Comparison of Acoustic Features and Deep Audio Representations*, into verifiable requirements.

The project is a research comparison, not a diagnostic product. Its central question is:

> Can models distinguish Parkinson's disease (PD)-associated from control sustained-vowel recordings from unseen participants, and do log-Mel or frozen speech representations improve participant-level discrimination over acoustic features when trained and evaluated on the same cohort and split?

The comparison is conditional on obtaining controlled access to mPower raw audio and participant identifiers. If that access is unavailable in time, the project will use the UCI replicated-acoustic-features dataset and compare only models supported by that dataset.

## 2. Scope

### 2.1 In scope

- Constructing a reproducible, participant-level cohort from an approved data source.
- Training and evaluating the model families permitted by the selected data path.
- Preventing participant leakage across training, validation, and test partitions.
- Aggregating recording-level predictions into participant-level predictions.
- Quantifying uncertainty at the participant level.
- Recording data exclusions, preprocessing decisions, model settings, and runtime environment.
- Producing a research report and, if permissions and model quality allow, an educational demonstration.
- Communicating limitations, uncertainty, data provenance, and non-diagnostic status.

### 2.2 Out of scope

- Diagnosing, screening for, or recommending treatment for PD.
- Claiming clinical validity, generalization to new populations, or causal disease biomarkers.
- Publishing or redistributing controlled mPower recordings or identifiable participant data.
- Treating repeated recordings as independent participants.
- Comparing raw-audio architectures on a feature-only dataset.
- Interpreting results from different datasets as if they were controlled architecture comparisons.
- Training a large speech encoder from scratch.
- End-to-end fine-tuning of a pretrained speech encoder in the required experiment set.

## 3. Operating modes and decision gate

Exactly one mode must be declared in the experiment manifest before model development proceeds.

| Mode | Entry condition | Required comparison | Unsupported claims |
|---|---|---|---|
| **A: Primary raw-audio study** | Approved access to mPower audio, usable participant IDs, and usable PD-status labels | Regularized acoustic-feature model vs. compact log-Mel CNN vs. frozen speech encoder with a regularized linear head | Clinical diagnosis or validation; generalization beyond the constructed cohort |
| **B: Feature-only fallback** | Primary access is unavailable or insufficient; UCI dataset 489 can be obtained and verified | Regularized logistic regression vs. small MLP using the released acoustic features | Any conclusion about spectrograms, raw audio, or pretrained audio embeddings |

The decision record must include the selected mode, date, owner, data-access status, dataset version or retrieval date, and reason for selecting the mode.

## 4. Stakeholders and users

- **Research team:** builds the cohort, models, evaluation, and report.
- **Course reviewers/instructors:** assess scientific validity, reproducibility, and alignment with the proposal.
- **Demo users:** may inspect a permitted example or submit an audio sample if uploads are implemented; they must not be led to treat the result as medical advice.
- **Data custodians and participants:** their consent, use restrictions, privacy, and redistribution constraints govern the project.

## 5. Functional requirements

Requirement keywords **MUST**, **SHOULD**, and **MAY** indicate required, recommended, and optional behavior.

### 5.1 Data acquisition and governance

- **FR-DATA-001:** The pipeline MUST use only data obtained under applicable licenses, data-use agreements, and institutional guidance.
- **FR-DATA-002:** For mPower, the team MUST document completion or status of Synapse verification, certification, intended-use approval, and agreement to the conditions of use.
- **FR-DATA-003:** Controlled data, credentials, tokens, participant identifiers, and unredacted intermediate files MUST NOT be committed to version control.
- **FR-DATA-004:** The repository MUST contain a data manifest recording source, version or retrieval date, license/access class, storage convention, and redistribution restrictions without exposing sensitive values.
- **FR-DATA-005:** Each usable sample MUST be linkable internally to a participant identifier, source recording identifier, label provenance, and exclusion status.
- **FR-DATA-006:** Released artifacts MUST contain only data and examples allowed by the selected dataset's terms.

### 5.2 Cohort construction

- **FR-COHORT-001:** The primary label MUST represent the dataset's participant-reported professional-diagnosis status when using mPower, not an independently adjudicated diagnosis.
- **FR-COHORT-002:** Participants with missing, uncertain, or unusable target status MUST be excluded rather than assigned to the control class.
- **FR-COHORT-003:** The raw-audio cohort MUST retain only sustained-vowel samples with a usable participant ID and sufficient voiced content.
- **FR-COHORT-004:** Countdown audio and other known non-task segments MUST be excluded from analyzed audio.
- **FR-COHORT-005:** Every exclusion MUST have a reason code; aggregate counts by reason MUST be included in the report.
- **FR-COHORT-006:** The same eligible participant and recording set MUST feed all models in a controlled comparison, except for documented model-specific extraction failures.
- **FR-COHORT-007:** For the raw-audio study, a training-only cap on recordings per participant MUST be selected and documented to reduce dominance by frequent contributors.
- **FR-COHORT-008:** The cohort summary MUST report participant and recording counts by class, recording-duration statistics, and permitted demographic/device coverage. It MUST not imply an exact sample size before cohort construction.

### 5.3 Partitioning and leakage prevention

- **FR-SPLIT-001:** All recordings, segments, augmentations, features, and embeddings derived from one participant MUST remain in one partition or cross-validation fold.
- **FR-SPLIT-002:** In Mode A, the default partition SHOULD be an approximately 70%/15%/15% class-stratified split by participant for training, validation, and held-out testing.
- **FR-SPLIT-003:** The split seed, algorithm, library version, input cohort fingerprint, and participant allocation artifact MUST be recorded, subject to sharing restrictions.
- **FR-SPLIT-004:** Preprocessing parameters, imputers, scalers, feature selectors, calibration models, and hyperparameters MUST be fitted or selected without access to held-out test outcomes.
- **FR-SPLIT-005:** The Mode A test partition MUST remain sealed until model families, selection procedure, aggregation rule, and threshold rule are frozen.
- **FR-SPLIT-006:** Mode B MUST use stratified grouped outer cross-validation by participant, with preprocessing and model selection performed within each outer training fold.
- **FR-SPLIT-007:** Automated checks MUST fail when a participant appears in more than one partition or a row lacks a participant group.

### 5.4 Signal and feature processing

- **FR-PREP-001:** Mode A recordings MUST be decoded consistently and resampled to the rate required by each downstream model.
- **FR-PREP-002:** The pipeline MUST identify usable voiced content and apply a documented fixed-duration crop/pad policy that does not use test-set statistics.
- **FR-PREP-003:** Acoustic inputs MUST include, where extraction is reliable, pitch, jitter, shimmer, harmonic-to-noise ratio, and MFCC summary statistics.
- **FR-PREP-004:** Feature-extraction failures MUST be flagged. Missing values MUST use imputation fitted only on training data.
- **FR-PREP-005:** CNN inputs MUST be log-Mel spectrograms generated with fixed and recorded parameters.
- **FR-PREP-006:** Frozen-encoder inputs MUST use the same source audio preprocessing as other Mode A models, adjusted only for the encoder's required sample rate and normalization.
- **FR-PREP-007:** Frozen embeddings MUST identify the exact published checkpoint and pooling rule.
- **FR-PREP-008:** Training-time augmentation MAY include small gain changes and mild background noise. It MUST be disabled for validation and test data.
- **FR-PREP-009:** Pitch shifting and time stretching MUST NOT be part of the primary experiment; they MAY appear as clearly labeled sensitivity analyses.
- **FR-PREP-010:** Processed artifacts MUST be traceable to raw inputs and preprocessing configuration.

### 5.5 Models and training

- **FR-MODEL-001:** The Mode A acoustic baseline MUST include regularized logistic regression over acoustic features. A small MLP MAY be added as a secondary baseline.
- **FR-MODEL-002:** The Mode A CNN MUST be compact, use fixed-duration log-Mel inputs, contain two or three convolutional blocks, use global pooling, and emit a binary score.
- **FR-MODEL-003:** The Mode A representation model MUST keep the speech encoder frozen and train a regularized linear classification head over temporally pooled embeddings.
- **FR-MODEL-004:** Mode B MUST compare regularized logistic regression with a small MLP over the same released feature columns and participant folds.
- **FR-MODEL-005:** Neural models MUST use binary cross-entropy or a documented equivalent binary classification objective.
- **FR-MODEL-006:** Neural model selection MUST use validation participant-level AUROC, with a documented early-stopping policy.
- **FR-MODEL-007:** Linear-model regularization MUST be selected using validation data or inner grouped cross-validation, depending on mode.
- **FR-MODEL-008:** Class weighting, sampling, or other imbalance handling MUST be selected using training data only and explicitly reported.
- **FR-MODEL-009:** Every run MUST record model configuration, seed, software/checkpoint versions, optimizer, batch size, epoch count, early-stopping result, hardware type, and elapsed time where applicable.
- **FR-MODEL-010:** Training MUST save enough state to reproduce the selected model and identify its data, split, and configuration.

### 5.6 Evaluation and statistical analysis

- **FR-EVAL-001:** Each recording MUST receive a model score; the primary participant score MUST be the arithmetic mean of that participant's recording scores.
- **FR-EVAL-002:** The primary metric MUST be participant-level AUROC.
- **FR-EVAL-003:** Secondary metrics MUST include participant-level precision, recall, F1, balanced accuracy, and a confusion matrix.
- **FR-EVAL-004:** Threshold-dependent test metrics MUST use a threshold prespecified before test access or selected using validation data only.
- **FR-EVAL-005:** The report MUST state class prevalence and denominators for every primary evaluation set.
- **FR-EVAL-006:** Mode A metric confidence intervals MUST use bootstrap resampling of participants, not recordings.
- **FR-EVAL-007:** Mode A model-difference confidence intervals MUST use paired participant bootstrap samples across models.
- **FR-EVAL-008:** Mode B MUST combine held-out predictions from grouped outer folds before participant-level metrics are computed; no training-fold prediction may enter final estimates.
- **FR-EVAL-009:** A score MUST be called a probability only after a documented validation-set calibration assessment. Otherwise, user-facing output MUST call it a model score.
- **FR-EVAL-010:** Subgroup analyses MUST be conditional on sufficient metadata/sample size and MUST report subgroup denominators and uncertainty.
- **FR-EVAL-011:** PCA or UMAP plots MUST be labeled exploratory and MUST NOT be presented as evidence of clinically distinct clusters.
- **FR-EVAL-012:** Cross-dataset numerical differences MUST NOT be described as architecture effects.

### 5.7 Reporting and reproducibility

- **FR-REPORT-001:** The final report MUST distinguish planned methods from completed methods and measured findings.
- **FR-REPORT-002:** The report MUST identify which operating mode ran and why.
- **FR-REPORT-003:** Results MUST include participant-level estimates, uncertainty, and the number of participants and recordings contributing to each estimate.
- **FR-REPORT-004:** The report MUST include a cohort flow/exclusion summary and limitations concerning self-reported labels, confounding, repeated measures, and external validity.
- **FR-REPORT-005:** A clean environment MUST reproduce cohort derivation and experiment outputs from documented commands, assuming authorized access to the same data.
- **FR-REPORT-006:** Tables and figures MUST be generated from saved result artifacts rather than manually transcribed values.
- **FR-REPORT-007:** Failed or inconclusive experiments MUST remain in the experiment log when they influenced decisions.

### 5.8 Educational demonstration

- **FR-DEMO-001:** The demo is conditional on data-use permission, adequate model behavior, and schedule; it is not required to establish the scientific result.
- **FR-DEMO-002:** If implemented, it MUST show a waveform and spectrogram for a permitted sample and display the selected model score.
- **FR-DEMO-003:** It MUST prominently state that the model is experimental and cannot diagnose PD.
- **FR-DEMO-004:** User-supplied audio MUST not be retained by default. Temporary files MUST be deleted after inference or session expiry.
- **FR-DEMO-005:** The public demo MUST NOT expose controlled recordings, participant identifiers, or identifiable examples.
- **FR-DEMO-006:** Demo preprocessing MUST match the exported model's inference contract and return understandable errors for invalid, empty, unsupported, or insufficiently voiced audio.
- **FR-DEMO-007:** The demo MUST avoid treatment guidance, diagnostic labels, or language that could create false reassurance or anxiety.

## 6. Non-functional requirements

- **NFR-001 — Reproducibility:** Seeded operations, environment versions, immutable split artifacts, and configuration files MUST make repeated runs materially consistent.
- **NFR-002 — Traceability:** Each reported metric MUST be traceable to a run, model artifact, split, cohort manifest, and code revision.
- **NFR-003 — Privacy:** Sensitive data MUST remain in approved storage and MUST appear in logs only through non-identifying keys.
- **NFR-004 — Security:** Secrets MUST come from environment or approved secret storage, never committed source files or notebooks.
- **NFR-005 — Reliability:** Validation MUST stop the pipeline on leakage, duplicate sample keys, unknown labels, corrupt split files, or incompatible schemas.
- **NFR-006 — Portability:** Core experiments SHOULD run from scripts or parameterized notebooks in a documented Python environment without machine-specific paths.
- **NFR-007 — Efficiency:** Deterministic features and embeddings SHOULD be cached with provenance; reruns SHOULD avoid unchanged computation.
- **NFR-008 — Auditability:** Logs SHOULD capture exclusions, warnings, seeds, configuration, and software versions without restricted data.
- **NFR-009 — Accessibility:** Charts SHOULD use readable labels and colors; the demo SHOULD support keyboard navigation and text alternatives where practical.
- **NFR-010 — Responsible communication:** Deliverables MUST use association/classification language and avoid implying a medical test.

## 7. Acceptance criteria

The project is ready for final reporting when all criteria applicable to the selected mode pass:

1. A mode decision and data manifest exist.
2. The eligible cohort can be regenerated, and inclusion/exclusion counts reconcile.
3. Automated checks confirm that participant groups do not cross partitions or outer folds.
4. Compared models use the same eligible cohort and allocations, with any extraction loss itemized.
5. Training-only transformations are demonstrably fitted without test information.
6. Required model families have completed with recorded configurations and artifacts.
7. Participant prediction files contain one final score per participant per model and audit information.
8. Required metrics and uncertainty estimates are generated from those prediction files.
9. The report limits conclusions to the selected operating mode and includes responsible-use limitations.
10. The reproducibility procedure succeeds in a clean environment with authorized data access.
11. If a demo is released, privacy, failure handling, wording, and inference-parity checks pass.

## 8. Open decisions

These choices should be resolved during implementation and recorded; the proposal intentionally does not invent them:

- Whether mPower access and institutional approval arrive in time for Mode A.
- Exact minimum voiced duration, crop/pad length, and voice-activity procedure.
- Training recording cap per participant.
- Acoustic extraction library and exact feature definitions.
- Log-Mel parameters and compact CNN width/depth within the proposed design.
- Published pretrained speech checkpoint and temporal pooling rule.
- Hyperparameter ranges, optimizer settings, batch size, and patience.
- Bootstrap replication count and confidence-interval method.
- Whether metadata supports responsible subgroup analysis.
- Whether the educational demo can be released under the data-use terms.
