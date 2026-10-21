# System and Experiment Design

## 1. Design goals

The system is designed to answer one controlled research question while preventing the most consequential failure mode in repeated-recording health datasets: leakage between recordings from the same participant.

The design prioritizes:

1. participant-level separation and evaluation;
2. an apples-to-apples model comparison on one eligible cohort;
3. isolation of controlled or identifiable data;
4. reproducible, configuration-driven experiments;
5. explicit handling of raw-audio and feature-only study modes; and
6. cautious reporting suitable for research, not diagnosis.

## 2. High-level architecture

```text
Approved source data
        |
        v
  Ingestion + validation -----> immutable source manifest
        |
        v
  Cohort construction --------> exclusions + cohort manifest
        |
        v
 Participant split/folds -----> immutable allocation artifact
        |
        +-------------------------------+
        |                               |
        v                               v
 Mode A: raw audio                 Mode B: UCI features
        |                               |
        +--> common audio cleanup       +--> schema validation
        |       |                       |       |
        |       +--> acoustic features  |       +--> fold-local transforms
        |       +--> log-Mel tensors    |               |
        |       +--> frozen embeddings  |               +--> logistic regression
        |                               |               +--> small MLP
        +--> model training              |
                    |                    |
                    +---------+----------+
                              v
                  recording/row predictions
                              |
                              v
                   participant aggregation
                              |
                              v
                 metrics + paired uncertainty
                              |
                 +------------+------------+
                 v                         v
          generated report          optional safe demo
```

The mode is selected once in a recorded decision gate. Raw-audio branches never silently substitute feature-only data, and the two modes do not share numerical conclusions about architecture quality.

## 3. Proposed repository layout

The exact files may evolve, but responsibilities should remain separated:

```text
.
├── configs/
│   ├── data/
│   ├── model/
│   └── experiment/
├── docs/
│   ├── design.md
│   ├── requirements.md
│   └── tasks.md
├── notebooks/
│   └── exploration only; no unique production logic
├── src/
│   └── parkinsons_voice/
│       ├── data/          # ingestion, validation, cohort, splits
│       ├── audio/         # decoding, voiced-region and signal transforms
│       ├── features/      # acoustic, log-Mel and embedding extraction
│       ├── models/        # linear, MLP and CNN definitions
│       ├── training/      # fit/select/checkpoint workflows
│       ├── evaluation/    # aggregation, metrics, bootstrap, plots
│       └── demo/          # optional inference adapter and UI
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/          # synthetic/non-sensitive inputs only
├── artifacts/             # ignored; manifests, runs, models, results
├── data/                  # ignored; approved local source/derived data
└── reports/               # generated tables, plots and final write-up
```

The current `starter_notebook.ipynb` can remain an exploration entry point, but reusable preprocessing, training, and evaluation logic should move into importable modules and commands.

## 4. Configuration and provenance

Every pipeline command should consume an explicit configuration and create a run record. A run ID should join predictions, checkpoints, and metrics.

### 4.1 Experiment configuration

Suggested top-level fields:

```yaml
study_mode: raw_audio          # raw_audio | feature_only
seed: 2026
data:
  source: mpower
  manifest_version: <fingerprint-or-date>
  cohort_rules_version: v1
split:
  strategy: stratified_participant_holdout
  train_fraction: 0.70
  validation_fraction: 0.15
  test_fraction: 0.15
preprocessing:
  audio: <named-config>
  feature_set: <named-config>
model:
  family: acoustic_logistic
  parameters: {}
evaluation:
  aggregation: mean
  primary_metric: participant_auroc
  threshold_source: validation
```

The sample values are structural examples, not finalized hyperparameters.

### 4.2 Provenance chain

Each result should resolve through this chain:

```text
reported metric
  -> evaluation artifact
  -> participant prediction file
  -> selected model/checkpoint
  -> experiment configuration
  -> split/fold artifact
  -> cohort manifest
  -> source manifest and code revision
```

Manifests should store hashes or opaque internal keys rather than identifying values. Restricted participant allocation files stay in approved local storage; shareable summaries contain only counts and fingerprints.

## 5. Data design

### 5.1 Canonical sample manifest

Both study modes normalize source data to a canonical table. Mode-specific fields may be null.

| Field | Type | Purpose |
|---|---|---|
| `sample_key` | string | Non-identifying unique key for a recording or released feature row |
| `participant_key` | string | Stable, non-identifying grouping key |
| `label` | integer | Binary dataset target after eligibility validation |
| `label_source` | string | Provenance of the target label |
| `source_kind` | enum | `audio` or `feature_row` |
| `source_locator` | string | Approved local locator; excluded from shareable artifacts if sensitive |
| `task_type` | string | Sustained vowel or source-defined task |
| `duration_seconds` | float/null | Decoded duration for audio mode |
| `voiced_seconds` | float/null | Estimated usable voiced content |
| `eligible` | boolean | Cohort membership flag |
| `exclusion_code` | string/null | Normalized exclusion reason |
| `metadata_available` | object/flags | Permitted metadata availability, not raw values in logs |
| `source_fingerprint` | string | Detects changed source inputs |

Validation rejects duplicate `sample_key` values, missing participant groups, unknown targets, impossible durations, and locators outside configured data roots.

### 5.2 Exclusion codes

Use stable codes so cohort counts reconcile across runs:

- `MISSING_PARTICIPANT_ID`
- `MISSING_OR_UNCERTAIN_LABEL`
- `WRONG_TASK`
- `DECODE_FAILURE`
- `INSUFFICIENT_AUDIO`
- `INSUFFICIENT_VOICED_CONTENT`
- `DUPLICATE_SAMPLE`
- `FEATURE_EXTRACTION_FAILURE`

The report should separate cohort exclusions from model-specific extraction failures. A model-specific failure cannot silently remove the participant from one comparison; it must be reported and handled under a predefined common-comparison policy.

### 5.3 Data storage zones

- **Restricted source:** immutable downloaded files and source metadata; never committed.
- **Restricted derived:** cleaned audio, features, embeddings, and participant allocations; never committed unless explicitly permitted and de-identified.
- **Shareable artifacts:** configuration, schema, code, aggregate counts, permitted model artifacts, metrics, and non-sensitive figures.
- **Temporary inference:** uploaded demo audio, deleted immediately after inference or on short session expiry.

## 6. Cohort and partition design

### 6.1 Mode A: participant holdout

1. Validate source records and normalize labels.
2. Apply eligibility rules without looking at model outcomes.
3. Create one row per eligible recording in the canonical manifest.
4. Stratify participants by target and allocate approximately 70%/15%/15% to train/validation/test.
5. Persist the allocation before feature fitting.
6. Apply the training recording cap only within the training partition. Do not alter validation/test representation to improve metrics.
7. Seal test labels from model-development code where practical.

The allocator must assert that participant sets are pairwise disjoint and that every derived segment inherits its source participant's allocation.

### 6.2 Mode B: grouped nested evaluation

The UCI fallback contains repeated rows per participant. Use grouped, class-stratified outer cross-validation. Within each outer iteration:

1. reserve outer-test participants;
2. fit preprocessing and select hyperparameters using only outer-training participants, preferably with grouped inner folds;
3. refit the chosen pipeline on all outer-training participants;
4. predict each outer-test row once; and
5. aggregate rows to participants before final metrics.

Persist fold membership so logistic regression and MLP receive identical outer tests. The out-of-fold participant prediction table is the source of final Mode B results.

## 7. Mode A audio processing

### 7.1 Common preprocessing contract

All Mode A models begin from the same decoded and task-trimmed recording. A preprocessing result contains:

```text
waveform: float array
sample_rate: integer
valid_length: integer
voice_quality_flags: set
source_sample_key: string
preprocessing_version: string
```

Recommended sequence:

1. decode to mono floating-point audio;
2. remove the known countdown/task lead-in using verified metadata or a deterministic rule;
3. detect usable voiced intervals;
4. reject or flag insufficient voiced content;
5. select a fixed-duration region using a documented deterministic policy;
6. pad shorter valid samples using a fixed policy; and
7. resample from this common result for each representation.

Amplitude normalization should be conservative and identical across partitions. Statistics-based normalization must be fitted on training data.

### 7.2 Acoustic feature branch

Extract pitch, jitter, shimmer, harmonic-to-noise ratio, and MFCC summaries. The extractor emits values and quality flags. The tabular pipeline encapsulates:

```text
training-fitted imputer
  -> training-fitted scaler
  -> optional predeclared selector
  -> regularized classifier
```

This encapsulation prevents fitting transformations on validation or test samples.

### 7.3 Log-Mel CNN branch

Generate log-Mel tensors from the fixed-duration waveform using recorded sample rate, FFT size, hop length, window length, Mel-band count, and frequency range. Compute normalization parameters from training only or use a documented sample-local transform.

The proposed compact CNN is:

```text
log-Mel input
  -> Conv block 1
  -> Conv block 2
  -> optional Conv block 3
  -> global pooling
  -> dropout (if selected on validation)
  -> one binary logit
```

A convolution block should use a consistent, recorded ordering such as convolution, normalization, activation, and pooling. Global pooling avoids an oversized dense head.

### 7.4 Frozen encoder branch

Use a published wav2vec 2.0 checkpoint or another explicitly justified speech encoder. The encoder remains frozen for the required comparison.

```text
common processed waveform
  -> checkpoint-required resampling/normalization
  -> frozen encoder
  -> mask-aware temporal pooling
  -> optional training-fitted scaler
  -> regularized linear classifier
```

Cache embeddings with checkpoint identifier, library version, audio-preprocessing version, pooling rule, and source fingerprint. The cache loader verifies metadata before reuse.

## 8. Mode B feature processing

The fallback adapter maps the released UCI table to a participant key, binary target, documented predictors, and source-row key. Identifiers and targets must never become predictors. Each outer fold owns its imputer, scaler, and model.

```text
imputer -> scaler -> regularized logistic regression
imputer -> scaler -> small MLP
```

The MLP should be intentionally small relative to the 80-participant dataset and use early stopping inside each outer training/validation process.

## 9. Training and model selection

### 9.1 Unified trainer interface

Each model adapter supports conceptually equivalent operations:

```python
fit(train_data, validation_data, config) -> fitted_model
predict(fitted_model, samples) -> recording_or_row_scores
save(fitted_model, provenance) -> artifact_ref
load(artifact_ref) -> fitted_model
```

Score semantics must be declared. The evaluator, not individual trainers, owns participant aggregation and thresholding.

### 9.2 Reproducibility controls

- Seed Python, numerical, framework, data-loader, and split generators where supported.
- Record nondeterministic operations and hardware limits rather than claiming bitwise reproducibility.
- Save selected epoch and validation participant-level AUROC for neural models.
- Record every searched hyperparameter range before final test access.
- Never tune using test metrics, plots, or individual test errors.

### 9.3 Mode A test-release gate

Before final test evaluation, freeze and record:

- cohort and participant split;
- each representation version;
- each selected model/checkpoint;
- participant aggregation rule;
- threshold rule;
- metric implementation; and
- bootstrap procedure.

The final test command should be separate from development evaluation and produce an append-only result artifact.

## 10. Prediction and evaluation design

### 10.1 Prediction schemas

Recording/row prediction artifact:

| Field | Meaning |
|---|---|
| `run_id` | Experiment identity |
| `model_family` | Compared model |
| `partition_or_fold` | Evaluation source |
| `sample_key` | Recording or feature-row key |
| `participant_key` | Grouping key |
| `target` | Validated binary target |
| `score` | Model output before participant aggregation |

The participant artifact adds `recording_count` and contains the mean score for each participant/model. Evaluation verifies that a participant's target is consistent across source rows.

### 10.2 Metrics

Primary metric: participant-level AUROC.

Secondary metrics: participant-level precision, recall, F1, balanced accuracy, confusion matrix, and sensitivity at the frozen threshold. Include class prevalence because threshold-dependent metrics depend on it.

### 10.3 Uncertainty

For Mode A, resample participants with replacement and recompute each metric. For model differences, use the same participant sample for every model in each replicate. Prefer stratified participant resampling if needed to avoid single-class replicates; document the method and invalid-replicate handling.

For Mode B, generate one outer-held-out score per participant, then calculate metrics over the pooled out-of-fold participant table. Use participant-aware uncertainty and state the limitations of an 80-participant sample.

### 10.4 Calibration and thresholds

Assess calibration on validation participants using a reliability diagram and a numeric measure such as Brier score or expected calibration error. If recalibration is used, fit it only on non-test data and record the method. Without adequate evidence, label output as a **model score**, not a probability.

Choose the threshold before test evaluation, either by a prespecified value or a validation-only criterion. Do not retrospectively select it from the test ROC curve.

## 11. Optional demo design

The demo is a thin client over the same versioned inference pipeline used in evaluation.

```text
permitted example or transient upload
  -> format/size validation
  -> common audio preprocessing
  -> exported model adapter
  -> score + waveform + spectrogram
  -> non-diagnostic explanation
  -> immediate temporary-file cleanup
```

Required states are ready/input guidance, processing, successful experimental score, insufficient voiced audio, unsupported/corrupt audio, inference failure without sensitive detail, and a clear privacy/non-diagnostic notice.

The demo must not map scores to “has Parkinson's” or “healthy.” If a thresholded research classification is shown for coursework, it must be secondary and described as an experimental model output on this dataset.

## 12. Validation and testing strategy

### 12.1 Unit tests

- Participant grouping and disjointness.
- Label normalization and uncertain-label exclusion.
- Duration, voiced-content, and countdown-removal rules.
- Deterministic crop/pad behavior.
- Feature schema and failure flags.
- Train-only fitting of imputers/scalers.
- Participant mean aggregation and threshold application.
- Metric calculations on known examples.
- Bootstrap pairing across models.

### 12.2 Integration tests

- Synthetic audio through decode, preprocessing, representation, and prediction.
- Tiny grouped dataset through split, train, aggregate, and evaluate.
- Mode B outer fold with no group overlap.
- Cache invalidation after preprocessing/checkpoint changes.
- Saved model reload produces equivalent scores within tolerance.
- Report tables regenerate from fixture results.

### 12.3 Fail-fast data assertions

Stop execution when:

- participant sets overlap;
- targets disagree within a participant;
- a forbidden identifier appears in predictors;
- test data is passed to a fitting operation;
- a dataset or checkpoint fingerprint changes unexpectedly;
- compared models receive different evaluation participants without a reconciliation artifact; or
- a controlled file is targeted for publication.

## 13. Key risks and responses

| Risk | Consequence | Design response |
|---|---|---|
| mPower approval delayed | Raw-audio comparison cannot run | Explicit Mode B fallback with narrower claims |
| Participant leakage | Inflated performance | Immutable grouped allocations and fail-fast tests |
| Frequent contributors dominate | Biased training | Training recording cap and participant-level evaluation |
| Self-reported/noisy labels | Misleading interpretation | Preserve label provenance and use association language |
| Device/demographic confounding | Learns cohort artifacts | Document metadata; conditional subgroup/sensitivity checks |
| Small sample size | Unstable estimates | Compact models, frozen encoder, uncertainty intervals |
| Feature extraction failures | Unfair comparison | Quality flags and reconciled comparison population |
| Repeated test use | Optimistic estimate | Sealed test-release gate |
| Demo misuse | Medical harm or anxiety | Non-diagnostic language and no treatment guidance |
| Controlled-data exposure | Privacy/agreement breach | Storage zones, ignored paths, no public raw examples |

## 14. Design decisions still to record

Implementation should create short decision records for choices that depend on acquired data or measured feasibility:

1. operating mode and data-access outcome;
2. cohort thresholds and training recording cap;
3. audio duration, voice-activity, and normalization method;
4. acoustic library and feature definitions;
5. log-Mel configuration and CNN capacity;
6. frozen encoder checkpoint and pooling;
7. imbalance handling and hyperparameter search space;
8. fold count or holdout allocation details;
9. threshold and calibration policy;
10. bootstrap method; and
11. demo release decision.
