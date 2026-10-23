# Implementation Tasks

## 1. How to use this plan

This plan is ordered by dependency and contains separate branches for the primary raw-audio study and feature-only fallback. Complete the shared foundation first, record the operating-mode decision, and execute only the selected required branch. Optional tasks must not delay the core participant-level comparison.

Status: `[ ]` not started · `[~]` in progress · `[x]` complete with evidence · `[!]` blocked with owner/reason.

Priority: **P0** valid result · **P1** complete deliverable · **P2** optional extension.

## 2. Milestones

| Milestone | Status | Exit condition |
|---|---|---|
| M0 — Governance ready | **Complete (2026-10-22)** | Mode B has a documented CC BY 4.0 data path, repository protections, and an accepted mode decision. Evidence: [governance status](data-access-status.md), [data manifest](data-manifest.md), and [DL-002](decision-log.md#dl-002-select-the-feature-only-operating-mode). |
| M1 — Cohort frozen | **Complete (2026-10-22)** | The Mode B manifest, 240-row/80-participant cohort, zero exclusions, and fixed 5x4 grouped folds pass schema, leakage, and provenance validation. Evidence: [M1 cohort report](m1-cohort-report.md). |
| M2 — Baseline complete | Not started | Linear acoustic/feature baseline produces participant-level validation or out-of-fold predictions |
| M3 — Comparison complete | Not started | Every required model has frozen artifacts and comparable predictions |
| M4 — Final evaluation complete | Not started | Participant metrics, uncertainty, figures, and limitations are generated without test leakage |
| M5 — Deliverables ready | Not started | Reproducibility run, report, and conditional demo checks pass |

## 3. Phase 0 — Project foundation

- [x] **T0.1 (P0): Record project decisions and assumptions**
  - Create a lightweight decision log.
  - Record the non-diagnostic purpose, candidate data modes, and owners.
  - Evidence: [DL-001: Project framing and candidate operating modes](decision-log.md#dl-001-project-framing-and-candidate-operating-modes), dated 2026-10-22.

- [x] **T0.2 (P0): Establish a reproducible Python environment**
  - Select a supported Python version.
  - Pin direct dependencies and capture resolved versions.
  - Include grouped splitting, audio, acoustic feature, deep learning, speech encoder, metric, and plotting libraries only as needed.
  - Evidence: Python 3.12 in [`pyproject.toml`](../pyproject.toml), resolved versions in [`uv.lock`](../uv.lock), and passing clean sync/import checks.

- [x] **T0.3 (P0): Protect data and secrets**
  - Ignore source data, derived audio, participant allocations, tokens, caches, checkpoints, uploads, and local artifacts.
  - Add a sample environment file containing names only, never secret values.
  - Verify tracked files contain no restricted data or credentials.
  - Evidence: [repository safety verification](repository-safety.md), `.gitignore`, and `.env.example`.

- [x] **T0.4 (P0): Create package and command structure**
  - Add modules for data, audio, features, models, training, evaluation, and optional demo code.
  - Add configuration and artifact directories.
  - Keep notebooks for exploration; move reusable logic into modules.
  - Evidence: [`src/parkinsons_voice`](../src/parkinsons_voice) package structure and passing `parkinsons-voice --help` smoke test.

- [x] **T0.5 (P0): Add automated quality checks**
  - Configure tests, formatting/linting, and a minimal CI or local check command.
  - Add synthetic fixtures only; never controlled samples.
  - Evidence: [`scripts/check.sh`](../scripts/check.sh) performs frozen sync, lint, tests, and command-help smoke testing.

## 4. Phase 1 — Data access and mode decision

- [x] **T1.1 (P0): Complete or track mPower access requirements**
  - Verify Synapse status and required certification.
  - Submit intended data use and accept applicable conditions.
  - Seek institutional/course guidance where needed.
  - Keep downloads outside the repository.
  - Evidence: [non-sensitive access-status record](data-access-status.md#mpower-access-tracker).

- [x] **T1.2 (P0): Define the decision deadline and feasibility criteria**
  - Set the latest date Mode A can begin without jeopardizing delivery.
  - Require raw audio, participant IDs, usable target labels, permitted analysis, and adequate decode success.
  - Evidence: [dated deadline and measurable feasibility criteria](data-access-status.md#mode-a-decision-deadline-and-feasibility-criteria).

- [x] **T1.3 (P0): Select and record the operating mode**
  - Choose **Mode A** if feasibility criteria pass by the deadline.
  - Otherwise choose **Mode B** and narrow the research claim.
  - Record date, owner, rationale, data version/retrieval date, and permitted uses.
  - Depends on: T1.1, T1.2.
  - Evidence: accepted [DL-002 mode decision](decision-log.md#dl-002-select-the-feature-only-operating-mode) and [shareable data manifest](data-manifest.md).

## 5. Phase 2 — Shared data and leakage controls

- [x] **T2.1 (P0): Implement the canonical sample manifest**
  - Define fields, types, uniqueness constraints, and non-identifying keys.
  - Separate restricted locators from shareable summaries.
  - Evidence: [manifest schema and validation report](m1-cohort-report.md#canonical-manifest-schema) plus synthetic schema tests.

- [x] **T2.2 (P0): Implement fingerprinting and provenance**
  - Fingerprint source metadata, cohort rules, configurations, and derived artifacts.
  - Refuse incompatible cached features or embeddings.
  - Evidence: [frozen fingerprint inventory](m1-cohort-report.md#frozen-fingerprints) and passing mismatch/tampering tests.

- [x] **T2.3 (P0): Implement participant-disjointness assertions**
  - Check all partition/fold pairs for overlap.
  - Check derived segments inherit the correct allocation.
  - Make failures fatal.
  - Evidence: [fold validation report](m1-cohort-report.md#frozen-participant-folds) and passing plus deliberately failing overlap/inheritance tests.

- [ ] **T2.4 (P0): Implement train-only transform tests**
  - Encapsulate imputation, scaling, selection, and calibration in fold-aware pipelines.
  - Prove held-out rows do not contribute fitted statistics.
  - Evidence: fixture test with intentionally shifted held-out values.

- [ ] **T2.5 (P1): Define experiment and result schemas**
  - Specify run metadata, sample predictions, participant predictions, metrics, and artifact metadata.
  - Evidence: schema documentation and load/save test.

## 6. Phase 3A — Mode A raw-audio cohort

Complete only if Mode A is selected.

- [ ] **T3A.1 (P0): Inventory approved mPower fields and files**
  - Map recording IDs, participant IDs, voice-activity markers, diagnosis status, permitted demographics, and device/context metadata.
  - Confirm label coding against source documentation.
  - Evidence: restricted data dictionary and shareable field-name summary.

- [ ] **T3A.2 (P0): Build ingestion and decode validation**
  - Normalize source records into the canonical manifest.
  - Probe format, sample rate, channels, duration, and decode success.
  - Record failures without identifying values.
  - Evidence: ingestion summary and decode-failure counts.

- [ ] **T3A.3 (P0): Implement label and eligibility rules**
  - Retain sustained-vowel samples with participant ID and sufficient voiced content.
  - Exclude missing/uncertain status and non-task files.
  - Assign stable exclusion codes.
  - Evidence: reviewed rules and reconciled exclusion table.

- [ ] **T3A.4 (P0): Validate countdown and voiced-region policy**
  - Inspect a permitted representative sample across labels/devices without exporting recordings.
  - Choose deterministic countdown removal and voiced-content detection.
  - Set minimum voiced duration, analysis duration, crop, and padding policy.
  - Evidence: decision record, aggregate QC, and tests.

- [ ] **T3A.5 (P0): Freeze the eligible cohort**
  - Produce participant/recording counts by class, duration summaries, and permitted metadata coverage.
  - Review imbalance and missingness.
  - Fingerprint the cohort manifest.
  - Depends on: T3A.1–T3A.4.
  - Evidence: cohort table and fingerprint.

- [ ] **T3A.6 (P0): Create the participant holdout split**
  - Allocate approximately 70%/15%/15% by participant with class stratification.
  - Record seed, algorithm, and library version.
  - Confirm class presence and participant disjointness.
  - Persist restricted allocation and publish only safe counts/fingerprint.
  - Depends on: T3A.5, T2.3.
  - Evidence: split validation report.

- [ ] **T3A.7 (P0): Set the training recording cap**
  - Inspect the training recording-count distribution only.
  - Choose maximum recordings per training participant and deterministic sampling.
  - Keep validation/test intact unless a computational rule was prespecified.
  - Evidence: decision record and before/after counts.

## 7. Phase 3B — Mode B feature-only cohort

Complete only if Mode B is selected.

- [x] **T3B.1 (P0): Retrieve and verify UCI dataset 489**
  - Record source URL, retrieval date, license, files, and checksums.
  - Confirm expected 240 rows, 80 participants, three repeated `/a/` recordings per person, and class coding using authoritative documentation.
  - Evidence: [data manifest](data-manifest.md) and [source verification](m1-cohort-report.md#source-verification).

- [x] **T3B.2 (P0): Implement the UCI adapter**
  - Map subject ID to participant key, target to label, and rows to sample keys.
  - Exclude identifiers and targets from predictors.
  - Validate numeric columns, missingness, duplicates, and participant label consistency.
  - Evidence: [adapter/schema summary](m1-cohort-report.md#uci-adapter-and-predictors) and passing validation tests.

- [x] **T3B.3 (P0): Create grouped outer and inner folds**
  - Choose fold count based on class/group feasibility.
  - Stratify by label and group by participant.
  - Persist identical outer folds for every model.
  - Depends on: T3B.2, T2.3.
  - Evidence: [group-disjoint fold report with class counts](m1-cohort-report.md#frozen-participant-folds).

- [x] **T3B.4 (P0): Freeze the fallback cohort and claims**
  - Fingerprint the eligible table and fold allocation.
  - Record that no raw-audio, spectrogram-CNN, or speech-embedding conclusions will be made.
  - Evidence: [cohort fingerprints](m1-cohort-report.md#frozen-fingerprints) and [scope statement](m1-cohort-report.md#scope-frozen-with-the-cohort).

## 8. Phase 4A — Mode A representations

- [ ] **T4A.1 (P0): Implement common audio preprocessing**
  - Decode, convert to mono, remove countdown, locate usable voice, crop/pad deterministically, and resample from a shared intermediate.
  - Return quality flags and valid lengths.
  - Evidence: tests and aggregate non-identifying QC.

- [ ] **T4A.2 (P0): Implement acoustic feature extraction**
  - Select and pin a maintained library.
  - Extract pitch, jitter, shimmer, harmonic-to-noise ratio, and MFCC summaries.
  - Flag failures and non-finite values.
  - Evidence: feature dictionary, tests, and failure report.

- [ ] **T4A.3 (P0): Reconcile acoustic extraction failures**
  - Quantify failures by partition/class.
  - Define a common-comparison policy without test outcomes.
  - Use training-fitted imputation where valid; never silently drop rows.
  - Evidence: signed decision and reconciled counts.

- [ ] **T4A.4 (P0): Implement log-Mel extraction**
  - Fix sample rate, FFT/window/hop, Mel bands, frequency bounds, log transform, and normalization.
  - Verify tensor shape and absence of held-out fitting.
  - Evidence: config, shape tests, and synthetic/permitted visualization.

- [ ] **T4A.5 (P0): Select and integrate the frozen encoder**
  - Record checkpoint, license, sample rate, library version, and output dimensions.
  - Freeze all encoder parameters and implement mask-aware temporal pooling.
  - Evidence: freeze assertion and embedding-shape test.

- [ ] **T4A.6 (P1): Add provenance-safe caching**
  - Cache deterministic features and embeddings.
  - Key caches by source and preprocessing/checkpoint fingerprints.
  - Evidence: cache hit and invalidation tests.

- [ ] **T4A.7 (P1): Define safe primary augmentation**
  - Specify gain and mild-noise ranges using training audio only.
  - Keep pitch/time changes outside the primary experiment.
  - Evidence: reviewed config and tests disabling augmentation outside training.

## 9. Phase 4B — Mode B preprocessing

- [ ] **T4B.1 (P0): Build tabular preprocessing**
  - Add fold-local missing-value handling and standardization.
  - Preserve feature names and prevent participant ID/target inclusion.
  - Evidence: feature allow-list and train-only fitting tests.

- [ ] **T4B.2 (P1): Produce training-only exploratory summaries**
  - Inspect distributions, correlations, missingness, and scale without using outer-test outcomes for selection.
  - Evidence: exploratory notebook/report with leakage caveat.

## 10. Phase 5 — Model implementation

### Shared infrastructure

- [ ] **T5.1 (P0): Implement the model adapter interface**
  - Standardize fit, predict, save, load, and provenance behavior.
  - Require sample and participant keys in prediction outputs.
  - Evidence: contract tests for each required model.

- [ ] **T5.2 (P0): Implement participant-level development evaluation**
  - Aggregate scores with the arithmetic mean.
  - Validate one target per participant.
  - Compute participant AUROC and secondary metrics.
  - Evidence: tests against hand-calculated fixtures.

- [ ] **T5.3 (P0): Predeclare hyperparameter search spaces**
  - Keep searches small and appropriate to participant count.
  - Record regularization, architecture, optimization, imbalance, and early-stopping choices.
  - Evidence: versioned experiment configs.

### Mode A models

- [ ] **T5A.1 (P0): Train the acoustic logistic baseline**
  - Use training-fitted imputer → scaler → regularized logistic regression.
  - Select regularization on validation participant AUROC.
  - Evidence: artifact, validation predictions, and run metadata.

- [ ] **T5A.2 (P0): Train the compact log-Mel CNN**
  - Use two or three convolutional blocks, global pooling, and one binary logit.
  - Use binary cross-entropy and early stopping on validation participant AUROC.
  - Record optimizer, batch size, epochs, selected epoch, hardware, and time.
  - Evidence: selected checkpoint and learning curves.

- [ ] **T5A.3 (P0): Train the frozen-embedding linear model**
  - Assert the encoder remains frozen.
  - Train a regularized linear head over pooled embeddings.
  - Select regularization using validation participants only.
  - Evidence: artifact, freeze assertion, and validation predictions.

- [ ] **T5A.4 (P1): Compare development results and freeze candidates**
  - Confirm common validation coverage or document reconciled exceptions.
  - Review performance, failures, and compute without test access.
  - Freeze one configuration/checkpoint per required model.
  - Evidence: pre-test freeze record.

### Mode B models

- [ ] **T5B.1 (P0): Train grouped-CV logistic regression**
  - Tune regularization within outer training sets using grouped inner folds.
  - Produce exactly one outer-held-out score per row.
  - Evidence: fold settings and out-of-fold predictions.

- [ ] **T5B.2 (P0): Train grouped-CV small MLP**
  - Predeclare a small architecture search appropriate to 80 participants.
  - Use grouped inner validation and early stopping.
  - Produce exactly one outer-held-out score per row.
  - Evidence: fold settings/checkpoints and out-of-fold predictions.

- [ ] **T5B.3 (P0): Validate fallback comparison integrity**
  - Confirm identical outer folds and participant coverage.
  - Confirm preprocessing was fitted inside every outer training fold.
  - Evidence: comparison-integrity report.

## 11. Phase 6 — Final evaluation and uncertainty

- [ ] **T6.1 (P0): Freeze evaluation protocol**
  - Record aggregation, metrics, threshold, calibration assessment, uncertainty method, and subgroup criteria.
  - Lock Mode A code/config before test release.
  - Evidence: dated evaluation protocol.

- [ ] **T6.2 (P0): Generate participant prediction tables**
  - Mode A: evaluate each frozen model once on held-out test data.
  - Mode B: pool only outer-held-out predictions.
  - Average source scores per participant and record source count.
  - Evidence: schema-valid tables with matching model coverage.

- [ ] **T6.3 (P0): Compute primary and secondary metrics**
  - Calculate AUROC, precision, recall, F1, balanced accuracy, confusion matrix, and sensitivity at the frozen threshold.
  - Report prevalence and participant/recording counts.
  - Evidence: machine-readable metrics and generated table.

- [ ] **T6.4 (P0): Quantify uncertainty**
  - Mode A: participant bootstrap intervals and paired model differences.
  - Mode B: participant-aware uncertainty over pooled outer-held-out predictions with small-sample caveats.
  - Record seed, replicate count, interval method, and invalid replicate handling.
  - Evidence: uncertainty artifact and reproducibility test.

- [ ] **T6.5 (P1): Assess calibration and score wording**
  - Use validation or fold-appropriate held-out data without contaminating selection.
  - Generate a reliability view and numeric measure.
  - Decide whether output can be called a probability; default to “model score.”
  - Evidence: calibration artifact and wording decision.

- [ ] **T6.6 (P1): Run conditional subgroup/sensitivity analyses**
  - Proceed only with adequate metadata, permissions, and subgroup sizes.
  - Predeclare comparisons and report denominators/uncertainty.
  - Evidence: go/no-go decision and clearly labeled exploratory results if run.

- [ ] **T6.7 (P2): Create exploratory representation plots**
  - Use PCA/UMAP only as exploratory visualization.
  - Avoid leaking test labels into model decisions.
  - Label plots non-confirmatory and non-clinical.
  - Evidence: generated figures and captions.

## 12. Phase 7 — Reporting

- [ ] **T7.1 (P1): Generate cohort flow and summary**
  - Include source records, exclusions, eligible counts, allocations, durations, prevalence, and permitted metadata coverage.
  - Evidence: generated table/flow tied to cohort artifacts.

- [ ] **T7.2 (P1): Generate model and experiment tables**
  - Include representation, classifier, selected settings, parameter counts where meaningful, training time, versions, and participant coverage.
  - Evidence: tables generated from run metadata.

- [ ] **T7.3 (P1): Write results with bounded claims**
  - State selected mode and report measured outcomes with uncertainty.
  - Do not promise superiority, imply diagnosis, or treat cross-dataset differences as architecture effects.
  - Evidence: reviewed results section with metric-source links.

- [ ] **T7.4 (P1): Write limitations and responsible use**
  - Address self-report, label noise, confounding, sustained-vowel scope, repeated measures, sample size, subgroup limits, error harms, and external validity.
  - Evidence: review against proposal and requirements.

- [ ] **T7.5 (P1): Document reproducibility**
  - Provide setup, local data layout, configuration, commands, artifact locations, and controlled-data boundaries.
  - Evidence: clean-environment or independent teammate verification.

- [ ] **T7.6 (P1): Audit proposal-to-report consistency**
  - Do not describe planned work as completed unless it ran.
  - Reconcile models, metrics, dataset, and counts across text, tables, figures, and demo.
  - Evidence: completed audit checklist.

## 13. Phase 8 — Optional educational demo

Start only after M4 and a recorded go/no-go decision.

- [ ] **T8.1 (P1, conditional): Decide whether a demo may ship**
  - Review model behavior, data terms, permitted examples, privacy, and schedule.
  - Select a model and define score semantics.
  - Evidence: go/no-go decision; no-go still satisfies the scientific project.

- [ ] **T8.2 (P1, conditional): Export and verify inference**
  - Package preprocessing and model together.
  - Compare exported predictions with the evaluation pipeline on permitted fixtures.
  - Evidence: inference-parity test.

- [ ] **T8.3 (P1, conditional): Build waveform/spectrogram/score flow**
  - Accept a permitted bundled example or transient upload.
  - Display waveform, spectrogram, and experimental score without diagnostic language.
  - Evidence: end-to-end test and non-restricted screenshots.

- [ ] **T8.4 (P0 if shipped): Implement privacy and cleanup**
  - Validate type/size/duration, avoid audio/identifier logs, and delete temporary uploads.
  - Confirm controlled examples are not bundled.
  - Evidence: cleanup test and release-content audit.

- [ ] **T8.5 (P0 if shipped): Add responsible-use and failure states**
  - State prominently that output cannot diagnose PD.
  - Handle corrupt, unsupported, empty, short, and insufficiently voiced audio.
  - Avoid treatment or individual medical guidance.
  - Evidence: content review and failure-path tests.

## 14. Phase 9 — Final quality gate

- [ ] **T9.1 (P0): Run the full test suite**
  - Include leakage, schema, aggregation, metric, cache, save/load, and report-generation tests.
  - Evidence: passing log tied to final code revision.

- [ ] **T9.2 (P0): Reproduce the selected experiment**
  - Use a clean environment and authorized data.
  - Compare cohort/split fingerprints, selected config, and metrics.
  - Evidence: reproducibility record and explained tolerances.

- [ ] **T9.3 (P0): Perform privacy and release audit**
  - Scan deliverables for credentials, IDs, controlled audio, restricted paths, and disallowed examples.
  - Verify license/attribution requirements.
  - Evidence: signed audit checklist.

- [ ] **T9.4 (P1): Perform scientific-claims audit**
  - Trace headline conclusions to participant-level held-out evidence.
  - Verify uncertainty, denominators, association language, and external-validity limits.
  - Evidence: claims-to-results traceability table.

- [ ] **T9.5 (P1): Package final deliverables**
  - Include report, code, environment/configs, safe aggregate artifacts, docs, and optional demo.
  - Exclude controlled data and machine-specific caches.
  - Evidence: release manifest.

## 15. Recommended first work session

Complete these tasks in order:

1. T0.1 — start the decision log.
2. T0.3 — add privacy-safe ignore rules before downloading data.
3. T1.1 and T1.2 — determine mPower status and a decision deadline.
4. T0.2 and T0.4 — establish environment and package structure.
5. T2.1 and T2.3 — build the manifest and leakage tests with synthetic data.

This sequence keeps both modes viable and puts leakage/privacy controls in place before real data enters the workflow.
