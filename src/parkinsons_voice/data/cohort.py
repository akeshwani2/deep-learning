"""Freeze and verify the Mode B cohort and nested group folds."""

from __future__ import annotations

import json
import platform
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from parkinsons_voice import __version__
from parkinsons_voice.data.provenance import (
    ProvenanceMismatchError,
    fingerprint_frame,
    fingerprint_json,
    require_matching_provenance,
    sha256_file,
)
from parkinsons_voice.data.schema import (
    EXCLUSION_CODES,
    MANIFEST_COLUMNS,
    MANIFEST_SCHEMA_VERSION,
    shareable_manifest,
    validate_manifest,
)
from parkinsons_voice.data.splits import (
    FOLD_ALGORITHM,
    build_nested_group_folds,
    validate_inner_folds,
    validate_outer_folds,
)
from parkinsons_voice.data.uci489 import FEATURE_COLUMNS, UCI489Data, load_uci489

PROVENANCE_KEYS = (
    "source_fingerprint",
    "configuration_fingerprint",
    "implementation_fingerprint",
    "manifest_fingerprint",
    "eligible_table_fingerprint",
    "outer_folds_fingerprint",
    "inner_folds_fingerprint",
)


class CohortValidationError(ValueError):
    """Raised when the source data does not meet the frozen cohort contract."""


def _implementation_fingerprint() -> str:
    """Fingerprint the code files that define cohort membership and allocation."""

    data_package = Path(__file__).parent
    implementation_files = ("cohort.py", "provenance.py", "schema.py", "splits.py", "uci489.py")
    hashes = {
        name: sha256_file(data_package / name)
        for name in implementation_files
    }
    return fingerprint_json(hashes)


def load_config(path: str | Path) -> dict[str, Any]:
    """Read a cohort configuration JSON document."""

    with Path(path).open(encoding="utf-8") as source:
        config = json.load(source)
    if config.get("study_mode") != "feature_only":
        raise CohortValidationError("M1 requires the selected feature_only operating mode")
    return config


def _validate_expected_dataset(data: UCI489Data, config: dict[str, Any]) -> None:
    expected = config["expected"]
    summary = data.summary
    comparisons = {
        "rows": (summary["rows"], expected["rows"]),
        "participants": (summary["participants"], expected["participants"]),
        "minimum recordings per participant": (
            summary["recordings_per_participant"]["minimum"],
            expected["recordings_per_participant"],
        ),
        "maximum recordings per participant": (
            summary["recordings_per_participant"]["maximum"],
            expected["recordings_per_participant"],
        ),
        "participant class counts": (
            summary["participant_class_counts"],
            expected["participant_class_counts"],
        ),
    }
    mismatches = {
        name: {"actual": actual, "expected": wanted}
        for name, (actual, wanted) in comparisons.items()
        if actual != wanted
    }
    if mismatches:
        raise CohortValidationError(f"source data does not match UCI 489 expectations: {mismatches}")


def _fold_summary(manifest: pd.DataFrame, outer: pd.DataFrame, inner: pd.DataFrame) -> dict:
    sample_counts = (
        manifest.loc[manifest["eligible"], ["participant_key", "label"]]
        .merge(outer, on=["participant_key", "label"], validate="many_to_one")
        .groupby(["outer_fold", "label"])
        .size()
    )
    participant_counts = outer.groupby(["outer_fold", "label"]).size()
    outer_summary = {
        str(fold): {
            str(label): {
                "participants": int(participant_counts.loc[(fold, label)]),
                "rows": int(sample_counts.loc[(fold, label)]),
            }
            for label in (0, 1)
        }
        for fold in sorted(outer["outer_fold"].unique())
    }

    inner_summary: dict[str, dict[str, dict[str, int]]] = {}
    for outer_fold in sorted(inner["outer_fold"].unique()):
        context = inner[inner["outer_fold"] == outer_fold]
        counts = context.groupby(["inner_fold", "label"]).size()
        inner_summary[str(outer_fold)] = {
            str(inner_fold): {str(label): int(counts.loc[(inner_fold, label)]) for label in (0, 1)}
            for inner_fold in sorted(context["inner_fold"].unique())
        }
    return {"outer_test_counts": outer_summary, "inner_validation_participant_counts": inner_summary}


def _build_provenance(
    *,
    data: UCI489Data,
    config: dict[str, Any],
    outer: pd.DataFrame,
    inner: pd.DataFrame,
) -> dict[str, Any]:
    return {
        "artifact_schema_version": "mode-b-cohort-v1",
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "source_fingerprint": data.source_fingerprint,
        "configuration_fingerprint": fingerprint_json(config),
        "implementation_fingerprint": _implementation_fingerprint(),
        "manifest_fingerprint": fingerprint_frame(
            data.manifest,
            sort_by=["sample_key"],
            columns=MANIFEST_COLUMNS,
        ),
        "eligible_table_fingerprint": fingerprint_frame(
            data.model_rows,
            sort_by=["sample_key"],
        ),
        "outer_folds_fingerprint": fingerprint_frame(
            outer,
            sort_by=["participant_key"],
        ),
        "inner_folds_fingerprint": fingerprint_frame(
            inner,
            sort_by=["outer_fold", "participant_key"],
        ),
        "fold_algorithm": FOLD_ALGORITHM,
        "package_version": __version__,
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def freeze_uci_cohort(
    source_path: str | Path,
    output_dir: str | Path,
    config_path: str | Path,
) -> dict[str, Any]:
    """Validate UCI 489, freeze manifests/folds, and return a safe summary."""

    source = Path(source_path)
    output = Path(output_dir)
    config = load_config(config_path)

    actual_source_hash = sha256_file(source)
    if actual_source_hash != config["source_sha256"]:
        raise ProvenanceMismatchError(
            "source checksum mismatch; "
            f"actual={actual_source_hash}, expected={config['source_sha256']}"
        )

    data = load_uci489(source)
    _validate_expected_dataset(data, config)
    folds = config["folds"]
    if folds["algorithm"] != FOLD_ALGORITHM:
        raise CohortValidationError(
            f"unsupported fold algorithm {folds['algorithm']}; expected {FOLD_ALGORITHM}"
        )
    outer, inner = build_nested_group_folds(
        data.manifest,
        outer_splits=int(folds["outer_splits"]),
        inner_splits=int(folds["inner_splits"]),
        seed=int(folds["seed"]),
    )
    provenance = _build_provenance(data=data, config=config, outer=outer, inner=inner)

    prior_path = output / "provenance.json"
    if prior_path.exists():
        with prior_path.open(encoding="utf-8") as source_file:
            prior = json.load(source_file)
        require_matching_provenance(prior, provenance, PROVENANCE_KEYS)

    exclusion_counts = {code: 0 for code in EXCLUSION_CODES}
    report: dict[str, Any] = {
        "status": "pass",
        "study_mode": "feature_only",
        "dataset_id": config["dataset_id"],
        "schema_version": config["schema_version"],
        "source_file": config["source_file"],
        "source_sha256": data.source_fingerprint,
        "cohort": data.summary,
        "exclusions": {"total": 0, "by_reason": exclusion_counts},
        "predictors": {
            "count": len(FEATURE_COLUMNS),
            "allow_list": list(FEATURE_COLUMNS),
            "forbidden": ["ID", "Recording", "Status", "Gender"],
        },
        "folds": {
            "algorithm": FOLD_ALGORITHM,
            "seed": int(folds["seed"]),
            "outer_splits": int(folds["outer_splits"]),
            "inner_splits": int(folds["inner_splits"]),
            **_fold_summary(data.manifest, outer, inner),
        },
        "fingerprints": {key: provenance[key] for key in PROVENANCE_KEYS},
        "scope": {
            "required_models": ["regularized_logistic_regression", "small_mlp"],
            "claims_excluded": [
                "raw_audio_performance",
                "spectrogram_cnn_performance",
                "pretrained_speech_embedding_performance",
                "clinical_diagnosis_or_validation",
            ],
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }

    output.mkdir(parents=True, exist_ok=True)
    data.manifest.to_csv(output / "manifest.restricted.csv", index=False)
    shareable_manifest(data.manifest).to_csv(output / "manifest.shareable.csv", index=False)
    data.model_rows.to_csv(output / "eligible_features.restricted.csv", index=False)
    outer.to_csv(output / "outer_folds.restricted.csv", index=False)
    inner.to_csv(output / "inner_folds.restricted.csv", index=False)
    _write_json(output / "provenance.json", provenance)
    _write_json(output / "validation_report.json", report)
    return report


def verify_frozen_cohort(artifact_dir: str | Path) -> dict[str, Any]:
    """Revalidate frozen cohort files and reject changed artifacts."""

    artifact = Path(artifact_dir)
    manifest = pd.read_csv(artifact / "manifest.restricted.csv")
    model_rows = pd.read_csv(artifact / "eligible_features.restricted.csv")
    outer = pd.read_csv(artifact / "outer_folds.restricted.csv")
    inner = pd.read_csv(artifact / "inner_folds.restricted.csv")
    with (artifact / "provenance.json").open(encoding="utf-8") as source:
        provenance = json.load(source)
    with (artifact / "validation_report.json").open(encoding="utf-8") as source:
        report = json.load(source)

    require_matching_provenance(
        provenance,
        report["fingerprints"],
        PROVENANCE_KEYS,
    )
    validate_manifest(manifest)
    outer_splits = int(report["folds"]["outer_splits"])
    inner_splits = int(report["folds"]["inner_splits"])
    validate_outer_folds(manifest, outer, n_splits=outer_splits)
    validate_inner_folds(
        outer,
        inner,
        outer_splits=outer_splits,
        inner_splits=inner_splits,
    )

    recalculated = {
        "implementation_fingerprint": _implementation_fingerprint(),
        "manifest_fingerprint": fingerprint_frame(
            manifest,
            sort_by=["sample_key"],
            columns=MANIFEST_COLUMNS,
        ),
        "eligible_table_fingerprint": fingerprint_frame(
            model_rows,
            sort_by=["sample_key"],
        ),
        "outer_folds_fingerprint": fingerprint_frame(
            outer,
            sort_by=["participant_key"],
        ),
        "inner_folds_fingerprint": fingerprint_frame(
            inner,
            sort_by=["outer_fold", "participant_key"],
        ),
    }
    require_matching_provenance(
        provenance,
        recalculated,
        recalculated.keys(),
    )
    return report
