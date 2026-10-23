from __future__ import annotations

import json

import pandas as pd
import pytest

from parkinsons_voice.data.cohort import freeze_uci_cohort, verify_frozen_cohort
from parkinsons_voice.data.provenance import ProvenanceMismatchError, sha256_file


def _write_config(path, source, *, seed=2026):
    config = {
        "schema_version": "uci489-test-v1",
        "dataset_id": "uci-489-synthetic",
        "study_mode": "feature_only",
        "source_file": source.name,
        "source_sha256": sha256_file(source),
        "expected": {
            "rows": 60,
            "participants": 20,
            "recordings_per_participant": 3,
            "participant_class_counts": {"0": 10, "1": 10},
        },
        "folds": {
            "algorithm": "sha256_stratified_round_robin_v1",
            "seed": seed,
            "outer_splits": 5,
            "inner_splits": 4,
        },
    }
    path.write_text(json.dumps(config), encoding="utf-8")


def test_freeze_and_verify_round_trip(tmp_path, synthetic_uci_frame):
    source = tmp_path / "source.csv"
    synthetic_uci_frame().to_csv(source, index=False)
    config = tmp_path / "config.json"
    _write_config(config, source)
    artifacts = tmp_path / "artifacts"

    report = freeze_uci_cohort(source, artifacts, config)
    verified = verify_frozen_cohort(artifacts)

    assert report["status"] == "pass"
    assert report["cohort"]["participants"] == 20
    assert verified["fingerprints"] == report["fingerprints"]


def test_verifier_rejects_tampered_feature_artifact(tmp_path, synthetic_uci_frame):
    source = tmp_path / "source.csv"
    synthetic_uci_frame().to_csv(source, index=False)
    config = tmp_path / "config.json"
    _write_config(config, source)
    artifacts = tmp_path / "artifacts"
    freeze_uci_cohort(source, artifacts, config)

    features_path = artifacts / "eligible_features.restricted.csv"
    features = pd.read_csv(features_path)
    features.iloc[0, 3] += 1
    features.to_csv(features_path, index=False)

    with pytest.raises(ProvenanceMismatchError, match="eligible_table_fingerprint"):
        verify_frozen_cohort(artifacts)


def test_freeze_refuses_incompatible_existing_artifact(tmp_path, synthetic_uci_frame):
    source = tmp_path / "source.csv"
    synthetic_uci_frame().to_csv(source, index=False)
    config = tmp_path / "config.json"
    _write_config(config, source)
    artifacts = tmp_path / "artifacts"
    freeze_uci_cohort(source, artifacts, config)

    _write_config(config, source, seed=2027)
    with pytest.raises(ProvenanceMismatchError, match="configuration_fingerprint"):
        freeze_uci_cohort(source, artifacts, config)
