from __future__ import annotations

import numpy as np
import pytest

from parkinsons_voice.data.uci489 import (
    FEATURE_COLUMNS,
    NON_PREDICTOR_COLUMNS,
    UCI489ValidationError,
    load_uci489,
)


def test_adapter_excludes_identifiers_target_and_metadata(tmp_path, synthetic_uci_frame):
    source = tmp_path / "source.csv"
    synthetic_uci_frame().to_csv(source, index=False)

    adapted = load_uci489(source)

    assert adapted.feature_names == FEATURE_COLUMNS
    assert set(NON_PREDICTOR_COLUMNS).isdisjoint(adapted.feature_names)
    assert list(adapted.model_rows.columns[:3]) == ["sample_key", "participant_key", "label"]
    assert adapted.summary["missing_feature_cells"] == 0


def test_adapter_allows_missing_predictor_for_later_fold_local_imputation(
    tmp_path,
    synthetic_uci_frame,
):
    raw = synthetic_uci_frame()
    raw.loc[0, FEATURE_COLUMNS[0]] = np.nan
    source = tmp_path / "source.csv"
    raw.to_csv(source, index=False)

    adapted = load_uci489(source)

    assert adapted.summary["missing_feature_cells"] == 1
    assert adapted.summary["features_with_missing_values"] == [FEATURE_COLUMNS[0]]


def test_adapter_rejects_duplicate_recording(tmp_path, synthetic_uci_frame):
    raw = synthetic_uci_frame()
    raw.loc[1, ["ID", "Recording"]] = raw.loc[0, ["ID", "Recording"]]
    source = tmp_path / "source.csv"
    raw.to_csv(source, index=False)

    with pytest.raises(UCI489ValidationError, match="duplicate"):
        load_uci489(source)


def test_adapter_rejects_inconsistent_participant_label(tmp_path, synthetic_uci_frame):
    raw = synthetic_uci_frame()
    raw.loc[0, "Status"] = 1
    source = tmp_path / "source.csv"
    raw.to_csv(source, index=False)

    with pytest.raises(UCI489ValidationError, match="inconsistent"):
        load_uci489(source)


def test_adapter_rejects_nonnumeric_predictor(tmp_path, synthetic_uci_frame):
    raw = synthetic_uci_frame()
    raw[FEATURE_COLUMNS[0]] = raw[FEATURE_COLUMNS[0]].astype(object)
    raw.loc[0, FEATURE_COLUMNS[0]] = "not-a-number"
    source = tmp_path / "source.csv"
    raw.to_csv(source, index=False)

    with pytest.raises(UCI489ValidationError, match="nonnumeric"):
        load_uci489(source)
