from __future__ import annotations

import pandas as pd
import pytest

from parkinsons_voice.data.splits import (
    SplitValidationError,
    assert_participant_disjoint,
    assert_samples_inherit_participant_fold,
    build_nested_group_folds,
    validate_inner_folds,
    validate_outer_folds,
)
from parkinsons_voice.data.uci489 import load_uci489


def test_nested_folds_are_balanced_and_group_disjoint(tmp_path, synthetic_uci_frame):
    source = tmp_path / "source.csv"
    synthetic_uci_frame().to_csv(source, index=False)
    manifest = load_uci489(source).manifest

    outer, inner = build_nested_group_folds(
        manifest,
        outer_splits=5,
        inner_splits=4,
        seed=2026,
    )

    validate_outer_folds(manifest, outer, n_splits=5)
    validate_inner_folds(outer, inner, outer_splits=5, inner_splits=4)
    assert (outer.groupby(["outer_fold", "label"]).size() == 2).all()
    assert (inner.groupby(["outer_fold", "inner_fold", "label"]).size() == 2).all()


def test_participant_overlap_fails():
    with pytest.raises(SplitValidationError, match="overlap"):
        assert_participant_disjoint({"train": ["p1", "p2"], "test": ["p2", "p3"]})


def test_derived_samples_must_inherit_participant_fold(tmp_path, synthetic_uci_frame):
    source = tmp_path / "source.csv"
    synthetic_uci_frame().to_csv(source, index=False)
    manifest = load_uci489(source).manifest
    outer, _ = build_nested_group_folds(
        manifest,
        outer_splits=5,
        inner_splits=4,
        seed=2026,
    )
    samples = manifest.loc[:, ["sample_key", "participant_key"]].merge(
        outer.loc[:, ["participant_key", "outer_fold"]],
        on="participant_key",
        validate="many_to_one",
    )
    assert_samples_inherit_participant_fold(samples, outer, fold_column="outer_fold")

    corrupted = samples.copy()
    corrupted.loc[0, "outer_fold"] = (int(corrupted.loc[0, "outer_fold"]) + 1) % 5
    with pytest.raises(SplitValidationError, match="inherit"):
        assert_samples_inherit_participant_fold(corrupted, outer, fold_column="outer_fold")


def test_duplicate_outer_participant_fails(tmp_path, synthetic_uci_frame):
    source = tmp_path / "source.csv"
    synthetic_uci_frame().to_csv(source, index=False)
    manifest = load_uci489(source).manifest
    outer, _ = build_nested_group_folds(
        manifest,
        outer_splits=5,
        inner_splits=4,
        seed=2026,
    )
    corrupted = pd.concat([outer, outer.iloc[[0]]], ignore_index=True)

    with pytest.raises(SplitValidationError, match="multiple outer folds"):
        validate_outer_folds(manifest, corrupted, n_splits=5)
