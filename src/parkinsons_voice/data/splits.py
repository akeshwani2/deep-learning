"""Deterministic participant-grouped fold allocation and leakage checks."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping

import pandas as pd

from parkinsons_voice.data.schema import validate_manifest

FOLD_ALGORITHM = "sha256_stratified_round_robin_v1"


class SplitValidationError(ValueError):
    """Raised when a participant allocation is incomplete or leaks groups."""


def assert_participant_disjoint(partitions: Mapping[str, Iterable[str]]) -> None:
    """Fail if any participant is present in more than one named partition."""

    normalized = {name: set(values) for name, values in partitions.items()}
    names = list(normalized)
    for index, left_name in enumerate(names):
        for right_name in names[index + 1 :]:
            overlap = normalized[left_name] & normalized[right_name]
            if overlap:
                raise SplitValidationError(
                    f"participant overlap between {left_name} and {right_name}: "
                    f"{sorted(overlap)[:3]}"
                )


def _participant_table(manifest: pd.DataFrame) -> pd.DataFrame:
    validated = validate_manifest(manifest)
    eligible = validated[validated["eligible"]]
    return (
        eligible.loc[:, ["participant_key", "label"]]
        .drop_duplicates()
        .sort_values("participant_key", kind="mergesort")
        .reset_index(drop=True)
    )


def _allocation_order(key: str, *, seed: int, namespace: str, label: int) -> str:
    value = f"{FOLD_ALGORITHM}|{seed}|{namespace}|{label}|{key}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def assign_stratified_group_folds(
    participants: pd.DataFrame,
    *,
    n_splits: int,
    seed: int,
    namespace: str,
    fold_column: str,
) -> pd.DataFrame:
    """Assign each participant to one deterministic, class-balanced fold."""

    required = {"participant_key", "label"}
    if not required.issubset(participants.columns):
        raise SplitValidationError(f"participants must contain columns: {sorted(required)}")
    if participants["participant_key"].duplicated().any():
        raise SplitValidationError("participant table contains duplicate participant keys")
    if n_splits < 2:
        raise SplitValidationError("n_splits must be at least 2")

    class_counts = participants.groupby("label")["participant_key"].nunique()
    if class_counts.empty or (class_counts < n_splits).any():
        raise SplitValidationError(
            f"each class needs at least {n_splits} participants; counts={class_counts.to_dict()}"
        )

    assignments: list[dict[str, object]] = []
    for label, class_rows in participants.groupby("label", sort=True):
        keys = class_rows["participant_key"].tolist()
        keys.sort(
            key=lambda key: _allocation_order(
                key,
                seed=seed,
                namespace=namespace,
                label=int(label),
            )
        )
        for index, participant_key in enumerate(keys):
            assignments.append(
                {
                    "participant_key": participant_key,
                    "label": int(label),
                    fold_column: index % n_splits,
                }
            )

    return pd.DataFrame(assignments).sort_values(
        [fold_column, "label", "participant_key"], kind="mergesort"
    ).reset_index(drop=True)


def build_nested_group_folds(
    manifest: pd.DataFrame,
    *,
    outer_splits: int,
    inner_splits: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create fixed outer folds and inner folds scoped to each outer training set."""

    participants = _participant_table(manifest)
    outer = assign_stratified_group_folds(
        participants,
        n_splits=outer_splits,
        seed=seed,
        namespace="outer",
        fold_column="outer_fold",
    )

    inner_parts: list[pd.DataFrame] = []
    for outer_fold in range(outer_splits):
        outer_train = outer[outer["outer_fold"] != outer_fold].loc[
            :, ["participant_key", "label"]
        ]
        inner = assign_stratified_group_folds(
            outer_train,
            n_splits=inner_splits,
            seed=seed,
            namespace=f"inner:outer={outer_fold}",
            fold_column="inner_fold",
        )
        inner.insert(0, "outer_fold", outer_fold)
        inner_parts.append(inner)

    nested = pd.concat(inner_parts, ignore_index=True)
    validate_outer_folds(manifest, outer, n_splits=outer_splits)
    validate_inner_folds(outer, nested, outer_splits=outer_splits, inner_splits=inner_splits)
    return outer, nested


def validate_outer_folds(
    manifest: pd.DataFrame,
    outer: pd.DataFrame,
    *,
    n_splits: int,
) -> None:
    """Validate coverage, group uniqueness, labels, and class presence."""

    required = {"participant_key", "label", "outer_fold"}
    if not required.issubset(outer.columns):
        raise SplitValidationError(f"outer allocations missing columns: {sorted(required)}")
    if outer["participant_key"].duplicated().any():
        raise SplitValidationError("a participant appears in multiple outer folds")

    expected = _participant_table(manifest).set_index("participant_key")["label"].sort_index()
    actual = outer.set_index("participant_key")["label"].sort_index()
    if set(expected.index) != set(actual.index):
        missing = sorted(set(expected.index) - set(actual.index))
        extra = sorted(set(actual.index) - set(expected.index))
        raise SplitValidationError(f"outer participant coverage mismatch; missing={missing}, extra={extra}")
    if not expected.equals(actual):
        raise SplitValidationError("outer allocation labels differ from the manifest")

    observed_folds = set(outer["outer_fold"])
    if observed_folds != set(range(n_splits)):
        raise SplitValidationError(f"outer folds must be 0..{n_splits - 1}")
    for fold, fold_rows in outer.groupby("outer_fold"):
        if set(fold_rows["label"]) != {0, 1}:
            raise SplitValidationError(f"outer fold {fold} does not contain both classes")

    partitions = {
        f"outer_{fold}": rows["participant_key"]
        for fold, rows in outer.groupby("outer_fold", sort=True)
    }
    assert_participant_disjoint(partitions)


def validate_inner_folds(
    outer: pd.DataFrame,
    inner: pd.DataFrame,
    *,
    outer_splits: int,
    inner_splits: int,
) -> None:
    """Validate each outer context's training-only inner allocations."""

    required = {"outer_fold", "participant_key", "label", "inner_fold"}
    if not required.issubset(inner.columns):
        raise SplitValidationError(f"inner allocations missing columns: {sorted(required)}")

    outer_labels = outer.set_index("participant_key")["label"].to_dict()
    all_participants = set(outer_labels)
    for outer_fold in range(outer_splits):
        context = inner[inner["outer_fold"] == outer_fold]
        if context["participant_key"].duplicated().any():
            raise SplitValidationError(
                f"participant appears in multiple inner folds for outer fold {outer_fold}"
            )

        outer_test = set(
            outer.loc[outer["outer_fold"] == outer_fold, "participant_key"].tolist()
        )
        expected_train = all_participants - outer_test
        actual_train = set(context["participant_key"])
        if actual_train != expected_train:
            raise SplitValidationError(
                f"inner coverage mismatch for outer fold {outer_fold}; "
                f"missing={sorted(expected_train - actual_train)[:3]}, "
                f"extra={sorted(actual_train - expected_train)[:3]}"
            )
        if outer_test & actual_train:
            raise SplitValidationError(f"outer-test participant leaked into inner fold {outer_fold}")
        if any(outer_labels[key] != label for key, label in zip(
            context["participant_key"], context["label"], strict=True
        )):
            raise SplitValidationError(f"inner labels differ from outer labels for fold {outer_fold}")

        observed_inner = set(context["inner_fold"])
        if observed_inner != set(range(inner_splits)):
            raise SplitValidationError(
                f"inner folds for outer fold {outer_fold} must be 0..{inner_splits - 1}"
            )
        for inner_fold, fold_rows in context.groupby("inner_fold"):
            if set(fold_rows["label"]) != {0, 1}:
                raise SplitValidationError(
                    f"inner fold {inner_fold} in outer fold {outer_fold} lacks a class"
                )


def assert_samples_inherit_participant_fold(
    samples: pd.DataFrame,
    participant_allocations: pd.DataFrame,
    *,
    fold_column: str,
) -> None:
    """Fail when any sample or derived segment disagrees with its participant fold."""

    required_samples = {"sample_key", "participant_key", fold_column}
    required_participants = {"participant_key", fold_column}
    if not required_samples.issubset(samples.columns):
        raise SplitValidationError(f"sample assignments missing: {sorted(required_samples)}")
    if not required_participants.issubset(participant_allocations.columns):
        raise SplitValidationError(
            f"participant assignments missing: {sorted(required_participants)}"
        )

    expected = participant_allocations.set_index("participant_key")[fold_column]
    if not samples["participant_key"].isin(expected.index).all():
        raise SplitValidationError("sample assignment contains an unknown participant")
    actual_expected = samples["participant_key"].map(expected)
    mismatch = samples[fold_column] != actual_expected
    if mismatch.any():
        bad_keys = samples.loc[mismatch, "sample_key"].tolist()
        raise SplitValidationError(f"samples do not inherit participant folds: {bad_keys[:3]}")
