"""Canonical sample-manifest schema and validation."""

from __future__ import annotations

import re

import pandas as pd

MANIFEST_SCHEMA_VERSION = "canonical-sample-manifest-v1"

MANIFEST_COLUMNS = (
    "sample_key",
    "participant_key",
    "label",
    "label_source",
    "source_kind",
    "source_locator",
    "task_type",
    "duration_seconds",
    "voiced_seconds",
    "eligible",
    "exclusion_code",
    "metadata_available",
    "source_fingerprint",
)

SHAREABLE_MANIFEST_COLUMNS = tuple(
    column for column in MANIFEST_COLUMNS if column != "source_locator"
)

EXCLUSION_CODES = (
    "MISSING_PARTICIPANT_ID",
    "MISSING_OR_UNCERTAIN_LABEL",
    "WRONG_TASK",
    "DECODE_FAILURE",
    "INSUFFICIENT_AUDIO",
    "INSUFFICIENT_VOICED_CONTENT",
    "DUPLICATE_SAMPLE",
    "FEATURE_EXTRACTION_FAILURE",
)

SOURCE_KINDS = frozenset({"audio", "feature_row"})
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ManifestValidationError(ValueError):
    """Raised when the canonical sample manifest violates its contract."""


def _require_nonempty_strings(frame: pd.DataFrame, column: str) -> None:
    values = frame[column]
    if values.isna().any() or values.astype(str).str.strip().eq("").any():
        raise ManifestValidationError(f"{column} must contain non-empty values")


def validate_manifest(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate and return a defensive copy of a canonical sample manifest."""

    missing = sorted(set(MANIFEST_COLUMNS) - set(frame.columns))
    if missing:
        raise ManifestValidationError(f"manifest is missing columns: {missing}")

    validated = frame.loc[:, MANIFEST_COLUMNS].copy()
    if validated.empty:
        raise ManifestValidationError("manifest must contain at least one row")

    for column in (
        "sample_key",
        "participant_key",
        "label_source",
        "source_kind",
        "source_locator",
        "task_type",
        "source_fingerprint",
    ):
        _require_nonempty_strings(validated, column)

    if validated["sample_key"].duplicated().any():
        duplicates = validated.loc[validated["sample_key"].duplicated(), "sample_key"].tolist()
        raise ManifestValidationError(f"duplicate sample_key values: {duplicates[:3]}")

    if not set(validated["source_kind"]).issubset(SOURCE_KINDS):
        unknown = sorted(set(validated["source_kind"]) - SOURCE_KINDS)
        raise ManifestValidationError(f"unknown source_kind values: {unknown}")

    if not validated["source_fingerprint"].astype(str).map(_SHA256_RE.fullmatch).all():
        raise ManifestValidationError("source_fingerprint values must be lowercase SHA-256 hex")

    eligible_values = set(validated["eligible"].dropna().tolist())
    if validated["eligible"].isna().any() or not eligible_values.issubset({True, False}):
        raise ManifestValidationError("eligible must be boolean and non-missing")

    eligible = validated[validated["eligible"]]
    if eligible["label"].isna().any() or not set(eligible["label"]).issubset({0, 1}):
        raise ManifestValidationError("eligible rows must have binary labels")
    if eligible["exclusion_code"].notna().any():
        raise ManifestValidationError("eligible rows cannot have exclusion codes")

    excluded = validated[~validated["eligible"]]
    if excluded["exclusion_code"].isna().any():
        raise ManifestValidationError("ineligible rows must have an exclusion code")
    invalid_codes = set(excluded["exclusion_code"].dropna()) - set(EXCLUSION_CODES)
    if invalid_codes:
        raise ManifestValidationError(f"unknown exclusion codes: {sorted(invalid_codes)}")

    for column in ("duration_seconds", "voiced_seconds"):
        numeric = pd.to_numeric(validated[column], errors="coerce")
        if numeric.dropna().lt(0).any():
            raise ManifestValidationError(f"{column} cannot be negative")

    label_counts = eligible.groupby("participant_key", sort=False)["label"].nunique(dropna=False)
    inconsistent = label_counts[label_counts != 1]
    if not inconsistent.empty:
        raise ManifestValidationError(
            "eligible participants must have one label; inconsistent keys: "
            f"{inconsistent.index.tolist()[:3]}"
        )

    return validated


def shareable_manifest(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a validated manifest without restricted source locators."""

    validated = validate_manifest(frame)
    return validated.loc[:, SHAREABLE_MANIFEST_COLUMNS].copy()
