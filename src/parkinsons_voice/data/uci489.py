"""Adapter for UCI dataset 489."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from parkinsons_voice.data.provenance import sha256_file
from parkinsons_voice.data.schema import MANIFEST_COLUMNS, validate_manifest

PARTICIPANT_COLUMN = "ID"
RECORDING_COLUMN = "Recording"
TARGET_COLUMN = "Status"
METADATA_COLUMNS = ("Gender",)

FEATURE_COLUMNS = (
    "Jitter_rel",
    "Jitter_abs",
    "Jitter_RAP",
    "Jitter_PPQ",
    "Shim_loc",
    "Shim_dB",
    "Shim_APQ3",
    "Shim_APQ5",
    "Shi_APQ11",
    "HNR05",
    "HNR15",
    "HNR25",
    "HNR35",
    "HNR38",
    "RPDE",
    "DFA",
    "PPE",
    "GNE",
    "MFCC0",
    "MFCC1",
    "MFCC2",
    "MFCC3",
    "MFCC4",
    "MFCC5",
    "MFCC6",
    "MFCC7",
    "MFCC8",
    "MFCC9",
    "MFCC10",
    "MFCC11",
    "MFCC12",
    "Delta0",
    "Delta1",
    "Delta2",
    "Delta3",
    "Delta4",
    "Delta5",
    "Delta6",
    "Delta7",
    "Delta8",
    "Delta9",
    "Delta10",
    "Delta11",
    "Delta12",
)

NON_PREDICTOR_COLUMNS = (PARTICIPANT_COLUMN, RECORDING_COLUMN, TARGET_COLUMN, *METADATA_COLUMNS)
EXPECTED_COLUMNS = (*NON_PREDICTOR_COLUMNS, *FEATURE_COLUMNS)


class UCI489ValidationError(ValueError):
    """Raised when UCI dataset 489 violates its frozen input contract."""


@dataclass(frozen=True)
class UCI489Data:
    """Validated canonical manifest and model-ready feature rows."""

    manifest: pd.DataFrame
    model_rows: pd.DataFrame
    feature_names: tuple[str, ...]
    summary: dict[str, object]
    source_fingerprint: str


def _opaque_key(kind: str, *parts: object, length: int) -> str:
    payload = ":".join(["uci489", kind, *(str(part) for part in parts)])
    return f"{kind[0]}_{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:length]}"


def load_uci489(path: str | Path) -> UCI489Data:
    """Load, validate, and normalize the released UCI 489 CSV."""

    source_path = Path(path)
    source_fingerprint = sha256_file(source_path)
    raw = pd.read_csv(source_path)

    missing_columns = sorted(set(EXPECTED_COLUMNS) - set(raw.columns))
    unexpected_columns = sorted(set(raw.columns) - set(EXPECTED_COLUMNS))
    if missing_columns or unexpected_columns:
        raise UCI489ValidationError(
            f"source schema mismatch; missing={missing_columns}, unexpected={unexpected_columns}"
        )
    if raw.empty:
        raise UCI489ValidationError("source table is empty")

    participant_ids = raw[PARTICIPANT_COLUMN].astype("string").str.strip()
    if participant_ids.isna().any() or participant_ids.eq("").any():
        raise UCI489ValidationError("participant IDs must be non-empty")

    recordings = pd.to_numeric(raw[RECORDING_COLUMN], errors="coerce")
    if recordings.isna().any() or (recordings <= 0).any() or (recordings % 1 != 0).any():
        raise UCI489ValidationError("recording identifiers must be positive integers")
    recordings = recordings.astype(int)

    labels = pd.to_numeric(raw[TARGET_COLUMN], errors="coerce")
    if labels.isna().any() or not set(labels).issubset({0, 1}):
        raise UCI489ValidationError("Status must contain only binary labels 0 and 1")
    labels = labels.astype(int)

    duplicates = pd.DataFrame({"participant": participant_ids, "recording": recordings}).duplicated()
    if duplicates.any():
        raise UCI489ValidationError("duplicate participant/recording rows are not allowed")

    participant_labels = pd.DataFrame({"participant": participant_ids, "label": labels})
    inconsistent = participant_labels.groupby("participant")["label"].nunique()
    if (inconsistent != 1).any():
        raise UCI489ValidationError("target labels are inconsistent within a participant")

    numeric_features = raw.loc[:, FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    introduced_missing = numeric_features.isna() & raw.loc[:, FEATURE_COLUMNS].notna()
    if introduced_missing.any().any():
        bad_columns = introduced_missing.any()[lambda values: values].index.tolist()
        raise UCI489ValidationError(f"nonnumeric predictor values in columns: {bad_columns}")
    if np.isinf(numeric_features.to_numpy(dtype=float)).any():
        raise UCI489ValidationError("predictors cannot contain infinite values")

    participant_keys = participant_ids.map(
        lambda value: _opaque_key("participant", value, length=20)
    )
    sample_keys = [
        _opaque_key("sample", participant, recording, length=24)
        for participant, recording in zip(participant_ids, recordings, strict=True)
    ]
    row_numbers = range(2, len(raw) + 2)
    gender_available = raw["Gender"].notna()

    manifest = pd.DataFrame(
        {
            "sample_key": sample_keys,
            "participant_key": participant_keys,
            "label": labels,
            "label_source": "uci489:Status",
            "source_kind": "feature_row",
            "source_locator": [f"uci489:csv_row:{number}" for number in row_numbers],
            "task_type": "sustained_vowel_a",
            "duration_seconds": pd.NA,
            "voiced_seconds": pd.NA,
            "eligible": True,
            "exclusion_code": pd.NA,
            "metadata_available": [
                json.dumps({"gender": bool(available)}, sort_keys=True)
                for available in gender_available
            ],
            "source_fingerprint": source_fingerprint,
        },
        columns=MANIFEST_COLUMNS,
    )
    manifest = validate_manifest(manifest)

    model_rows = pd.concat(
        [
            manifest.loc[:, ["sample_key", "participant_key", "label"]].reset_index(drop=True),
            numeric_features.reset_index(drop=True),
        ],
        axis=1,
    )
    participant_table = model_rows.loc[:, ["participant_key", "label"]].drop_duplicates()
    class_counts = participant_table["label"].value_counts().sort_index()
    summary: dict[str, object] = {
        "rows": int(len(model_rows)),
        "participants": int(model_rows["participant_key"].nunique()),
        "participant_class_counts": {
            str(label): int(count) for label, count in class_counts.items()
        },
        "row_class_counts": {
            str(label): int(count)
            for label, count in model_rows["label"].value_counts().sort_index().items()
        },
        "recordings_per_participant": {
            "minimum": int(model_rows.groupby("participant_key").size().min()),
            "maximum": int(model_rows.groupby("participant_key").size().max()),
        },
        "feature_count": len(FEATURE_COLUMNS),
        "missing_feature_cells": int(numeric_features.isna().sum().sum()),
        "features_with_missing_values": numeric_features.columns[
            numeric_features.isna().any()
        ].tolist(),
        "duplicate_sample_keys": int(manifest["sample_key"].duplicated().sum()),
    }

    return UCI489Data(
        manifest=manifest,
        model_rows=model_rows,
        feature_names=FEATURE_COLUMNS,
        summary=summary,
        source_fingerprint=source_fingerprint,
    )
