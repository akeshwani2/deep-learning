"""Stable fingerprints and provenance compatibility checks."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import pandas as pd


class ProvenanceMismatchError(RuntimeError):
    """Raised when an artifact does not match required provenance."""


def sha256_file(path: str | Path) -> str:
    """Return a streaming SHA-256 digest for a file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint_json(value: Any) -> str:
    """Hash JSON-compatible data using a stable serialization."""

    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _canonical_cell(value: Any) -> str:
    if pd.isna(value):
        return "<NA>"
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return format(value, ".17g")
    return str(value)


def fingerprint_frame(
    frame: pd.DataFrame,
    *,
    sort_by: Iterable[str],
    columns: Iterable[str] | None = None,
) -> str:
    """Hash a table after canonical column selection and row sorting."""

    selected_columns = list(columns) if columns is not None else list(frame.columns)
    sort_columns = list(sort_by)
    missing = sorted((set(selected_columns) | set(sort_columns)) - set(frame.columns))
    if missing:
        raise ValueError(f"cannot fingerprint missing columns: {missing}")

    ordered = frame.sort_values(sort_columns, kind="mergesort").loc[:, selected_columns]
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(selected_columns)
    for row in ordered.itertuples(index=False, name=None):
        writer.writerow([_canonical_cell(value) for value in row])
    return hashlib.sha256(buffer.getvalue().encode("utf-8")).hexdigest()


def require_matching_provenance(
    actual: Mapping[str, Any],
    expected: Mapping[str, Any],
    keys: Iterable[str],
) -> None:
    """Reject an artifact when any required fingerprint differs."""

    mismatches = {
        key: {"actual": actual.get(key), "expected": expected.get(key)}
        for key in keys
        if actual.get(key) != expected.get(key)
    }
    if mismatches:
        raise ProvenanceMismatchError(f"incompatible artifact provenance: {mismatches}")
