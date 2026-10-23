"""Synthetic, non-sensitive test fixtures."""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd
import pytest

from parkinsons_voice.data.uci489 import FEATURE_COLUMNS


@pytest.fixture
def synthetic_uci_frame() -> Callable[..., pd.DataFrame]:
    """Return a factory for balanced UCI-shaped synthetic feature tables."""

    def factory(participants_per_class: int = 10, recordings: int = 3) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for label, prefix in ((0, "CONTROL"), (1, "PD")):
            for participant_number in range(participants_per_class):
                participant_id = f"{prefix}-{participant_number:03d}"
                for recording in range(1, recordings + 1):
                    row: dict[str, object] = {
                        "ID": participant_id,
                        "Recording": recording,
                        "Status": label,
                        "Gender": participant_number % 2,
                    }
                    for feature_number, feature in enumerate(FEATURE_COLUMNS):
                        row[feature] = (
                            label * 10
                            + participant_number * 0.1
                            + recording * 0.01
                            + feature_number * 0.001
                        )
                    rows.append(row)
        return pd.DataFrame(rows)

    return factory
