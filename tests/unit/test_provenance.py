from __future__ import annotations

import pandas as pd
import pytest

from parkinsons_voice.data.provenance import (
    ProvenanceMismatchError,
    fingerprint_frame,
    require_matching_provenance,
)


def test_frame_fingerprint_is_stable_to_row_order():
    frame = pd.DataFrame({"sample_key": ["b", "a"], "value": [2.0, 1.0]})
    reversed_frame = frame.iloc[::-1].reset_index(drop=True)

    assert fingerprint_frame(frame, sort_by=["sample_key"]) == fingerprint_frame(
        reversed_frame,
        sort_by=["sample_key"],
    )


def test_provenance_mismatch_is_fatal():
    with pytest.raises(ProvenanceMismatchError, match="source_fingerprint"):
        require_matching_provenance(
            {"source_fingerprint": "old"},
            {"source_fingerprint": "new"},
            ["source_fingerprint"],
        )
