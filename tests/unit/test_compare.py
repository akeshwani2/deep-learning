from __future__ import annotations

import pandas as pd
import pytest

from parkinsons_voice.evaluation.compare import (
    brier_score,
    expected_calibration_error,
    paired_auroc_difference,
    reliability_table,
)


def people(scores: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"person_id": list("abcdef"), "label": [0, 0, 0, 1, 1, 1], "score": scores})


def test_brier_score_matches_hand_calculation() -> None:
    assert brier_score(people([0.0, 0.5, 0.0, 1.0, 0.5, 1.0])) == pytest.approx(0.5 / 6)


def test_reliability_table_and_calibration_error_match_hand_calculation() -> None:
    # Bin 0 holds 0.1, 0.1, 0.1 (no PD); bin 4 holds 0.9, 0.9, 0.9 (two of three PD).
    table = pd.DataFrame(
        {"person_id": list("abcdef"), "label": [0, 0, 0, 0, 1, 1], "score": [0.1] * 3 + [0.9] * 3}
    )

    bins = reliability_table(table)

    assert bins["bin"].tolist() == [0, 4]
    assert bins["pd_fraction"].tolist() == pytest.approx([0.0, 2 / 3])
    assert expected_calibration_error(table) == pytest.approx((0.1 + (0.9 - 2 / 3)) / 2)


def test_paired_difference_is_zero_for_identical_models() -> None:
    table = people([0.1, 0.6, 0.3, 0.4, 0.8, 0.9])

    result = paired_auroc_difference(table, table, replicates=200)

    assert result == {"difference": 0.0, "ci95_low": 0.0, "ci95_high": 0.0}


def test_paired_difference_favours_the_better_model() -> None:
    perfect = people([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    reversed_scores = people([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])

    result = paired_auroc_difference(perfect, reversed_scores, replicates=200)

    assert result["difference"] == pytest.approx(1.0)
    assert result["ci95_low"] == pytest.approx(1.0)


def test_paired_difference_rejects_different_participants() -> None:
    first = people([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])

    with pytest.raises(ValueError, match="same participants"):
        paired_auroc_difference(first, first.iloc[:-1], replicates=10)
