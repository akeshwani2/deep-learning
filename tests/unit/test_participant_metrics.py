import pandas as pd
import pytest

from parkinsons_voice.evaluation.participant import (
    aggregate_participants,
    bootstrap_intervals,
    participant_metrics,
)


def _recordings() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "person_id": ["a", "a", "b", "c", "c", "d"],
            "label": [0, 0, 0, 1, 1, 1],
            "score": [0.2, 0.4, 0.7, 0.9, 0.5, 0.4],
        }
    )


def test_scores_are_averaged_per_participant():
    people = aggregate_participants(_recordings()).set_index("person_id")

    assert people["score"].to_dict() == pytest.approx({"a": 0.3, "b": 0.7, "c": 0.7, "d": 0.4})
    assert people["recording_count"].to_dict() == {"a": 2, "b": 1, "c": 2, "d": 1}


def test_metrics_match_hand_calculation():
    metrics = participant_metrics(aggregate_participants(_recordings()))

    # PD scores (0.7, 0.4) vs healthy (0.3, 0.7): 2 wins + 1 tie out of 4 pairs.
    assert metrics["auroc"] == pytest.approx(0.625)
    assert metrics["sensitivity"] == pytest.approx(0.5)
    assert metrics["specificity"] == pytest.approx(0.5)
    assert metrics["balanced_accuracy"] == pytest.approx(0.5)


def test_participant_with_two_labels_is_rejected():
    recordings = _recordings()
    recordings.loc[1, "label"] = 1

    with pytest.raises(ValueError, match="more than one label"):
        aggregate_participants(recordings)


def test_bootstrap_interval_contains_the_estimate():
    people = aggregate_participants(_recordings())
    low, high = bootstrap_intervals(people, replicates=200)["auroc"]

    assert low <= participant_metrics(people)["auroc"] <= high


def test_precision_and_f1_match_hand_calculation() -> None:
    participants = pd.DataFrame(
        {
            "person_id": list("abcde"),
            "label": [1, 1, 1, 0, 0],
            "score": [0.9, 0.8, 0.2, 0.7, 0.1],
        }
    )

    metrics = participant_metrics(participants)

    # 2 true positives, 1 false positive, 1 false negative.
    assert metrics["precision"] == pytest.approx(2 / 3)
    assert metrics["f1"] == pytest.approx(2 / 3)


def test_precision_and_f1_are_zero_without_predicted_positives() -> None:
    participants = pd.DataFrame({"person_id": list("ab"), "label": [1, 0], "score": [0.2, 0.1]})

    metrics = participant_metrics(participants)

    assert metrics["precision"] == 0.0
    assert metrics["f1"] == 0.0
