import numpy as np
import pandas as pd
import pytest

from parkinsons_voice.evaluation.participant import aggregate_participants, participant_metrics
from parkinsons_voice.models.logistic import nested_cv_predict

FEATURES = ["x0", "x1"]


def _recordings(seed: int = 0) -> pd.DataFrame:
    """40 people in 5 folds, 3 recordings each; x0 separates the classes, x1 is noise."""

    rng = np.random.default_rng(seed)
    rows = []
    for person in range(40):
        label = person % 2
        for _ in range(3):
            rows.append(
                {
                    "person_id": f"p{person:02d}",
                    "label": label,
                    "fold": (person // 2) % 5,
                    "x0": rng.normal(2.0 * label, 1.0),
                    "x1": rng.normal(),
                }
            )
    frame = pd.DataFrame(rows)
    frame.loc[5, "x1"] = np.nan  # exercises the imputer
    return frame


def test_every_recording_is_scored_and_signal_is_found():
    scored, folds = nested_cv_predict(_recordings(), FEATURES)

    assert scored["score"].between(0, 1).all()
    assert [f.fold for f in folds] == [0, 1, 2, 3, 4]
    assert participant_metrics(aggregate_participants(scored))["auroc"] > 0.85


def test_held_out_rows_do_not_affect_what_is_fitted():
    recordings = _recordings()
    shifted = recordings.copy()
    held_out = shifted["fold"] == 2
    shifted.loc[held_out, FEATURES] += 1e6
    shifted.loc[held_out, "label"] = 1 - shifted.loc[held_out, "label"]

    _, original = nested_cv_predict(recordings, FEATURES)
    _, changed = nested_cv_predict(shifted, FEATURES)

    before, after = original[2], changed[2]
    assert after.chosen_c == before.chosen_c
    assert after.inner_auroc == before.inner_auroc
    for step in ("simpleimputer", "standardscaler", "logisticregression"):
        fitted_before, fitted_after = before.pipeline[step], after.pipeline[step]
        for attribute in ("statistics_", "mean_", "scale_", "coef_", "intercept_"):
            if hasattr(fitted_before, attribute):
                np.testing.assert_array_equal(
                    getattr(fitted_after, attribute), getattr(fitted_before, attribute)
                )
    # The shift must be visible where it should be: in models trained on fold 2.
    assert changed[0].pipeline["standardscaler"].mean_[0] > 1e4


def test_participant_split_across_folds_is_rejected():
    recordings = _recordings()
    recordings.loc[0, "fold"] = 4

    with pytest.raises(ValueError, match="more than one fold"):
        nested_cv_predict(recordings, FEATURES)
