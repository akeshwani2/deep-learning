"""Non-speech background of a recording, used as a recording-condition control.

The background contains no voice, so a model that separates the groups from it alone is
reading the room, microphone, or recording procedure.
"""

from __future__ import annotations

import librosa
import numpy as np

TOP_DB = 30  # same silence threshold as scripts/03_clean_vowels.py
MARGIN_S = 0.1  # also dropped on each side of every voiced stretch
MIN_BACKGROUND_S = 0.25


def background_samples(samples: np.ndarray, sample_rate: int) -> np.ndarray | None:
    """Peak-scaled samples outside every voiced stretch, or None if there are too few."""

    margin = int(MARGIN_S * sample_rate)
    keep = np.ones(len(samples), dtype=bool)
    for start, end in librosa.effects.split(samples, top_db=TOP_DB):
        keep[max(0, start - margin) : end + margin] = False
    if keep.sum() < MIN_BACKGROUND_S * sample_rate:
        return None
    return samples[keep] / np.abs(samples).max()
