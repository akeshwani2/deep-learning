import numpy as np

from parkinsons_voice.features.acoustic import FEATURE_NAMES, extract_acoustic_features

SAMPLE_RATE = 16000


def _harmonic_tone(f0: float, seconds: float = 3.0) -> np.ndarray:
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    tone = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 11))
    return 0.9 * tone / np.abs(tone).max()


def test_steady_tone_gives_expected_pitch_and_complete_features():
    features = extract_acoustic_features(_harmonic_tone(150.0), SAMPLE_RATE)

    assert set(features) == {*FEATURE_NAMES, "n_missing"}
    assert features["n_missing"] == 0
    assert abs(features["f0_mean_hz"] - 150.0) < 1.0
    assert features["voiced_fraction"] > 0.95
    # A synthetic tone is perfectly periodic, unlike any real voice.
    assert features["jitter_local"] < 0.001
    assert features["hnr_mean_db"] > 30


def test_unvoiced_clip_is_flagged_instead_of_failing():
    features = extract_acoustic_features(np.zeros(3 * SAMPLE_RATE), SAMPLE_RATE)

    assert features["voiced_fraction"] == 0.0
    assert np.isnan(features["f0_mean_hz"])
    assert np.isnan(features["jitter_local"])
    assert features["n_missing"] > 0
