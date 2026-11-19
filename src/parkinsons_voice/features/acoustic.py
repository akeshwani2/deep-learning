"""Handcrafted acoustic features for one cleaned vowel clip.

Pitch, jitter, shimmer, and harmonics-to-noise ratio come from Praat (via parselmouth);
MFCC summaries come from librosa. Every clip uses the same fixed pitch range, so no
participant metadata (such as sex) enters the features. Values that cannot be measured
are returned as NaN and counted in ``n_missing``; imputation is left to the model pipeline
so it can be fitted on training participants only.
"""

from __future__ import annotations

import librosa
import numpy as np
import parselmouth
from parselmouth.praat import call

PITCH_FLOOR_HZ = 75.0
PITCH_CEILING_HZ = 500.0
N_MFCC = 13

# Praat's standard perturbation settings: shortest/longest period, period and amplitude factors.
_PERIOD = (0.0001, 0.02, 1.3)
_AMPLITUDE_FACTOR = 1.6

JITTER_MEASURES = {
    "jitter_local": "Get jitter (local)",
    "jitter_local_abs": "Get jitter (local, absolute)",
    "jitter_rap": "Get jitter (rap)",
    "jitter_ppq5": "Get jitter (ppq5)",
}
SHIMMER_MEASURES = {
    "shimmer_local": "Get shimmer (local)",
    "shimmer_local_db": "Get shimmer (local_dB)",
    "shimmer_apq3": "Get shimmer (apq3)",
    "shimmer_apq5": "Get shimmer (apq5)",
    "shimmer_apq11": "Get shimmer (apq11)",
}

FEATURE_NAMES: tuple[str, ...] = (
    "f0_mean_hz",
    "f0_std_hz",
    "voiced_fraction",
    *JITTER_MEASURES,
    *SHIMMER_MEASURES,
    "hnr_mean_db",
    *(f"mfcc{i}_mean" for i in range(N_MFCC)),
    *(f"mfcc{i}_std" for i in range(N_MFCC)),
)


def _praat_features(samples: np.ndarray, sample_rate: int) -> dict[str, float]:
    sound = parselmouth.Sound(samples.astype(np.float64), sampling_frequency=sample_rate)
    pitch = sound.to_pitch(pitch_floor=PITCH_FLOOR_HZ, pitch_ceiling=PITCH_CEILING_HZ)
    f0 = pitch.selected_array["frequency"]
    voiced = f0[f0 > 0]
    features = {"voiced_fraction": float(len(voiced) / len(f0)) if len(f0) else 0.0}
    if len(voiced):
        features["f0_mean_hz"] = float(voiced.mean())
        features["f0_std_hz"] = float(voiced.std())

    pulses = call(sound, "To PointProcess (periodic, cc)", PITCH_FLOOR_HZ, PITCH_CEILING_HZ)
    for name, command in JITTER_MEASURES.items():
        features[name] = call(pulses, command, 0, 0, *_PERIOD)
    for name, command in SHIMMER_MEASURES.items():
        features[name] = call([sound, pulses], command, 0, 0, *_PERIOD, _AMPLITUDE_FACTOR)

    harmonicity = sound.to_harmonicity_cc(minimum_pitch=PITCH_FLOOR_HZ)
    features["hnr_mean_db"] = call(harmonicity, "Get mean", 0, 0)
    return features


def _mfcc_features(samples: np.ndarray, sample_rate: int) -> dict[str, float]:
    mfcc = librosa.feature.mfcc(y=samples.astype(np.float32), sr=sample_rate, n_mfcc=N_MFCC)
    features = {f"mfcc{i}_mean": float(mfcc[i].mean()) for i in range(N_MFCC)}
    features.update({f"mfcc{i}_std": float(mfcc[i].std()) for i in range(N_MFCC)})
    return features


def extract_acoustic_features(samples: np.ndarray, sample_rate: int) -> dict[str, float]:
    """Return every name in ``FEATURE_NAMES`` plus ``n_missing`` for one mono clip."""

    measured: dict[str, float] = {}
    for extractor in (_praat_features, _mfcc_features):
        try:
            measured.update(extractor(samples, sample_rate))
        except (parselmouth.PraatError, librosa.util.exceptions.ParameterError):
            continue  # this extractor's features stay missing and are counted below

    features = {name: float(measured.get(name, np.nan)) for name in FEATURE_NAMES}
    for name, value in features.items():
        if not np.isfinite(value):
            features[name] = np.nan
    features["n_missing"] = int(sum(np.isnan(value) for value in features.values()))
    return features
