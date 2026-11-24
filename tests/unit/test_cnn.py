from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from parkinsons_voice.features.logmel import HOP_LENGTH, N_MELS, LogMel
from parkinsons_voice.models import cnn

SAMPLES = 8000  # half a second keeps training fast


def synthetic_clips(people_per_class: int = 10, clips_per_person: int = 2):
    """Healthy people are a 150 Hz tone, PD people a 600 Hz tone, plus noise."""

    generator = torch.Generator().manual_seed(0)
    time = torch.arange(SAMPLES) / 16000
    rows, waves = [], []
    for label, frequency in ((0, 150.0), (1, 600.0)):
        for person in range(people_per_class):
            for _ in range(clips_per_person):
                rows.append({"person_id": f"p{label}{person}", "label": label, "fold": person % 5})
                noise = 0.05 * torch.randn(SAMPLES, generator=generator)
                waves.append(0.5 * torch.sin(2 * torch.pi * frequency * time) + noise)
    return pd.DataFrame(rows), torch.stack(waves)


def test_log_mel_shape_and_per_clip_standardization() -> None:
    waveforms = torch.randn(3, 16000) * torch.tensor([[0.01], [0.1], [1.0]])

    tensors = LogMel()(waveforms)

    assert tensors.shape == (3, 1, N_MELS, 16000 // HOP_LENGTH + 1)
    assert tensors.mean(dim=(1, 2, 3)).abs().max() < 1e-4
    assert torch.allclose(tensors.std(dim=(1, 2, 3)), torch.ones(3), atol=1e-3)


def test_cnn_returns_one_logit_per_clip() -> None:
    assert cnn.CompactCNN()(torch.randn(4, SAMPLES)).shape == (4,)


def test_prediction_is_deterministic_and_unaugmented() -> None:
    model = cnn.CompactCNN()
    waveforms = torch.randn(4, SAMPLES)

    first = cnn.predict(model, waveforms)
    second = cnn.predict(model, waveforms)

    assert not model.training
    np.testing.assert_array_equal(first, second)


def test_augmentation_changes_gain_and_adds_noise_within_declared_ranges() -> None:
    waveforms = torch.ones(200, SAMPLES) * 0.1

    augmented = cnn.augment(waveforms, torch.Generator().manual_seed(0))

    gain_db = 20 * torch.log10(augmented.mean(dim=1) / 0.1)
    assert gain_db.abs().max() <= cnn.GAIN_DB + 0.1
    assert gain_db.std() > 1.0
    noise_rms = (augmented / augmented.mean(dim=1, keepdim=True) - 1).std(dim=1)
    snr_db = -20 * torch.log10(noise_rms)
    assert snr_db.min() >= cnn.SNR_DB[0] - 0.5
    assert snr_db.max() <= cnn.SNR_DB[1] + 0.5


def test_cv_scores_every_clip_and_learns_an_obvious_signal() -> None:
    clips, waveforms = synthetic_clips()

    scored, folds = cnn.cv_predict(clips, waveforms, max_epochs=6)

    assert scored["score"].notna().all()
    assert len(folds) == 5
    assert all(1 <= result.selected_epoch <= 6 for result in folds)
    people = scored.groupby("person_id").agg(label=("label", "first"), score=("score", "mean"))
    assert (
        people.loc[people["label"] == 1, "score"].mean()
        > people.loc[people["label"] == 0, "score"].mean()
    )


def test_cv_rejects_a_participant_split_across_folds() -> None:
    clips, waveforms = synthetic_clips()
    clips.loc[0, "fold"] = 4

    with pytest.raises(ValueError, match="more than one fold"):
        cnn.cv_predict(clips, waveforms, max_epochs=1)
