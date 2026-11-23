from __future__ import annotations

import numpy as np
import pytest
import torch
from transformers import Wav2Vec2Config, Wav2Vec2Model

from parkinsons_voice.features import embeddings


def tiny_encoder() -> Wav2Vec2Model:
    """Randomly initialized encoder with the wav2vec 2.0 layout; nothing is downloaded."""

    torch.manual_seed(0)
    config = Wav2Vec2Config(
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        conv_dim=(8, 8),
        conv_stride=(5, 4),
        conv_kernel=(10, 8),
        num_feat_extract_layers=2,
        num_conv_pos_embeddings=4,
        num_conv_pos_embedding_groups=2,
    )
    return Wav2Vec2Model(config)


def test_freeze_leaves_no_trainable_parameters() -> None:
    encoder = embeddings.freeze(tiny_encoder())

    embeddings.assert_frozen(encoder)
    assert not any(p.requires_grad for p in encoder.parameters())


def test_embedding_refuses_an_unfrozen_encoder() -> None:
    with pytest.raises(AssertionError, match="no trainable parameters"):
        embeddings.embed(tiny_encoder(), np.zeros(4000, dtype=np.float32))


def test_embedding_has_one_vector_per_clip_and_ignores_gain() -> None:
    encoder = embeddings.freeze(tiny_encoder())
    samples = np.random.default_rng(0).normal(size=8000).astype(np.float32)

    quiet = embeddings.embed(encoder, samples * 0.1)
    loud = embeddings.embed(encoder, samples)

    assert quiet.shape == (16,)
    np.testing.assert_allclose(quiet, loud, atol=1e-4)


def test_masked_mean_ignores_padded_frames() -> None:
    hidden = torch.tensor([[[1.0], [3.0], [100.0]]])
    mask = torch.tensor([[1, 1, 0]])

    assert embeddings.masked_mean(hidden, mask).item() == pytest.approx(2.0)
