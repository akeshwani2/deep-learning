"""Pooled embeddings from a frozen wav2vec 2.0 encoder."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

CHECKPOINT = "facebook/wav2vec2-base"  # self-supervised only, Apache-2.0
REVISION = "0b5b8e868dd84f03fd87d01f9c4ff0f080fecfe8"
SAMPLE_RATE = 16000


def load_frozen_encoder() -> nn.Module:
    from transformers import Wav2Vec2Model

    return freeze(Wav2Vec2Model.from_pretrained(CHECKPOINT, revision=REVISION))


def freeze(encoder: nn.Module) -> nn.Module:
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    return encoder


def assert_frozen(encoder: nn.Module) -> None:
    if encoder.training or any(p.requires_grad for p in encoder.parameters()):
        raise AssertionError("the encoder must be in eval mode with no trainable parameters")


def masked_mean(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean over time of ``hidden`` (batch, frames, dim), counting only frames where mask is 1."""

    weights = mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * weights).sum(dim=1) / weights.sum(dim=1)


@torch.no_grad()
def embed(encoder: nn.Module, samples: np.ndarray) -> np.ndarray:
    """One pooled embedding for one mono 16 kHz clip of any length."""

    assert_frozen(encoder)
    # The checkpoint expects zero-mean, unit-variance input.
    wave = torch.from_numpy((samples - samples.mean()) / (samples.std() + 1e-7)).float()
    hidden = encoder(wave.unsqueeze(0)).last_hidden_state
    # A single unpadded clip has no padded frames, so every frame counts.
    return masked_mean(hidden, torch.ones(hidden.shape[:2])).squeeze(0).numpy()
