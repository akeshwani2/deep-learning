"""Compact log-Mel CNN evaluated with participant-grouped cross-validation.

For each outer fold, a quarter of the training participants are set aside as a validation
group. The network trains on the rest and the epoch with the best validation participant
AUROC is the one that scores the held-out fold. Gain and noise augmentation are applied to
training batches only.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn

from parkinsons_voice.evaluation.participant import aggregate_participants, participant_metrics
from parkinsons_voice.features.logmel import LogMel

CHANNELS = (16, 32, 64)
DROPOUT = 0.3
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 32
MAX_EPOCHS = 40
PATIENCE = 8
INNER_SPLITS = 4
GAIN_DB = 6.0  # training gain drawn uniformly from +-GAIN_DB
SNR_DB = (20.0, 40.0)  # training noise level drawn uniformly from this range
SEED = 0


class CompactCNN(nn.Module):
    """Three conv blocks, global average pooling, dropout, one binary logit."""

    def __init__(self) -> None:
        super().__init__()
        self.front_end = LogMel()
        blocks = []
        previous = 1
        for width in CHANNELS:
            blocks += [
                nn.Conv2d(previous, width, kernel_size=3, padding=1),
                nn.BatchNorm2d(width),
                nn.ReLU(),
                nn.MaxPool2d(2),
            ]
            previous = width
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Sequential(nn.Dropout(DROPOUT), nn.Linear(previous, 1))

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        features = self.blocks(self.front_end(waveforms))
        return self.head(features.mean(dim=(2, 3))).squeeze(1)


def augment(waveforms: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """Random gain and additive white noise, one draw per clip."""

    count = waveforms.shape[0]
    gain_db = (torch.rand(count, 1, generator=generator) * 2 - 1) * GAIN_DB
    snr_db = SNR_DB[0] + torch.rand(count, 1, generator=generator) * (SNR_DB[1] - SNR_DB[0])
    signal_rms = waveforms.pow(2).mean(dim=1, keepdim=True).sqrt()
    noise = torch.randn(waveforms.shape, generator=generator) * signal_rms / 10 ** (snr_db / 20)
    return (waveforms + noise) * 10 ** (gain_db / 20)


@torch.no_grad()
def predict(model: CompactCNN, waveforms: torch.Tensor) -> np.ndarray:
    """PD-class scores for each clip, with dropout and augmentation off."""

    model.eval()
    logits = [model(waveforms[i : i + 128]) for i in range(0, len(waveforms), 128)]
    return torch.sigmoid(torch.cat(logits)).numpy()


@dataclass(frozen=True)
class CnnFoldResult:
    fold: int
    selected_epoch: int
    validation_auroc: list[float]
    train_loss: list[float]
    seconds: float
    state: dict


def _participant_auroc(clips: pd.DataFrame, scores: np.ndarray) -> float:
    return participant_metrics(aggregate_participants(clips.assign(score=scores)))["auroc"]


def train_fold(
    clips: pd.DataFrame, waveforms: torch.Tensor, fold: int, max_epochs: int = MAX_EPOCHS
) -> tuple[CompactCNN, CnnFoldResult]:
    """Train on ``clips`` (the outer training participants) with early stopping."""

    started = time.perf_counter()
    torch.manual_seed(SEED + fold)
    generator = torch.Generator().manual_seed(SEED + fold)
    splitter = StratifiedGroupKFold(INNER_SPLITS, shuffle=True, random_state=SEED)
    fit_idx, val_idx = next(splitter.split(clips, clips["label"], clips["person_id"]))
    fit_x, val_x = waveforms[fit_idx], waveforms[val_idx]
    fit_y = torch.tensor(clips["label"].to_numpy()[fit_idx], dtype=torch.float32)
    validation = clips.iloc[val_idx]

    model = CompactCNN()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    # Weight the PD class so both classes contribute equally to the loss.
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=(1 - fit_y).sum() / fit_y.sum())
    best_auroc, best_epoch, best_state = -1.0, 0, None
    aurocs, losses = [], []
    for epoch in range(1, max_epochs + 1):
        model.train()
        order = torch.randperm(len(fit_x), generator=generator)
        total = 0.0
        for start in range(0, len(order), BATCH_SIZE):
            batch = order[start : start + BATCH_SIZE]
            loss = loss_fn(model(augment(fit_x[batch], generator)), fit_y[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(batch)
        losses.append(total / len(order))
        aurocs.append(_participant_auroc(validation, predict(model, val_x)))
        if aurocs[-1] > best_auroc:
            best_auroc, best_epoch = aurocs[-1], epoch
            best_state = copy.deepcopy(model.state_dict())
        elif epoch - best_epoch >= PATIENCE:
            break
    model.load_state_dict(best_state)
    seconds = time.perf_counter() - started
    return model, CnnFoldResult(fold, best_epoch, aurocs, losses, seconds, best_state)


def cv_predict(
    clips: pd.DataFrame, waveforms: torch.Tensor, max_epochs: int = MAX_EPOCHS
) -> tuple[pd.DataFrame, list[CnnFoldResult]]:
    """Score every clip with a network that never saw its participant.

    ``clips`` needs ``person_id``, ``label``, and ``fold``; row i is ``waveforms[i]``.
    """

    if (clips.groupby("person_id")["fold"].nunique() != 1).any():
        raise ValueError("a participant appears in more than one fold")

    scored = clips.reset_index(drop=True).copy()
    scored["score"] = np.nan
    results = []
    for fold in sorted(scored["fold"].unique()):
        held_out = (scored["fold"] == fold).to_numpy()
        model, result = train_fold(scored[~held_out], waveforms[~held_out], int(fold), max_epochs)
        scored.loc[held_out, "score"] = predict(model, waveforms[held_out])
        results.append(result)
    return scored, results
