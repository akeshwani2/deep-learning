"""Compact log-Mel CNN -> artifacts/cnn/.

Each person is scored by a network trained on the other four folds. Inside those folds a
quarter of the people are held back to pick the stopping epoch; gain and noise augmentation
touch training batches only. Clip scores are averaged per person before any metric.
"""

import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch

from parkinsons_voice.evaluation.participant import (
    THRESHOLD,
    aggregate_participants,
    bootstrap_intervals,
    participant_metrics,
)
from parkinsons_voice.features import logmel
from parkinsons_voice.models import cnn

VOWELS = Path("data/derived/vowels.csv")
OUT_DIR = Path("artifacts/cnn")

torch.set_num_threads(8)
clips = pd.read_csv(VOWELS)
waveforms = torch.from_numpy(
    np.stack([sf.read(path, dtype="float32")[0] for path in clips["clean_path"]])
)
scored, folds = cnn.cv_predict(clips, waveforms)
people = aggregate_participants(scored)
metrics = participant_metrics(people)
intervals = bootstrap_intervals(people)

OUT_DIR.mkdir(parents=True, exist_ok=True)
scored[["clean_path", "person_id", "label", "fold", "score"]].to_csv(
    OUT_DIR / "clip_predictions.csv", index=False
)
people.to_csv(OUT_DIR / "participant_predictions.csv", index=False)
for result in folds:
    torch.save(result.state, OUT_DIR / f"fold{result.fold}.pt")
summary = {
    "model": "logmel_cnn",
    "participants": len(people),
    "clips": len(scored),
    "threshold": THRESHOLD,
    "parameters": sum(p.numel() for p in cnn.CompactCNN().parameters()),
    "log_mel": {
        "sample_rate": logmel.SAMPLE_RATE,
        "n_fft": logmel.N_FFT,
        "hop_length": logmel.HOP_LENGTH,
        "n_mels": logmel.N_MELS,
        "f_min": logmel.F_MIN,
        "f_max": logmel.F_MAX,
        "normalization": "per-clip mean and standard deviation",
    },
    "training": {
        "channels": cnn.CHANNELS,
        "dropout": cnn.DROPOUT,
        "optimizer": "AdamW",
        "learning_rate": cnn.LEARNING_RATE,
        "weight_decay": cnn.WEIGHT_DECAY,
        "batch_size": cnn.BATCH_SIZE,
        "max_epochs": cnn.MAX_EPOCHS,
        "patience": cnn.PATIENCE,
        "augmentation": {"gain_db": cnn.GAIN_DB, "snr_db": cnn.SNR_DB},
        "seed": cnn.SEED,
        "hardware": f"CPU, {torch.get_num_threads()} threads, {platform.machine()}",
    },
    "folds": [
        {
            "fold": r.fold,
            "selected_epoch": r.selected_epoch,
            "epochs_run": len(r.train_loss),
            "seconds": round(r.seconds, 1),
            "train_loss": r.train_loss,
            "validation_auroc": r.validation_auroc,
        }
        for r in folds
    ],
    "metrics": metrics,
    "ci95": intervals,
}
(OUT_DIR / "metrics.json").write_text(json.dumps(summary, indent=2))

print(f"{len(people)} people, {len(scored)} clips -> {OUT_DIR}")
for r in folds:
    print(f"fold {r.fold}: epoch {r.selected_epoch} of {len(r.train_loss)}, {r.seconds:.0f} s")
for name, value in metrics.items():
    low, high = intervals[name]
    print(f"{name:18s} {value:.3f}  (95% CI {low:.3f}-{high:.3f})")
