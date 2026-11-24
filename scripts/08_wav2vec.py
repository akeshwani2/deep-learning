"""Frozen wav2vec 2.0 embeddings + logistic regression -> artifacts/wav2vec/.

The encoder is never trained. Each clip becomes one mean-pooled embedding, cached in
data/derived/, and the linear head is evaluated with the same nested person-level CV as the
acoustic baseline. The same pipeline is also run on the non-speech background of each raw
recording as a recording-condition control.
"""

import json
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import transformers

from parkinsons_voice.audio.background import background_samples
from parkinsons_voice.evaluation.participant import (
    THRESHOLD,
    aggregate_participants,
    bootstrap_intervals,
    participant_metrics,
)
from parkinsons_voice.features import embeddings
from parkinsons_voice.models.logistic import C_GRID, nested_cv_predict

VOWELS = Path("data/derived/vowels.csv")
CACHE = Path("data/derived/wav2vec2_embeddings.npz")
OUT_DIR = Path("artifacts/wav2vec")
CACHE_KEY = {
    "checkpoint": embeddings.CHECKPOINT,
    "revision": embeddings.REVISION,
    "transformers": transformers.__version__,
    "pooling": "mean over all frames of the last hidden state",
    "preprocessing": "03_clean_vowels.py 3 s 16 kHz clips; background = audio.background",
}


def load_or_compute(clips: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Embeddings of every cleaned clip and of every raw recording's background (NaN if none)."""

    key = json.dumps(CACHE_KEY, sort_keys=True)
    if CACHE.exists():
        cached = np.load(CACHE)
        if str(cached["key"]) == key and list(cached["paths"]) == list(clips["clean_path"]):
            return cached["voice"], cached["background"]
    encoder = embeddings.load_frozen_encoder()
    voice, background = [], []
    for row in clips.itertuples():
        voice.append(embeddings.embed(encoder, sf.read(row.clean_path, dtype="float32")[0]))
        raw, _ = librosa.load(row.path, sr=embeddings.SAMPLE_RATE)
        quiet = background_samples(raw, embeddings.SAMPLE_RATE)
        background.append(
            np.full_like(voice[-1], np.nan) if quiet is None else embeddings.embed(encoder, quiet)
        )
    voice, background = np.stack(voice), np.stack(background)
    np.savez(
        CACHE, key=key, paths=clips["clean_path"].to_numpy(), voice=voice, background=background
    )
    return voice, background


def evaluate(clips: pd.DataFrame, vectors: np.ndarray, out_dir: Path, model: str) -> None:
    names = [f"e{i}" for i in range(vectors.shape[1])]
    table = pd.concat([clips.reset_index(drop=True), pd.DataFrame(vectors, columns=names)], axis=1)
    scored, folds = nested_cv_predict(table, names)
    people = aggregate_participants(scored)
    metrics = participant_metrics(people)
    intervals = bootstrap_intervals(people)

    out_dir.mkdir(parents=True, exist_ok=True)
    scored[["clean_path", "person_id", "label", "fold", "score"]].to_csv(
        out_dir / "clip_predictions.csv", index=False
    )
    people.to_csv(out_dir / "participant_predictions.csv", index=False)
    summary = {
        "model": model,
        "encoder": CACHE_KEY,
        "embedding_dim": vectors.shape[1],
        "participants": len(people),
        "clips": len(scored),
        "threshold": THRESHOLD,
        "c_grid": C_GRID,
        "chosen_c_by_fold": {f.fold: f.chosen_c for f in folds},
        "metrics": metrics,
        "ci95": intervals,
    }
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(f"{model}: {len(people)} people, {len(scored)} clips -> {out_dir}")
    for name, value in metrics.items():
        low, high = intervals[name]
        print(f"  {name:18s} {value:.3f}  (95% CI {low:.3f}-{high:.3f})")


clips = pd.read_csv(VOWELS)
voice, background = load_or_compute(clips)
evaluate(clips, voice, OUT_DIR, "wav2vec2_linear")
has_background = ~np.isnan(background).any(axis=1)
evaluate(
    clips[has_background],
    background[has_background],
    OUT_DIR / "background_only",
    "wav2vec2_linear_background_only",
)
