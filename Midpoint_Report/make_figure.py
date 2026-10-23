"""Draw waveform + log-Mel spectrogram of one cleaned /a/ clip per group -> Midpoint_Report/clips.pdf."""

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

v = pd.read_csv("data/derived/vowels.csv")
clips = v[(v.task == "VA") & (v.rep == 1) & (v.sex == "F") & (v.source_sr == 16000)].groupby("label").first()

fig, axes = plt.subplots(2, 2, figsize=(7, 3.2), sharex=True, height_ratios=[1, 2])
for col, (label, title) in enumerate([(0, "Healthy control"), (1, "PD")]):
    y, sr = librosa.load(clips.loc[label, "clean_path"], sr=None)
    t = np.arange(len(y)) / sr
    axes[0, col].plot(t, y, lw=0.3, color="k")
    axes[0, col].set_title(f"{title}, female, /a/", fontsize=9)
    axes[0, col].set_yticks([])
    mel = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64), ref=np.max)
    librosa.display.specshow(mel, sr=sr, x_axis="time", y_axis="mel", ax=axes[1, col], cmap="magma")
    axes[1, col].set_xlabel("Time (s)", fontsize=8)
    axes[1, col].tick_params(labelsize=7)
    axes[1, col].yaxis.label.set_size(8)
fig.tight_layout()
fig.savefig("Midpoint_Report/clips.pdf")
