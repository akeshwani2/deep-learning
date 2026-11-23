"""Log-Mel spectrogram front end for the CNN.

Each clip is standardized with its own mean and standard deviation, so no statistic is
fitted across clips and nothing can leak from held-out participants.
"""

from __future__ import annotations

import librosa
import torch
from torch import nn

SAMPLE_RATE = 16000
N_FFT = 400  # 25 ms window
HOP_LENGTH = 160  # 10 ms hop
N_MELS = 64
F_MIN = 50.0
F_MAX = 8000.0


class LogMel(nn.Module):
    """Waveforms (batch, samples) -> standardized log-Mel tensors (batch, 1, N_MELS, frames)."""

    def __init__(self) -> None:
        super().__init__()
        filters = librosa.filters.mel(
            sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS, fmin=F_MIN, fmax=F_MAX
        )
        self.register_buffer("filters", torch.from_numpy(filters))
        self.register_buffer("window", torch.hann_window(N_FFT))

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        spectrum = torch.stft(waveforms, N_FFT, HOP_LENGTH, window=self.window, return_complex=True)
        mel = self.filters @ spectrum.abs().pow(2)
        log_mel = torch.log(mel + 1e-10)
        mean = log_mel.mean(dim=(1, 2), keepdim=True)
        std = log_mel.std(dim=(1, 2), keepdim=True)
        return ((log_mel - mean) / (std + 1e-6)).unsqueeze(1)
