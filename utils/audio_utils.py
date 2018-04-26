import numpy as np
import librosa
from scipy import signal
import fnmatch
import os


def preemphasis(x, coeff=0.97):
    return signal.lfilter([1, -coeff], [1], x)


def inv_preemphasis(x, coeff=0.97):
    return signal.lfilter([1], [1, -coeff], x)


def griffin_lim(stft_matrix_,
                n_fft,
                hop_size,
                win_size=None,
                max_iter=50,
                delta=20):

    n_frames = stft_matrix_.shape[1]
    expected_len = n_fft + hop_size*(n_frames - 1)
    shape = (expected_len - n_fft,)
    y = np.random.random(shape)

    for i in range(max_iter):
        stft_matrix = librosa.core.stft(
                y,
                n_fft=n_fft,
                hop_length=hop_size,
                win_length=win_size)
        stft_matrix = stft_matrix_ * stft_matrix / np.abs(stft_matrix)
        y = librosa.core.istft(
                stft_matrix,
                hop_length=hop_size,
                win_length=win_size)
    return y


def log_magnitude_postproc(stftm, magnitude_enphasis):
    # emphasizing magnitude
    stftm = stftm * magnitude_enphasis
    # Undo log and square
    stftm = np.sqrt(np.exp(stftm))
    return stftm
