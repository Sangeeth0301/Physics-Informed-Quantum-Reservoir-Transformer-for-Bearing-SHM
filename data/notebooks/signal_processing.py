# ============================================================
# signal_processing.py
# Research-grade bearing preprocessing (OPTIMAL)
# ============================================================

import numpy as np
from scipy.signal import butter, filtfilt, hilbert


# ============================================================
# BANDPASS FILTER (CWRU STANDARD)
# ============================================================

def bandpass_filter(signal, fs=12000, lowcut=2000, highcut=6000, order=4):
    """
    Robust bandpass filter for bearing signals.
    """

    signal = np.asarray(signal, dtype=np.float64)

    nyquist = 0.5 * fs

    # ---- safety clipping ----
    lowcut = max(1.0, lowcut)
    highcut = min(highcut, nyquist - 1.0)

    low = lowcut / nyquist
    high = highcut / nyquist

    if low >= high:
        raise ValueError("Invalid bandpass frequencies")

    b, a = butter(order, [low, high], btype='band')
    filtered = filtfilt(b, a, signal)

    return filtered


# ============================================================
# ENVELOPE EXTRACTION
# ============================================================

def envelope_signal(signal):
    """Hilbert envelope detection"""
    analytic_signal = hilbert(signal)
    envelope = np.abs(analytic_signal)
    return envelope


# ============================================================
# FULL PIPELINE (RECOMMENDED)
# ============================================================

def preprocess_bearing_signal(signal, fs=12000, normalize=True):
    """
    Full research-grade preprocessing.

    raw → bandpass → envelope → mean removal → normalization
    """

    # ---- bandpass ----
    filtered = bandpass_filter(signal, fs=fs)

    # ---- envelope ----
    env = envelope_signal(filtered)

    # ---- mean removal ----
    env = env - np.mean(env)

    # ---- normalization (important for DMD stability) ----
    if normalize:
        std = np.std(env)
        if std > 1e-8:
            env = env / std

    return env