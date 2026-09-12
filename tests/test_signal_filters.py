"""
Unit tests for AuraPulse signal filters and preprocessors.
"""

import numpy as np
import pytest
from core.filters.signal_filters import SignalPreprocessor


def test_color_normalization():
    # Constant baseline plus small AC signal
    rgb = np.ones((100, 3)) * 150.0
    rgb[:, 1] += np.sin(np.linspace(0, 10, 100)) * 2.0  # Green channel AC modulation

    norm_rgb = SignalPreprocessor.normalize_color_traces(rgb)
    assert norm_rgb.shape == (100, 3)
    # Means of normalized traces should be approximately 0
    assert np.allclose(np.mean(norm_rgb, axis=0), 0.0, atol=1e-5)


def test_detrending():
    t = np.linspace(0, 10, 300)
    linear_trend = 2.5 * t + 50.0
    sine_wave = np.sin(2 * np.pi * 1.2 * t)
    corrupted = linear_trend + sine_wave

    detrended = SignalPreprocessor.detrend(corrupted)
    assert len(detrended) == len(corrupted)
    # Trend slope should be removed (mean close to 0)
    assert np.abs(np.mean(detrended)) < 0.1


def test_butterworth_bandpass():
    fs = 30.0
    t = np.linspace(0, 10, int(10 * fs), endpoint=False)
    
    # 1.2 Hz (72 BPM) - inside passband [0.7, 3.5] Hz
    f_in = np.sin(2 * np.pi * 1.2 * t)
    # 0.2 Hz (12 BPM) - out of passband
    f_low = np.sin(2 * np.pi * 0.2 * t)
    # 8.0 Hz - out of passband
    f_high = np.sin(2 * np.pi * 8.0 * t)

    combined = f_in + f_low + f_high
    filtered = SignalPreprocessor.butterworth_bandpass(combined, lowcut=0.7, highcut=3.5, fs=fs, order=3)

    # In-band component power should be preserved much more than out-of-band
    assert np.var(filtered) < np.var(combined)
    assert np.corrcoef(filtered[30:-30], f_in[30:-30])[0, 1] > 0.85


def test_remove_motion_spikes():
    data = np.sin(np.linspace(0, 10, 200))
    # Inject large motion spike
    data[50] = 50.0
    data[120] = -45.0

    cleaned = SignalPreprocessor.remove_motion_spikes(data, window_size=5, n_sigmas=3.0)
    assert np.abs(cleaned[50]) < 5.0
    assert np.abs(cleaned[120]) < 5.0
