"""
Signal Filtering and Preprocessing Pipeline for AuraPulse.
Implements zero-phase forward-backward Butterworth IIR filtering, temporal detrending,
illumination normalization, outlier suppression, and windowing.
"""

from __future__ import annotations
from typing import Optional, Tuple
import numpy as np
from scipy import signal


class SignalPreprocessor:
    """
    State-of-the-art preprocessing for raw rPPG RGB time series.
    """

    @staticmethod
    def normalize_color_traces(rgb_series: np.ndarray) -> np.ndarray:
        """
        Illumination normalization: C_norm(t) = (C(t) - mean(C)) / mean(C)
        Input shape: (N, 3) where columns are [R, G, B].
        """
        if rgb_series.shape[0] < 2:
            return rgb_series

        means = np.mean(rgb_series, axis=0)
        # Avoid division by zero
        means_safe = np.where(np.abs(means) < 1e-6, 1.0, means)
        norm_rgb = (rgb_series - means) / means_safe
        return norm_rgb

    @staticmethod
    def detrend(data: np.ndarray, method: str = 'smoothness_priors', lambda_param: float = 100.0) -> np.ndarray:
        """
        Detrending using standard SciPy detrend or moving average subtract.
        """
        if data.size < 4:
            return data

        # Linear detrending as fast edge baseline
        return signal.detrend(data, type='linear')

    @staticmethod
    def butterworth_bandpass(
        data: np.ndarray,
        lowcut: float = 0.7,   # 42 BPM
        highcut: float = 3.5,  # 210 BPM
        fs: float = 30.0,
        order: int = 3
    ) -> np.ndarray:
        """
        Zero-phase forward-backward Butterworth bandpass filter.
        """
        n_samples = len(data)
        if n_samples < 15:
            return data

        nyquist = 0.5 * fs
        # Guard against cutoffs outside Nyquist
        low = max(0.01, lowcut / nyquist)
        high = min(0.99, highcut / nyquist)

        if low >= high:
            return data

        # Compute Butterworth filter coefficients
        b, a = signal.butter(order, [low, high], btype='band')

        # Use filtfilt for zero phase distortion
        # Pad len check: filtfilt requires n_samples > 3 * max(len(a), len(b))
        padlen = min(n_samples - 1, 3 * (max(len(a), len(b))))
        if n_samples > padlen:
            filtered = signal.filtfilt(b, a, data, padlen=padlen)
        else:
            filtered = signal.lfilter(b, a, data)

        return filtered

    @staticmethod
    def remove_motion_spikes(data: np.ndarray, window_size: int = 5, n_sigmas: float = 3.0) -> np.ndarray:
        """
        Hampel-style outlier rejection for motion spike suppression.
        """
        if len(data) < window_size:
            return data

        filtered = np.copy(data)
        k = 1.4826  # Scale factor for Gaussian distribution
        rolling_median = signal.medfilt(data, kernel_size=window_size if window_size % 2 == 1 else window_size + 1)
        difference = np.abs(data - rolling_median)
        median_abs_deviation = k * signal.medfilt(difference, kernel_size=window_size if window_size % 2 == 1 else window_size + 1)

        threshold = n_sigmas * median_abs_deviation
        outlier_idx = np.where(difference > threshold)[0]

        filtered[outlier_idx] = rolling_median[outlier_idx]
        return filtered

    @staticmethod
    def apply_window(data: np.ndarray, window_type: str = 'hamming') -> np.ndarray:
        """Applies a tapering window (Hamming / Hanning) for spectral analysis."""
        if len(data) < 4:
            return data
        if window_type == 'hanning':
            win = np.hanning(len(data))
        else:
            win = np.hamming(len(data))
        return data * win
