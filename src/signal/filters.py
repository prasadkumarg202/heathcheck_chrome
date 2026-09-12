import numpy as np
from core.filters.signal_filters import SignalPreprocessor

def butter_bandpass_filter(data: np.ndarray, lowcut: float = 0.7, highcut: float = 3.5, fs: float = 30.0, order: int = 3) -> np.ndarray:
    return SignalPreprocessor.butterworth_bandpass(data, lowcut=lowcut, highcut=highcut, fs=fs, order=order)

def detrend_signal(data: np.ndarray, method: str = 'linear') -> np.ndarray:
    return SignalPreprocessor.detrend(data, method=method)

def normalize_color_signals(rgb_series: np.ndarray) -> np.ndarray:
    return SignalPreprocessor.normalize_color_traces(rgb_series)

def remove_motion_spikes(data: np.ndarray, window_size: int = 5, n_sigmas: float = 3.0) -> np.ndarray:
    return SignalPreprocessor.remove_motion_spikes(data, window_size=window_size, n_sigmas=n_sigmas)

def apply_window(data: np.ndarray, window_type: str = 'hamming') -> np.ndarray:
    return SignalPreprocessor.apply_window(data, window_type=window_type)

__all__ = [
    'SignalPreprocessor',
    'butter_bandpass_filter',
    'detrend_signal',
    'normalize_color_signals',
    'remove_motion_spikes',
    'apply_window'
]
