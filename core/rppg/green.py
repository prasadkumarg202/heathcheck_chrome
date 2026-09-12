"""
Green Channel Baseline and Normalized Color Difference rPPG Algorithm.
Uses the primary green absorption peak of hemoglobin (Verkruysse et al., 2008).
"""

from __future__ import annotations
import numpy as np


class GreenAlgorithm:
    """
    Extracts rPPG pulse signal directly from the Green channel or Green-Red difference.
    Green wavelength (approx 520-570nm) contains the strongest plethysmographic modulation.
    """

    @staticmethod
    def extract_green(rgb_series: np.ndarray) -> np.ndarray:
        """
        Input: rgb_series of shape (N, 3) [R, G, B].
        Output: 1D array of length N containing normalized Green signal.
        """
        if rgb_series.shape[0] < 2:
            return np.zeros(len(rgb_series), dtype=np.float32)

        g = rgb_series[:, 1]
        mean_g = np.mean(g)
        if mean_g == 0:
            mean_g = 1.0

        # Invert so systolic peak (absorption increase = reflection decrease) maps to positive pulse peak
        norm_g = -(g - mean_g) / mean_g
        return norm_g.astype(np.float32)

    @staticmethod
    def extract_green_red_diff(rgb_series: np.ndarray) -> np.ndarray:
        """
        Normalized difference: -(G/mean(G) - R/mean(R))
        Cancels common-mode ambient lighting fluctuations.
        """
        if rgb_series.shape[0] < 2:
            return np.zeros(len(rgb_series), dtype=np.float32)

        r = rgb_series[:, 0]
        g = rgb_series[:, 1]

        mean_r = np.mean(r) if np.mean(r) != 0 else 1.0
        mean_g = np.mean(g) if np.mean(g) != 0 else 1.0

        diff = -((g / mean_g) - (r / mean_r))
        return diff.astype(np.float32)
