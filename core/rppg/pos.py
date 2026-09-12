"""
Plane-Orthogonal-to-Skin (POS) rPPG Algorithm.
Reference: Wang, W., den Brinker, A. C., Stuijk, S., & de Haan, G. (2017).
"Algorithmic Principles of Remote PPG." IEEE TBME, 64(7), 1479-1491.
"""

from __future__ import annotations
import numpy as np


class POSAlgorithm:
    """
    POS projects temporal RGB color traces onto a plane orthogonal to the skin tone
    vector, eliminating specular reflection and intensity variations.
    """

    def __init__(self, window_len_s: float = 1.6):
        self.window_len_s = window_len_s

    def extract(self, rgb_series: np.ndarray, fs: float = 30.0) -> np.ndarray:
        """
        Extracts pulse signal from temporal RGB traces.
        Input: rgb_series of shape (N, 3), where columns are [R, G, B].
        Output: 1D array of length N containing pulse waveform (BVP).
        """
        N = rgb_series.shape[0]
        if N < 15:
            return np.zeros(N, dtype=np.float32)

        # Window length in samples
        l_samples = int(np.round(self.window_len_s * fs))
        l_samples = max(4, min(l_samples, N))

        # Output signal accumulator
        H = np.zeros(N, dtype=np.float64)

        # Normalized temporal RGB: C_n(t) = C(t) / mean(C_window)
        # Sliding sub-window projection
        for m in range(N - l_samples + 1):
            sub_rgb = rgb_series[m : m + l_samples, :]  # (l_samples, 3)
            mean_rgb = np.mean(sub_rgb, axis=0)
            mean_rgb = np.where(mean_rgb == 0, 1.0, mean_rgb)

            # Normalized sub-window
            cn = sub_rgb / mean_rgb  # shape: (l_samples, 3)

            # Projection vectors
            # S1 = G_n - B_n
            # S2 = -2 * R_n + G_n + B_n
            s1 = cn[:, 1] - cn[:, 2]
            s2 = -2.0 * cn[:, 0] + cn[:, 1] + cn[:, 2]

            std_s1 = np.std(s1)
            std_s2 = np.std(s2)

            alpha = (std_s1 / (std_s2 + 1e-8)) if std_s2 > 1e-8 else 0.0

            # Signal segment
            h_segment = s1 + alpha * s2
            # Zero-mean the segment
            h_segment = h_segment - np.mean(h_segment)

            # Overlap-add
            H[m : m + l_samples] += h_segment

        # Handle boundary extrapolation if needed
        if N > l_samples:
            weights = np.zeros(N)
            for m in range(N - l_samples + 1):
                weights[m : m + l_samples] += 1.0
            weights = np.where(weights == 0, 1.0, weights)
            H = H / weights

        return H.astype(np.float32)
