"""
Chrominance-Based (CHROM) rPPG Algorithm.
Reference: de Haan, G., & Jeanne, V. (2013).
"Robust Pulse Rate from Chrominance-Based rPPG." IEEE TBME, 60(10), 2878-2886.
"""

from __future__ import annotations
import numpy as np


class CHROMAlgorithm:
    """
    CHROM projects normalized color variations into two orthogonal chrominance signals
    Xs and Ys, eliminating motion-induced specular reflections.
    """

    def __init__(self, window_len_s: float = 1.6):
        self.window_len_s = window_len_s

    def extract(self, rgb_series: np.ndarray, fs: float = 30.0) -> np.ndarray:
        """
        Extracts pulse signal from temporal RGB traces using chrominance projection.
        Input: rgb_series of shape (N, 3) [R, G, B].
        Output: 1D array of length N containing pulse waveform (BVP).
        """
        N = rgb_series.shape[0]
        if N < 15:
            return np.zeros(N, dtype=np.float32)

        l_samples = int(np.round(self.window_len_s * fs))
        l_samples = max(4, min(l_samples, N))

        H = np.zeros(N, dtype=np.float64)

        for m in range(N - l_samples + 1):
            sub_rgb = rgb_series[m : m + l_samples, :]
            mean_rgb = np.mean(sub_rgb, axis=0)
            mean_rgb = np.where(mean_rgb == 0, 1.0, mean_rgb)

            cn = sub_rgb / mean_rgb  # Normalized RGB

            # Chrominance signals:
            # Xs = 3 * Rn - 2 * Gn
            # Ys = 1.5 * Rn + Gn - 1.5 * Bn
            xs = 3.0 * cn[:, 0] - 2.0 * cn[:, 1]
            ys = 1.5 * cn[:, 0] + cn[:, 1] - 1.5 * cn[:, 2]

            std_xs = np.std(xs)
            std_ys = np.std(ys)

            alpha = (std_xs / (std_ys + 1e-8)) if std_ys > 1e-8 else 0.0

            h_segment = xs - alpha * ys
            h_segment = h_segment - np.mean(h_segment)

            H[m : m + l_samples] += h_segment

        if N > l_samples:
            weights = np.zeros(N)
            for m in range(N - l_samples + 1):
                weights[m : m + l_samples] += 1.0
            weights = np.where(weights == 0, 1.0, weights)
            H = H / weights

        return H.astype(np.float32)
