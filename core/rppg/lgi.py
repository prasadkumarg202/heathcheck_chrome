"""
Local Group Invariance (LGI) Algorithm for Contactless rPPG.
Reference: Pilz et al., Local Group Invariance for Remote Photoplethysmography, CVPR Workshops 2018.
"""

from __future__ import annotations
import numpy as np


class LGIAlgorithm:
    """
    Local Group Invariance (LGI) algorithm.
    Projects normalized spatial-temporal RGB trajectories onto the orthogonal subspace of illumination shift.
    """

    def __init__(self, window_len_s: float = 1.6):
        self.window_len_s = window_len_s

    def extract(self, rgb_series: np.ndarray, fs: float = 30.0) -> np.ndarray:
        """
        Extracts pulse signal from temporal RGB traces using LGI.
        Input: rgb_series of shape (N, 3), where columns are [R, G, B].
        """
        c = np.asarray(rgb_series, dtype=np.float64)
        if len(c.shape) != 2 or c.shape[1] != 3:
            return np.zeros(len(c), dtype=np.float32)

        N = c.shape[0]
        if N < 4:
            return np.zeros(N, dtype=np.float32)

        l = int(round(self.window_len_s * fs))
        l = max(4, min(l, N))
        h = np.zeros(N, dtype=np.float64)

        for m in range(0, N - l + 1):
            c_win = c[m : m + l]
            c_mean = np.mean(c_win, axis=0) + 1e-8
            u = c_win - c_mean

            try:
                _, _, vh = np.linalg.svd(u, full_matrices=False)
                pulse_comp = np.matmul(u, vh[1].T)
                h[m : m + l] += pulse_comp
            except Exception:
                h[m : m + l] += (c_win[:, 1] - c_win[:, 2])

        return h.astype(np.float32)
