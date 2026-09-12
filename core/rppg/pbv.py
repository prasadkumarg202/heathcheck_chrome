"""
Pulse Blood Volume (PBV) Algorithm for Contactless rPPG.
Reference: de Haan & van Leest, Improved photoplethysmographic imaging using normalized blood volume pulse (PBV), IEEE TBME 2014.
"""

from __future__ import annotations
import numpy as np


class PBVAlgorithm:
    """
    Pulse Blood Volume (PBV) rPPG extraction method.
    Uses characteristic blood absorption signature vector across RGB channels.
    """

    def __init__(self, pbv_vector: tuple = (-0.499, 0.763, -0.412)):
        pbv_arr = np.array(pbv_vector, dtype=np.float64)
        self.pbv = pbv_arr / (np.linalg.norm(pbv_arr) + 1e-8)

    def extract(self, rgb_series: np.ndarray, fs: float = 30.0) -> np.ndarray:
        """
        Extracts 1D BVP signal from raw RGB temporal traces.
        Input: rgb_series of shape (N, 3), where columns are [R, G, B].
        Output: 1D array of length N containing pulse waveform (BVP).
        """
        rgb = np.asarray(rgb_series, dtype=np.float64)
        if len(rgb.shape) != 2 or rgb.shape[1] != 3:
            return np.zeros(len(rgb), dtype=np.float32)

        N = rgb.shape[0]
        if N < 4:
            return np.zeros(N, dtype=np.float32)

        # 1. Temporal Normalization
        mean_rgb = np.mean(rgb, axis=0) + 1e-8
        cn = (rgb / mean_rgb) - 1.0  # Shape (N, 3)

        # 2. Covariance matrix C * C^T
        cov = np.matmul(cn.T, cn) / float(N)  # (3, 3)

        # 3. Optimal projection weights: W = (C C^T)^-1 * Pbv / (Pbv^T * (C C^T)^-1 * Pbv)
        try:
            cov_inv = np.linalg.pinv(cov + 1e-6 * np.eye(3))
            w = np.matmul(cov_inv, self.pbv)
            denom = np.matmul(self.pbv.T, w) + 1e-8
            weights = w / denom  # (3,)
        except Exception:
            weights = self.pbv

        # 4. Extract BVP
        bvp = np.matmul(cn, weights)
        return bvp.astype(np.float32)
