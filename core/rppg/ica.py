"""
Independent Component Analysis (ICA) and PCA rPPG Extraction Algorithm.
Reference: Poh, M. Z., McDuff, D. J., & Picard, R. W. (2010).
"Non-contact, automated cardiac pulse measurements using video imaging and blind source separation."
Optics Express, 18(10), 10762-10774.
"""

from __future__ import annotations
import numpy as np
from scipy import signal


class ICAAlgorithm:
    """
    Decomposes normalized RGB traces into independent source signals and selects
    the cardiac component based on maximum spectral peak prominence in the heart rate band.
    """

    def __init__(self, n_components: int = 3, max_iter: int = 200, tol: float = 1e-4):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol

    def extract(self, rgb_series: np.ndarray, fs: float = 30.0, hr_low_hz: float = 0.7, hr_high_hz: float = 3.5) -> np.ndarray:
        """
        Input: rgb_series of shape (N, 3) [R, G, B].
        Output: 1D array of length N containing selected BVP signal.
        """
        N = rgb_series.shape[0]
        if N < 30:
            return np.zeros(N, dtype=np.float32)

        # 1. Zero-mean and Normalize
        X = rgb_series.T.astype(np.float64)  # shape (3, N)
        X = X - np.mean(X, axis=1, keepdims=True)

        # 2. PCA Whitening
        cov = np.dot(X, X.T) / N
        eigvals, eigvecs = np.linalg.eigh(cov)
        # Sort descending
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        # Regularize small eigenvalues
        eigvals = np.maximum(eigvals, 1e-8)
        whitening_matrix = np.dot(np.diag(1.0 / np.sqrt(eigvals)), eigvecs.T)
        X_white = np.dot(whitening_matrix, X)  # shape (3, N)

        # 3. FastICA iteration (deflation or parallel)
        W = np.random.RandomState(42).randn(self.n_components, self.n_components)
        # Gram-Schmidt orthonormalization
        W, _ = np.linalg.qr(W)

        # FastICA g(u) = tanh(u), g'(u) = 1 - tanh^2(u)
        for i in range(self.n_components):
            w = W[i, :].copy()
            w = w / np.linalg.norm(w)
            for _ in range(self.max_iter):
                u = np.dot(w, X_white)
                tanh_u = np.tanh(u)
                w_new = np.mean(X_white * tanh_u, axis=1) - np.mean(1.0 - tanh_u**2) * w
                # Decorrelate with previous components
                w_new = w_new - np.dot(np.dot(w_new, W[:i, :].T), W[:i, :])
                w_new = w_new / (np.linalg.norm(w_new) + 1e-10)

                if np.abs(np.abs(np.dot(w_new, w)) - 1.0) < self.tol:
                    break
                w = w_new
            W[i, :] = w

        # Independent sources
        S = np.dot(W, X_white)  # shape (3, N)

        # 4. Component Selection: choose component with highest spectral power in HR band
        best_comp_idx = 0
        best_snr = -1.0

        for c in range(self.n_components):
            comp = S[c, :]
            # Compute PSD
            freqs, psd = signal.welch(comp, fs=fs, nperseg=min(len(comp), int(fs * 4)))
            hr_mask = (freqs >= hr_low_hz) & (freqs <= hr_high_hz)

            if np.any(hr_mask):
                hr_power = np.sum(psd[hr_mask])
                total_power = np.sum(psd) + 1e-8
                snr = hr_power / total_power
                if snr > best_snr:
                    best_snr = snr
                    best_comp_idx = c

        selected_signal = S[best_comp_idx, :]
        return selected_signal.astype(np.float32)
