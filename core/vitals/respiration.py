"""
Respiration Rate (RR) Estimation Engine for AuraPulse.
Extracts respiratory rhythm from photoplethysmogram baseline modulation (RIBV)
and amplitude modulation (RIAV) in the physiological breathing band (6 to 30 breaths/min).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
from scipy import signal


@dataclass
class RespirationResult:
    rr_rpm: float                # Breaths per minute (rpm)
    confidence: float            # 0.0 to 1.0
    is_valid: bool
    rejection_reason: Optional[str] = None


class RespirationEngine:
    """
    Computes Respiration Rate from rPPG respiratory-induced variations.
    """

    def __init__(
        self,
        min_rr_rpm: float = 6.0,   # 0.10 Hz
        max_rr_rpm: float = 30.0,  # 0.50 Hz
    ):
        self.min_rr_rpm = min_rr_rpm
        self.max_rr_rpm = max_rr_rpm
        self.min_hz = min_rr_rpm / 60.0
        self.max_hz = max_rr_rpm / 60.0

    def estimate_rr(
        self,
        raw_bvp_or_green: np.ndarray,
        fs: float = 30.0,
        sqi_score: float = 75.0
    ) -> RespirationResult:
        """
        Calculates respiration rate from low-frequency modulation of the rPPG signal.
        Requires at least 12 seconds of stable data for reliable frequency resolution.
        """
        N = len(raw_bvp_or_green)
        if N < int(fs * 10):  # Minimum 10 seconds
            return RespirationResult(
                rr_rpm=0.0,
                confidence=0.0,
                is_valid=False,
                rejection_reason="INSUFFICIENT_WINDOW_FOR_RESPIRATION",
            )

        # 1. Extract Low-Frequency Baseline Variation (RIBV)
        # Bandpass filter in [0.1, 0.5] Hz
        nyquist = 0.5 * fs
        low = max(0.01, self.min_hz / nyquist)
        high = min(0.99, self.max_hz / nyquist)

        b, a = signal.butter(3, [low, high], btype='band')
        ribv_sig = signal.filtfilt(b, a, raw_bvp_or_green)

        # 2. Spectral Analysis of Respiratory Signal
        nfft = max(2048, N * 4)
        win = np.hamming(N)
        fft_vals = np.abs(np.fft.rfft(ribv_sig * win, n=nfft))
        freqs = np.fft.rfftfreq(nfft, d=1.0/fs)

        resp_mask = (freqs >= self.min_hz) & (freqs <= self.max_hz)
        if not np.any(resp_mask):
            return RespirationResult(
                rr_rpm=0.0,
                confidence=0.0,
                is_valid=False,
                rejection_reason="NO_RESPIRATORY_ENERGY",
            )

        sub_freqs = freqs[resp_mask]
        sub_fft = fft_vals[resp_mask]

        peak_idx = np.argmax(sub_fft)
        peak_freq = sub_freqs[peak_idx]

        # Sub-bin interpolation
        if 0 < peak_idx < len(sub_fft) - 1:
            alpha = sub_fft[peak_idx - 1]
            beta = sub_fft[peak_idx]
            gamma = sub_fft[peak_idx + 1]
            delta = 0.5 * (alpha - gamma) / (alpha - 2 * beta + gamma + 1e-10)
            df = sub_freqs[1] - sub_freqs[0]
            peak_freq += delta * df

        rr_rpm = float(peak_freq * 60.0)

        # Confidence metric based on peak prominence and SQI
        mean_resp_power = np.mean(sub_fft) + 1e-6
        prominence = float(sub_fft[peak_idx] / (mean_resp_power * 3.5))
        raw_conf = min(1.0, max(0.0, prominence)) * (sqi_score / 100.0)

        is_valid = (raw_conf >= 0.35) and (self.min_rr_rpm <= rr_rpm <= self.max_rr_rpm)

        return RespirationResult(
            rr_rpm=round(rr_rpm, 1) if is_valid else 0.0,
            confidence=round(raw_conf, 3) if is_valid else 0.0,
            is_valid=is_valid,
            rejection_reason=None if is_valid else "LOW_RESPIRATION_CONFIDENCE",
        )
