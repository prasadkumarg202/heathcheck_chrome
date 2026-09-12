"""
Signal Quality Index (SQI) Subsystem for AuraPulse.
Quantifies physiological signal purity via SNR, spectral peak prominence,
autocorrelation periodicity, and cross-ROI coherence.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np
from scipy import signal


@dataclass
class SQIResult:
    sqi_score: float             # 0 to 100
    category: str                # Invalid, Poor, Fair, Good, Excellent
    is_acceptable: bool          # True if >= threshold
    snr_db: float                # Signal-to-Noise Ratio in dB
    peak_prominence: float       # Spectral peak prominence ratio
    periodicity_score: float     # Autocorrelation peak magnitude (0 to 1)
    cross_roi_correlation: float # Inter-ROI signal correlation (-1 to 1)
    rejection_reason: Optional[str] = None


class SignalQualityEngine:
    """
    Evaluates physiological signal fidelity and rejects motion-corrupted or noisy windows.
    """

    def __init__(self, min_acceptable_sqi: float = 35.0):
        self.min_acceptable_sqi = min_acceptable_sqi

    def compute_sqi(
        self,
        bvp_signal: np.ndarray,
        fs: float = 30.0,
        hr_low_hz: float = 0.7,
        hr_high_hz: float = 3.5,
        other_roi_signals: Optional[Dict[str, np.ndarray]] = None,
    ) -> SQIResult:
        """
        Computes composite Signal Quality Index on a temporal BVP signal window.
        """
        N = len(bvp_signal)
        if N < int(fs * 3):  # Minimum 3 seconds
            return SQIResult(
                sqi_score=0.0,
                category="Invalid",
                is_acceptable=False,
                snr_db=-20.0,
                peak_prominence=0.0,
                periodicity_score=0.0,
                cross_roi_correlation=0.0,
                rejection_reason="WINDOW_TOO_SHORT",
            )

        # 1. Welch PSD Analysis
        nperseg = min(N, int(fs * 8))
        freqs, psd = signal.welch(bvp_signal, fs=fs, nperseg=nperseg, noverlap=nperseg // 2)

        # Filter within HR band [0.7, 3.5] Hz
        hr_mask = (freqs >= hr_low_hz) & (freqs <= hr_high_hz)
        if not np.any(hr_mask) or np.all(psd == 0):
            return SQIResult(
                sqi_score=0.0,
                category="Invalid",
                is_acceptable=False,
                snr_db=-20.0,
                peak_prominence=0.0,
                periodicity_score=0.0,
                cross_roi_correlation=0.0,
                rejection_reason="ZERO_SPECTRAL_ENERGY",
            )

        hr_freqs = freqs[hr_mask]
        hr_psd = psd[hr_mask]

        # Find dominant peak in HR band
        peak_idx = np.argmax(hr_psd)
        peak_freq = hr_freqs[peak_idx]
        peak_power = hr_psd[peak_idx]

        # Compute SNR: power within peak ± 0.18 Hz + 1st harmonic (2*peak_freq ± 0.18 Hz)
        # vs all other spectral energy in [0.5, 4.0] Hz
        band_mask = (freqs >= 0.5) & (freqs <= 4.0)
        signal_mask = (np.abs(freqs - peak_freq) <= 0.18) | (np.abs(freqs - 2 * peak_freq) <= 0.18)
        
        signal_power = np.sum(psd[band_mask & signal_mask])
        noise_power = np.sum(psd[band_mask & (~signal_mask)]) + 1e-8

        snr_ratio = signal_power / noise_power
        snr_db = float(10.0 * np.log10(max(snr_ratio, 1e-4)))

        # 2. Peak Prominence / Sharpness
        mean_psd_band = np.mean(hr_psd) + 1e-8
        prominence_ratio = float(peak_power / mean_psd_band)
        peak_prominence = min(1.0, prominence_ratio / 4.0)

        # 3. Autocorrelation Periodicity
        sig_norm = bvp_signal - np.mean(bvp_signal)
        sig_std = np.std(sig_norm) + 1e-8
        sig_norm = sig_norm / sig_std

        autocorr = signal.correlate(sig_norm, sig_norm, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        autocorr = autocorr / (autocorr[0] + 1e-8)

        # Search autocorrelation peak in physiological lag range: [fs/3.5, fs/0.7]
        min_lag = int(fs / hr_high_hz)
        max_lag = min(len(autocorr) - 1, int(fs / hr_low_hz))

        if max_lag > min_lag:
            ac_slice = autocorr[min_lag:max_lag]
            periodicity_score = float(max(0.0, np.max(ac_slice)))
        else:
            periodicity_score = 0.0

        # 4. Cross-ROI Correlation (if other ROIs available)
        cross_corr_val = 1.0
        if other_roi_signals:
            corrs = []
            for other_name, other_sig in other_roi_signals.items():
                if len(other_sig) == len(bvp_signal):
                    std_o = np.std(other_sig)
                    if std_o > 1e-6:
                        c_matrix = np.corrcoef(bvp_signal, other_sig)
                        if not np.isnan(c_matrix[0, 1]):
                            corrs.append(c_matrix[0, 1])
            if corrs:
                cross_corr_val = float(np.mean(corrs))

        # Composite SQI Score (0 to 100)
        # Map SNR: -8 dB -> 0, +12 dB -> 100
        snr_norm = float(np.clip((snr_db + 8.0) / 20.0 * 100.0, 0.0, 100.0))
        prom_norm = float(peak_prominence * 100.0)
        period_norm = float(np.clip(periodicity_score * 100.0, 0.0, 100.0))
        cross_norm = float(np.clip(max(0.0, cross_corr_val) * 100.0, 0.0, 100.0))

        sqi_raw = (
            0.40 * snr_norm +
            0.25 * period_norm +
            0.20 * prom_norm +
            0.15 * cross_norm
        )
        sqi_score = float(np.clip(sqi_raw, 0.0, 100.0))

        # Classification
        if sqi_score < 25.0:
            category = "Invalid"
        elif sqi_score < 45.0:
            category = "Poor"
        elif sqi_score < 68.0:
            category = "Fair"
        elif sqi_score < 85.0:
            category = "Good"
        else:
            category = "Excellent"

        is_acceptable = sqi_score >= self.min_acceptable_sqi
        rejection_reason = None if is_acceptable else f"LOW_SIGNAL_QUALITY_{category.upper()}"

        return SQIResult(
            sqi_score=round(sqi_score, 1),
            category=category,
            is_acceptable=is_acceptable,
            snr_db=round(snr_db, 2),
            peak_prominence=round(peak_prominence, 3),
            periodicity_score=round(periodicity_score, 3),
            cross_roi_correlation=round(cross_corr_val, 3),
            rejection_reason=rejection_reason,
        )
