"""
Pulse Rate Variability (PRV) Engine for AuraPulse.
Computes optical pulse-to-pulse intervals (PPI) and derives time-domain and geometric
variability metrics (RMSSD, SDNN, pNN50, Mean PPI, Poincaré SD1/SD2).

SCIENTIFIC PRINCIPLE:
These metrics represent Pulse Rate Variability (PRV) derived from photoplethysmogram waveforms,
which correlate with but are not identical to ECG-derived Heart Rate Variability (HRV) due to
pulse transit time variability and respiratory dynamics.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
from scipy import signal


@dataclass
class PRVResult:
    mean_ppi_ms: float           # Mean Pulse-to-Pulse Interval in ms
    rmssd_ms: float              # Root Mean Square of Successive Differences in ms
    sdnn_ms: float               # Standard Deviation of NN intervals in ms
    pnn50_pct: float             # Percentage of successive intervals differing > 50ms
    sd1_ms: float                # Poincaré short-term variability
    sd2_ms: float                # Poincaré long-term variability
    sd1_sd2_ratio: float         # SD1 / SD2 ratio
    num_peaks_detected: int
    duration_s: float
    confidence: float
    is_valid: bool
    label: str = "Pulse Rate Variability (PRV - Optical BVP)"
    rejection_reason: Optional[str] = None


class PRVEngine:
    """
    Computes Pulse Rate Variability from detected systolic BVP peaks.
    """

    def __init__(self, min_duration_s: float = 15.0):
        self.min_duration_s = min_duration_s

    def compute_prv(
        self,
        bvp_signal: np.ndarray,
        fs: float = 30.0,
        sqi_score: float = 75.0,
        hr_bpm_hint: Optional[float] = None,
    ) -> PRVResult:
        """
        Extracts pulse peaks, filters ectopic / outlier intervals, and computes PRV statistics.
        """
        N = len(bvp_signal)
        duration_s = N / fs

        if duration_s < self.min_duration_s:
            return PRVResult(
                mean_ppi_ms=0.0,
                rmssd_ms=0.0,
                sdnn_ms=0.0,
                pnn50_pct=0.0,
                sd1_ms=0.0,
                sd2_ms=0.0,
                sd1_sd2_ratio=0.0,
                num_peaks_detected=0,
                duration_s=duration_s,
                confidence=0.0,
                is_valid=False,
                rejection_reason="INSUFFICIENT_DURATION_FOR_PRV",
            )

        # Expected distance between peaks
        if hr_bpm_hint and hr_bpm_hint > 0:
            expected_distance = int(fs / (hr_bpm_hint / 60.0) * 0.65)
        else:
            expected_distance = int(fs * 0.4)  # ~150 BPM max

        # Bandpass smoothed for clean peak detection
        b, a = signal.butter(3, [0.7 / (0.5 * fs), 3.0 / (0.5 * fs)], btype='band')
        sig_smooth = signal.filtfilt(b, a, bvp_signal)

        # Find systolic peaks
        peaks, _ = signal.find_peaks(
            sig_smooth,
            distance=expected_distance,
            prominence=np.std(sig_smooth) * 0.4
        )

        if len(peaks) < 8:
            return PRVResult(
                mean_ppi_ms=0.0,
                rmssd_ms=0.0,
                sdnn_ms=0.0,
                pnn50_pct=0.0,
                sd1_ms=0.0,
                sd2_ms=0.0,
                sd1_sd2_ratio=0.0,
                num_peaks_detected=len(peaks),
                duration_s=duration_s,
                confidence=0.0,
                is_valid=False,
                rejection_reason="INSUFFICIENT_PEAKS_DETECTED",
            )

        # Compute Pulse-to-Pulse Intervals (PPI) in milliseconds
        ppi_ms = (np.diff(peaks) / fs) * 1000.0

        # Physiological filtering (300 ms to 1600 ms, and remove outliers > 20% from local median)
        med_ppi = np.median(ppi_ms)
        valid_mask = (ppi_ms >= 300.0) & (ppi_ms <= 1600.0) & (np.abs(ppi_ms - med_ppi) <= 0.25 * med_ppi)
        clean_ppi = ppi_ms[valid_mask]

        if len(clean_ppi) < 6:
            return PRVResult(
                mean_ppi_ms=0.0,
                rmssd_ms=0.0,
                sdnn_ms=0.0,
                pnn50_pct=0.0,
                sd1_ms=0.0,
                sd2_ms=0.0,
                sd1_sd2_ratio=0.0,
                num_peaks_detected=len(peaks),
                duration_s=duration_s,
                confidence=0.0,
                is_valid=False,
                rejection_reason="EXCESSIVE_ECTOPIC_INTERVALS",
            )

        # 1. Mean PPI
        mean_ppi = float(np.mean(clean_ppi))

        # 2. SDNN (Standard Deviation of NN intervals)
        sdnn = float(np.std(clean_ppi, ddof=1))

        # 3. RMSSD (Root Mean Square of Successive Differences)
        successive_diffs = np.diff(clean_ppi)
        rmssd = float(np.sqrt(np.mean(successive_diffs ** 2))) if len(successive_diffs) > 0 else 0.0

        # 4. pNN50
        nn50 = np.count_nonzero(np.abs(successive_diffs) > 50.0) if len(successive_diffs) > 0 else 0
        pnn50 = float((nn50 / len(successive_diffs)) * 100.0) if len(successive_diffs) > 0 else 0.0

        # 5. Poincaré Features (SD1 and SD2)
        if len(successive_diffs) > 0:
            sd1 = float(np.sqrt(0.5 * (rmssd ** 2)))
            sd2_val = 2.0 * (sdnn ** 2) - 0.5 * (rmssd ** 2)
            sd2 = float(np.sqrt(max(0.0, sd2_val)))
            sd1_sd2 = float(sd1 / sd2) if sd2 > 1e-4 else 0.0
        else:
            sd1, sd2, sd1_sd2 = 0.0, 0.0, 0.0

        conf = min(1.0, (sqi_score / 100.0) * (len(clean_ppi) / (duration_s * 1.0)))

        return PRVResult(
            mean_ppi_ms=round(mean_ppi, 1),
            rmssd_ms=round(rmssd, 1),
            sdnn_ms=round(sdnn, 1),
            pnn50_pct=round(pnn50, 1),
            sd1_ms=round(sd1, 1),
            sd2_ms=round(sd2, 1),
            sd1_sd2_ratio=round(sd1_sd2, 3),
            num_peaks_detected=len(clean_ppi) + 1,
            duration_s=round(duration_s, 1),
            confidence=round(conf, 3),
            is_valid=True,
            label="Pulse Rate Variability (PRV - Optical BVP)",
            rejection_reason=None,
        )
