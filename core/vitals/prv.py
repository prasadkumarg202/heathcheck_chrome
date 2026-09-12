"""
Pulse Rate Variability (PRV) Engine for AuraPulse.
Computes optical pulse-to-pulse intervals (PPI) with sub-sample peak interpolation
and derives accurate time-domain and geometric variability metrics (RMSSD, SDNN, pNN50, Poincaré).

SCIENTIFIC PRINCIPLE:
Sub-sample parabolic peak refinement eliminates the ~33.3ms video frame quantization error,
yielding physiologically grounded PRV statistics.
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
    Computes Pulse Rate Variability from detected systolic BVP peaks with sub-frame interpolation.
    """

    def __init__(self, min_duration_s: float = 12.0):
        self.min_duration_s = min_duration_s

    def compute_prv(
        self,
        bvp_signal: np.ndarray,
        fs: float = 30.0,
        sqi_score: float = 75.0,
        hr_bpm_hint: Optional[float] = None,
    ) -> PRVResult:
        """
        Extracts pulse peaks with sub-sample peak refinement, filters ectopic intervals,
        and computes PRV statistics.
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

        if hr_bpm_hint and hr_bpm_hint > 0:
            expected_distance = max(int(fs / (hr_bpm_hint / 60.0) * 0.60), 6)
        else:
            expected_distance = max(int(fs * 0.38), 6)

        # High-order Butterworth smoothing for peak isolation
        b, a = signal.butter(3, [0.7 / (0.5 * fs), 3.0 / (0.5 * fs)], btype='band')
        sig_smooth = signal.filtfilt(b, a, bvp_signal)

        # Find discrete peak indices
        peaks, _ = signal.find_peaks(
            sig_smooth,
            distance=expected_distance,
            prominence=np.std(sig_smooth) * 0.35
        )

        if len(peaks) < 6:
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

        # Sub-sample parabolic peak refinement
        refined_peaks = []
        for p in peaks:
            if 0 < p < len(sig_smooth) - 1:
                y0 = sig_smooth[p - 1]
                y1 = sig_smooth[p]
                y2 = sig_smooth[p + 1]
                denom = 2 * (y0 - 2 * y1 + y2)
                if abs(denom) > 1e-6:
                    delta = (y0 - y2) / denom
                    refined_p = p + delta
                else:
                    refined_p = float(p)
            else:
                refined_p = float(p)
            refined_peaks.append(refined_p)

        refined_peaks = np.array(refined_peaks)

        # Pulse-to-Pulse Intervals in milliseconds
        ppi_ms = (np.diff(refined_peaks) / fs) * 1000.0

        # Physiological outlier rejection (350 ms to 1500 ms)
        med_ppi = np.median(ppi_ms)
        valid_mask = (ppi_ms >= 350.0) & (ppi_ms <= 1500.0) & (np.abs(ppi_ms - med_ppi) <= 0.20 * med_ppi)
        clean_ppi = ppi_ms[valid_mask]

        if len(clean_ppi) < 4:
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

        # 2. SDNN
        sdnn = float(np.std(clean_ppi, ddof=1)) if len(clean_ppi) > 1 else float(np.std(clean_ppi))

        # 3. RMSSD
        successive_diffs = np.diff(clean_ppi)
        rmssd = float(np.sqrt(np.mean(successive_diffs ** 2))) if len(successive_diffs) > 0 else 0.0

        # 4. pNN50
        nn50 = np.count_nonzero(np.abs(successive_diffs) > 50.0) if len(successive_diffs) > 0 else 0
        pnn50 = float((nn50 / len(successive_diffs)) * 100.0) if len(successive_diffs) > 0 else 0.0

        # 5. Poincaré Features
        if len(successive_diffs) > 0:
            sd1 = float(np.sqrt(0.5 * (rmssd ** 2)))
            sd2_val = 2.0 * (sdnn ** 2) - 0.5 * (rmssd ** 2)
            sd2 = float(np.sqrt(max(0.0, sd2_val)))
            sd1_sd2 = float(sd1 / sd2) if sd2 > 1e-4 else 0.0
        else:
            sd1, sd2, sd1_sd2 = 0.0, 0.0, 0.0

        conf = min(1.0, (sqi_score / 100.0) * (len(clean_ppi) / (duration_s * 0.8)))

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
