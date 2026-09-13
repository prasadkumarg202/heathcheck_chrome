"""
Heart Rate (HR) Estimation Engine for AuraPulse.
Implements multi-method ensemble estimation (FFT, Welch PSD, Autocorrelation, Peak Detection)
with sub-bin spectral interpolation and temporal smoothing.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
from scipy import signal


@dataclass
class HRResult:
    hr_bpm: float
    confidence: float            # 0.0 to 1.0
    method_estimates: dict
    is_valid: bool
    max_hr_bpm: float = 185.0
    hr_zone: str = "Resting"      # "Resting (<60%)", "Warmup (60-70%)", "Aerobic (70-80%)", "Threshold (80-90%)", "Peak (>90%)"
    hr_zone_pct: float = 0.0
    hr_reserve_pct: float = 0.0
    rejection_reason: Optional[str] = None


# Alias for backward compatibility
HeartRateResult = HRResult


class HeartRateEngine:
    """
    Robust Heart Rate extractor using multi-method spectral and temporal consensus.
    """

    def __init__(
        self,
        min_hr_bpm: float = 42.0,   # 0.7 Hz
        max_hr_bpm: float = 210.0,  # 3.5 Hz
        smoothing_alpha: float = 0.35,
    ):
        self.min_hr_bpm = min_hr_bpm
        self.max_hr_bpm = max_hr_bpm
        self.min_hz = min_hr_bpm / 60.0
        self.max_hz = max_hr_bpm / 60.0
        self.smoothing_alpha = smoothing_alpha
        self.last_smoothed_hr: Optional[float] = None

    def reset_tracking(self) -> None:
        self.last_smoothed_hr = None

    def estimate_hr(
        self,
        bvp_signal: np.ndarray,
        fs: float = 30.0,
        sqi_score: float = 80.0,
        age: float = 35.0,
        resting_hr_hint: float = 65.0
    ) -> HRResult:
        """
        Calculates heart rate from BVP signal using ensemble spectral and temporal analysis.
        """
        N = len(bvp_signal)
        if N < int(fs * 3.0):  # Minimum 3 seconds
            return HRResult(
                hr_bpm=0.0,
                confidence=0.0,
                method_estimates={},
                is_valid=False,
                rejection_reason="INSUFFICIENT_WINDOW_LENGTH",
            )

        estimates = {}

        # 1. METHOD A: Zero-Padded High-Resolution FFT
        hr_fft, conf_fft = self._estimate_fft(bvp_signal, fs)
        if hr_fft > 0:
            estimates["fft"] = (hr_fft, conf_fft)

        # 2. METHOD B: Welch PSD with Parabolic Peak Interpolation
        hr_welch, conf_welch = self._estimate_welch(bvp_signal, fs)
        if hr_welch > 0:
            estimates["welch"] = (hr_welch, conf_welch)

        # 3. METHOD C: Autocorrelation Lag
        hr_ac, conf_ac = self._estimate_autocorr(bvp_signal, fs)
        if hr_ac > 0:
            estimates["autocorr"] = (hr_ac, conf_ac)

        # 4. METHOD D: Time-Domain Pulse Peak Inter-Beat Interval (IBI)
        hr_peaks, conf_peaks = self._estimate_peaks(bvp_signal, fs)
        if hr_peaks > 0:
            estimates["peak_detection"] = (hr_peaks, conf_peaks)

        if not estimates:
            return HRResult(
                hr_bpm=0.0,
                confidence=0.0,
                method_estimates={},
                is_valid=False,
                rejection_reason="NO_CONVERGING_HR_PEAK",
            )

        # Consensus Voting across methods
        vals = [v[0] for v in estimates.values()]
        confs = [v[1] for v in estimates.values()]

        # Weight each method by its own confidence
        weights = np.array(confs) / (np.sum(confs) + 1e-6)
        raw_hr = float(np.sum(np.array(vals) * weights))

        # Check spread / consistency among methods
        hr_spread = float(np.ptp(vals))  # max - min
        consistency_conf = max(0.0, 1.0 - (hr_spread / 15.0))
        mean_conf = float(np.mean(confs)) * consistency_conf * (sqi_score / 100.0)

        # Temporal Smoothing: prevent sudden erratic jumps
        if self.last_smoothed_hr is not None:
            delta = abs(raw_hr - self.last_smoothed_hr)
            if delta > 25.0:
                # Potential jump/artifact: downweight new reading
                smoothed_hr = (1 - 0.15) * self.last_smoothed_hr + 0.15 * raw_hr
            else:
                smoothed_hr = (1 - self.smoothing_alpha) * self.last_smoothed_hr + self.smoothing_alpha * raw_hr
        else:
            smoothed_hr = raw_hr

        self.last_smoothed_hr = smoothed_hr

        # Calculate Gellish formula Max HR: 207 - (0.7 * Age)
        max_hr = float(round(207.0 - (0.7 * max(18.0, min(85.0, age))), 1))
        zone_pct = float(round((smoothed_hr / max(max_hr, 100.0)) * 100.0, 1))
        hr_reserve_pct = float(round(max(0.0, (smoothed_hr - resting_hr_hint) / max(1.0, max_hr - resting_hr_hint)) * 100.0, 1))

        if zone_pct < 60.0:
            zone_label = "Resting / Recovery"
        elif zone_pct < 70.0:
            zone_label = "Zone 1: Warmup"
        elif zone_pct < 80.0:
            zone_label = "Zone 2: Aerobic Fat Burn"
        elif zone_pct < 90.0:
            zone_label = "Zone 3: Cardio / Threshold"
        else:
            zone_label = "Zone 4: Peak / Anaerobic"

        return HRResult(
            hr_bpm=round(smoothed_hr, 1),
            confidence=round(min(1.0, max(0.0, mean_conf)), 3),
            method_estimates={k: round(v[0], 1) for k, v in estimates.items()},
            is_valid=True,
            max_hr_bpm=max_hr,
            hr_zone=zone_label,
            hr_zone_pct=zone_pct,
            hr_reserve_pct=hr_reserve_pct,
            rejection_reason=None,
        )

    def _estimate_fft(self, signal_data: np.ndarray, fs: float) -> Tuple[float, float]:
        N = len(signal_data)
        # 8x Zero-padding for sub-bin resolution
        nfft = max(2048, N * 8)
        win = np.hamming(N)
        fft_vals = np.abs(np.fft.rfft(signal_data * win, n=nfft))
        freqs = np.fft.rfftfreq(nfft, d=1.0/fs)

        mask = (freqs >= self.min_hz) & (freqs <= self.max_hz)
        if not np.any(mask):
            return 0.0, 0.0

        sub_freqs = freqs[mask]
        sub_fft = fft_vals[mask]

        peak_idx = np.argmax(sub_fft)
        peak_freq = sub_freqs[peak_idx]

        # Parabolic interpolation around peak
        if 0 < peak_idx < len(sub_fft) - 1:
            alpha = sub_fft[peak_idx - 1]
            beta = sub_fft[peak_idx]
            gamma = sub_fft[peak_idx + 1]
            delta = 0.5 * (alpha - gamma) / (alpha - 2 * beta + gamma + 1e-10)
            df = sub_freqs[1] - sub_freqs[0]
            peak_freq += delta * df

        hr_bpm = float(peak_freq * 60.0)
        conf = float(sub_fft[peak_idx] / (np.mean(sub_fft) * 4.0 + 1e-6))
        return hr_bpm, min(1.0, max(0.0, conf))

    def _estimate_welch(self, signal_data: np.ndarray, fs: float) -> Tuple[float, float]:
        nperseg = min(len(signal_data), int(fs * 6))
        freqs, psd = signal.welch(signal_data, fs=fs, nperseg=nperseg, noverlap=nperseg // 2, nfft=2048)
        mask = (freqs >= self.min_hz) & (freqs <= self.max_hz)
        if not np.any(mask):
            return 0.0, 0.0

        sub_freqs = freqs[mask]
        sub_psd = psd[mask]
        peak_idx = np.argmax(sub_psd)
        peak_freq = sub_freqs[peak_idx]

        hr_bpm = float(peak_freq * 60.0)
        conf = float(sub_psd[peak_idx] / (np.mean(sub_psd) * 3.0 + 1e-6))
        return hr_bpm, min(1.0, max(0.0, conf))

    def _estimate_autocorr(self, signal_data: np.ndarray, fs: float) -> Tuple[float, float]:
        sig = signal_data - np.mean(signal_data)
        std = np.std(sig)
        if std < 1e-6:
            return 0.0, 0.0
        sig = sig / std

        ac = signal.correlate(sig, sig, mode='full')
        ac = ac[len(ac)//2:]
        ac = ac / (ac[0] + 1e-8)

        min_lag = int(fs / self.max_hz)
        max_lag = min(len(ac) - 1, int(fs / self.min_hz))

        if max_lag <= min_lag:
            return 0.0, 0.0

        search_slice = ac[min_lag:max_lag]
        peak_idx_rel = np.argmax(search_slice)
        peak_lag = min_lag + peak_idx_rel

        if ac[peak_lag] < 0.2:  # Weak periodicity
            return 0.0, 0.0

        period_s = peak_lag / fs
        hr_bpm = float(60.0 / period_s)
        conf = float(ac[peak_lag])
        return hr_bpm, min(1.0, max(0.0, conf))

    def _estimate_peaks(self, signal_data: np.ndarray, fs: float) -> Tuple[float, float]:
        # Minimum distance between peaks corresponding to max HR
        min_distance = int(fs / self.max_hz * 0.7)
        peaks, props = signal.find_peaks(signal_data, distance=min_distance, prominence=np.std(signal_data) * 0.5)

        if len(peaks) < 3:
            return 0.0, 0.0

        ibis = np.diff(peaks) / fs  # Inter-beat intervals in seconds
        # Reject unreasonable IBIs
        valid_ibis = ibis[(ibis >= (1.0 / self.max_hz)) & (ibis <= (1.0 / self.min_hz))]

        if len(valid_ibis) < 2:
            return 0.0, 0.0

        mean_ibi = np.median(valid_ibis)
        hr_bpm = float(60.0 / mean_ibi)
        std_ibi = np.std(valid_ibis)
        conf = max(0.0, 1.0 - (std_ibi / (mean_ibi + 1e-6)))
        return hr_bpm, min(1.0, max(0.0, conf))
