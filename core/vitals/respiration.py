"""
Respiration Rate (RR) Estimation Engine for AuraPulse.
Extracts respiratory rhythm using a robust multi-feature ensemble:
1. Respiratory-Induced Baseline Variation (RIBV) via zero-phase bandpass filtering.
2. Respiratory-Induced Amplitude Variation (RIAV) on systolic peak envelopes.
3. Respiratory Sinus Arrhythmia (RSA) on instantaneous Inter-Beat Intervals.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, List
import numpy as np
from scipy import signal


@dataclass
class RespirationResult:
    rr_rpm: float                # Breaths per minute (rpm)
    confidence: float            # 0.0 to 1.0
    is_valid: bool
    pattern: str = "Eupnea"      # "Eupnea", "Tachypnea", "Bradypnea", "Hyperventilation"
    ie_ratio: float = 1.6        # Inhalation : Exhalation ratio (1:1.6)
    respiration_waveform: Optional[List[float]] = None  # Downsampled respiratory waveform
    methods_rpm: Optional[dict] = None
    rejection_reason: Optional[str] = None


class RespirationEngine:
    """
    Computes Respiration Rate from rPPG respiratory-induced variations with multi-signal ensemble.
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
        sqi_score: float = 75.0,
        ppi_intervals_ms: Optional[List[float]] = None
    ) -> RespirationResult:
        N = len(raw_bvp_or_green)
        if N < int(fs * 8):
            return RespirationResult(
                rr_rpm=0.0,
                confidence=0.0,
                is_valid=False,
                pattern="Indeterminate",
                ie_ratio=0.0,
                respiration_waveform=None,
                rejection_reason="INSUFFICIENT_WINDOW_FOR_RESPIRATION",
            )

        estimates = {}
        confidences = {}
        extracted_resp_sig = None

        # 1. Method A: RIBV (Low-Frequency Baseline Variation)
        nyquist = 0.5 * fs
        low = max(0.01, self.min_hz / nyquist)
        high = min(0.99, self.max_hz / nyquist)

        try:
            b, a = signal.butter(3, [low, high], btype='band')
            ribv_sig = signal.filtfilt(b, a, raw_bvp_or_green)
            extracted_resp_sig = ribv_sig

            nfft = max(2048, N * 4)
            win = np.hamming(N)
            fft_vals = np.abs(np.fft.rfft((ribv_sig - np.mean(ribv_sig)) * win, n=nfft))
            freqs = np.fft.rfftfreq(nfft, d=1.0/fs)

            resp_mask = (freqs >= self.min_hz) & (freqs <= self.max_hz)
            if np.any(resp_mask):
                sub_freqs = freqs[resp_mask]
                sub_fft = fft_vals[resp_mask]
                peak_idx = int(np.argmax(sub_fft))
                peak_freq = float(sub_freqs[peak_idx])

                if 0 < peak_idx < len(sub_fft) - 1:
                    alpha = float(sub_fft[peak_idx - 1])
                    beta = float(sub_fft[peak_idx])
                    gamma = float(sub_fft[peak_idx + 1])
                    denom = alpha - 2 * beta + gamma
                    if abs(denom) > 1e-10:
                        delta = 0.5 * (alpha - gamma) / denom
                        df = float(sub_freqs[1] - sub_freqs[0])
                        peak_freq += delta * df

                rr_ribv = float(np.clip(peak_freq * 60.0, self.min_rr_rpm, self.max_rr_rpm))
                mean_p = float(np.mean(sub_fft)) + 1e-6
                prominence = float(sub_fft[peak_idx] / (mean_p * 2.5))
                conf_ribv = float(np.clip(prominence, 0.1, 1.0))
                estimates["RIBV"] = rr_ribv
                confidences["RIBV"] = conf_ribv
        except Exception:
            pass

        # 2. Method B: RIAV (Amplitude Variation on systolic peak envelopes)
        try:
            b_c, a_c = signal.butter(3, [0.7 / nyquist, 3.5 / nyquist], btype='band')
            cardiac = signal.filtfilt(b_c, a_c, raw_bvp_or_green)
            analytic = signal.hilbert(cardiac)
            env = np.abs(analytic)
            env_detrend = signal.filtfilt(b, a, env)

            nfft = max(2048, N * 4)
            fft_env = np.abs(np.fft.rfft((env_detrend - np.mean(env_detrend)) * np.hamming(N), n=nfft))
            freqs_env = np.fft.rfftfreq(nfft, d=1.0/fs)
            mask_env = (freqs_env >= self.min_hz) & (freqs_env <= self.max_hz)
            if np.any(mask_env):
                sub_f = freqs_env[mask_env]
                sub_fft_env = fft_env[mask_env]
                pk_i = int(np.argmax(sub_fft_env))
                rr_riav = float(np.clip(sub_f[pk_i] * 60.0, self.min_rr_rpm, self.max_rr_rpm))
                prom_riav = float(sub_fft_env[pk_i] / (np.mean(sub_fft_env) * 2.5 + 1e-6))
                conf_riav = float(np.clip(prom_riav, 0.1, 0.95))
                estimates["RIAV"] = rr_riav
                confidences["RIAV"] = conf_riav
        except Exception:
            pass

        # 3. Method C: RSA (Respiratory Sinus Arrhythmia)
        if ppi_intervals_ms is not None and len(ppi_intervals_ms) >= 12:
            try:
                t_cum = np.cumsum(np.array(ppi_intervals_ms)) / 1000.0
                t_grid = np.arange(0, t_cum[-1], 0.25)
                if len(t_grid) > 16:
                    ibi_interp = np.interp(t_grid, t_cum, ppi_intervals_ms)
                    f_rsa, psd_rsa = signal.welch(ibi_interp - np.mean(ibi_interp), fs=4.0, nperseg=min(len(ibi_interp), 64))
                    rsa_mask = (f_rsa >= self.min_hz) & (f_rsa <= self.max_hz)
                    if np.any(rsa_mask):
                        sub_f_rsa = f_rsa[rsa_mask]
                        sub_p_rsa = psd_rsa[rsa_mask]
                        pk_rsa = int(np.argmax(sub_p_rsa))
                        rr_rsa = float(np.clip(sub_f_rsa[pk_rsa] * 60.0, self.min_rr_rpm, self.max_rr_rpm))
                        estimates["RSA"] = rr_rsa
                        confidences["RSA"] = 0.80
            except Exception:
                pass

        if not estimates:
            return RespirationResult(
                rr_rpm=0.0,
                confidence=0.0,
                is_valid=False,
                pattern="Indeterminate",
                ie_ratio=0.0,
                respiration_waveform=None,
                rejection_reason="NO_RESPIRATORY_ENERGY",
            )

        # Method selection / fusion:
        # If RIBV is prominent (>0.6 conf), it is the primary direct optical baseline
        if "RIBV" in estimates and confidences["RIBV"] >= 0.55:
            final_rr = estimates["RIBV"]
            final_conf = confidences["RIBV"] * (sqi_score / 100.0)
        else:
            weights = [confidences[k] for k in estimates]
            vals = [estimates[k] for k in estimates]
            final_rr = float(sum(v * w for v, w in zip(vals, weights)) / sum(weights))
            final_conf = float(np.mean(list(confidences.values())) * (sqi_score / 100.0))

        is_valid = (final_conf >= 0.35) and (sqi_score >= 30.0)

        # Pattern classification & I:E ratio
        if final_rr > 22.0:
            pattern = "Tachypnea"
            ie_ratio = 1.3
        elif final_rr < 11.0:
            pattern = "Bradypnea"
            ie_ratio = 1.9
        else:
            pattern = "Eupnea"
            ie_ratio = 1.6

        # Downsample respiratory waveform
        resp_wave = None
        if extracted_resp_sig is not None and len(extracted_resp_sig) > 0:
            norm_r = extracted_resp_sig / (np.max(np.abs(extracted_resp_sig)) + 1e-6)
            resp_wave = [round(float(v), 3) for v in norm_r[-45:]]

        return RespirationResult(
            rr_rpm=float(round(final_rr, 1)) if is_valid else 0.0,
            confidence=float(round(final_conf, 2)),
            is_valid=is_valid,
            pattern=pattern if is_valid else "Indeterminate",
            ie_ratio=ie_ratio if is_valid else 0.0,
            respiration_waveform=resp_wave,
            methods_rpm=estimates,
            rejection_reason=None if is_valid else "LOW_RESPIRATION_CONFIDENCE",
        )
