"""
Advanced Physiological Stress & Autonomic Recovery Engine for AuraPulse.
Features:
1. Baevsky's Stress Index (SI = (AMo * 100) / (2 * Mo * MxDMn)) using 50ms histogram binning.
2. Frequency-Domain HRV: Low Frequency (LF: 0.04–0.15 Hz), High Frequency (HF: 0.15–0.40 Hz), LF/HF Ratio.
3. Sympathetic (SNS) Arousal Zone & Parasympathetic (PNS) Recovery Tone.
4. Pulse-Respiration Quotient (PRQ = HR / RR, synchronized, null when RR invalid).
5. Composite Autonomic Stress Score (0–100).
"""

from __future__ import annotations
import math
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from scipy import signal


@dataclass
class StressResult:
    stress_score: float                # 0 to 100 (0 = relaxed, 100 = high autonomic load)
    stress_level: str                  # Low, Moderate, High
    baevsky_stress_index: float        # Classical Baevsky SI (norm 50-150)
    parasympathetic_score: float       # PNS tone & recovery score (0-100)
    sympathetic_score: float           # SNS tone & arousal score (0-100)
    pulse_respiration_quotient: Optional[float]  # PRQ = HR / RR (null if RR invalid)
    lf_power: float                    # LF Power (0.04-0.15 Hz) in ms^2
    hf_power: float                    # HF Power (0.15-0.40 Hz) in ms^2
    lf_hf_ratio: float                 # Sympathovagal balance (LF / HF)
    ans_balance_ratio: float           # RMSSD / HR proxy
    confidence: float                  # 0.0 to 1.0
    is_valid: bool
    disclaimer: str = "Physiological autonomic indicator — not a clinical diagnosis."
    rejection_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class StressEngine:
    """
    Evaluates autonomic stress load, Baevsky Index, PNS tone, SNS zone, and PRQ.
    """

    def compute_stress_index(
        self,
        hr_bpm: float,
        rmssd_ms: float,
        rr_rpm: float,
        confidence_hr: float = 0.8,
        confidence_prv: float = 0.8,
        ppi_intervals_ms: Optional[List[float]] = None,
        bvp_signal: Optional[np.ndarray] = None,
        fs: float = 30.0,
        confidence_rr: float = 0.8,
    ) -> StressResult:
        if hr_bpm <= 0 or rmssd_ms <= 0:
            return StressResult(
                stress_score=0.0,
                stress_level="Unknown",
                baevsky_stress_index=0.0,
                parasympathetic_score=0.0,
                sympathetic_score=0.0,
                pulse_respiration_quotient=None,
                lf_power=0.0,
                hf_power=0.0,
                lf_hf_ratio=1.0,
                ans_balance_ratio=0.0,
                confidence=0.0,
                is_valid=False,
                rejection_reason="INSUFFICIENT_VITAL_METRICS_FOR_STRESS",
            )

        rmssd_clamped = max(5.0, min(150.0, rmssd_ms))
        hr_clamped = max(42.0, min(180.0, hr_bpm))

        # 1. Classical Baevsky Stress Index calculation with 50ms histogram bins
        if ppi_intervals_ms is not None and len(ppi_intervals_ms) >= 8:
            ppis = np.array(ppi_intervals_ms)
            min_p = float(np.min(ppis))
            max_p = float(np.max(ppis))
            bins = np.arange(min_p, max_p + 50.0, 50.0)
            if len(bins) < 2:
                bins = np.array([min_p - 25.0, min_p + 25.0])
            counts, bin_edges = np.histogram(ppis, bins=bins)
            max_bin_idx = int(np.argmax(counts))
            mo_s = float((bin_edges[max_bin_idx] + 25.0) / 1000.0)
            amo_pct = float((counts[max_bin_idx] / len(ppis)) * 100.0)
            mx_d_mn_s = max(0.04, (max_p - min_p) / 1000.0)
            baevsky_si = float(amo_pct / (2.0 * max(0.35, mo_s) * mx_d_mn_s))
        else:
            approx_mo = 60.0 / hr_clamped
            approx_amo = min(80.0, max(25.0, 30.0 + (hr_clamped - 60.0) * 0.75))
            approx_range = max(0.06, min(0.45, rmssd_clamped / 240.0))
            baevsky_si = float(approx_amo / (2.0 * approx_mo * approx_range))

        baevsky_si = float(np.clip(baevsky_si, 15.0, 850.0))

        # 2. Spectral LF / HF Decomposition
        lf_p = 500.0
        hf_p = 400.0
        if ppi_intervals_ms is not None and len(ppi_intervals_ms) >= 16:
            try:
                t_cum = np.cumsum(np.array(ppi_intervals_ms)) / 1000.0
                t_uniform = np.arange(0, t_cum[-1], 0.25)
                ppi_uniform = np.interp(t_uniform, t_cum, ppi_intervals_ms)
                freqs, psd = signal.welch(ppi_uniform - np.mean(ppi_uniform), fs=4.0, nperseg=min(len(ppi_uniform), 64))
                lf_mask = (freqs >= 0.04) & (freqs < 0.15)
                hf_mask = (freqs >= 0.15) & (freqs <= 0.40)
                lf_p = float(np.trapz(psd[lf_mask], freqs[lf_mask])) if np.any(lf_mask) else 450.0
                hf_p = float(np.trapz(psd[hf_mask], freqs[hf_mask])) if np.any(hf_mask) else 350.0
            except Exception:
                pass

        if hf_p <= 1e-4:
            hf_p = 100.0
        lf_hf = float(np.clip(lf_p / max(1.0, hf_p), 0.2, 8.0))

        # 3. Parasympathetic Recovery Score (PNS Index 0-100)
        pns_score = float(np.clip((math.log(rmssd_clamped) - 1.5) / 2.7 * 100.0, 5.0, 98.0))

        # 4. Sympathetic Arousal Score (SNS Index 0-100)
        sns_linear = 0.60 * (baevsky_si / 300.0) + 0.40 * (lf_hf / 3.0)
        sns_score = float(np.clip(sns_linear * 65.0, 5.0, 98.0))

        # 5. Pulse-Respiration Quotient (PRQ = HR / RR) - Synchronized!
        if rr_rpm > 0 and confidence_rr >= 0.40:
            prq = float(round(hr_bpm / rr_rpm, 2))
        else:
            prq = None

        # 6. Composite Physiological Stress (0-100)
        ln_rmssd = math.log(rmssd_clamped)
        rmssd_factor = max(0.0, min(1.0, (4.4 - ln_rmssd) / 2.8))
        hr_factor = max(0.0, min(1.0, (hr_clamped - 55.0) / 45.0))

        if rr_rpm > 0 and confidence_rr >= 0.40:
            rr_factor = max(0.0, min(1.0, (rr_rpm - 12.0) / 14.0))
            composite_stress = 0.40 * rmssd_factor + 0.35 * hr_factor + 0.15 * rr_factor + 0.10 * (sns_score / 100.0)
        else:
            composite_stress = 0.50 * rmssd_factor + 0.35 * hr_factor + 0.15 * (sns_score / 100.0)

        stress_score = float(round(composite_stress * 100.0, 1))

        if stress_score < 35.0:
            level = "Low"
        elif stress_score < 65.0:
            level = "Moderate"
        else:
            level = "High"

        confidence = float(round(min(confidence_hr, confidence_prv), 3))

        return StressResult(
            stress_score=stress_score,
            stress_level=level,
            baevsky_stress_index=float(round(baevsky_si, 1)),
            parasympathetic_score=float(round(pns_score, 1)),
            sympathetic_score=float(round(sns_score, 1)),
            pulse_respiration_quotient=prq,
            lf_power=float(round(lf_p, 1)),
            hf_power=float(round(hf_p, 1)),
            lf_hf_ratio=float(round(lf_hf, 2)),
            ans_balance_ratio=round(float(rmssd_clamped / (hr_clamped + 1e-4)), 3),
            confidence=confidence,
            is_valid=True,
            disclaimer="Physiological autonomic indicator — not a clinical diagnosis.",
            rejection_reason=None,
        )
