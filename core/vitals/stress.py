"""
Advanced Physiological Stress & Autonomic Recovery Engine for AuraPulse.
Features:
1. Baevsky's Stress Index (SI = AMo / (2 * Mo * MxDMn))
2. Parasympathetic Activity & Recovery Score (PNS Index)
3. Pulse-Respiration Quotient (PRQ = HR / RR)
4. Composite Physiological Stress Score (0-100)
"""

from __future__ import annotations
import math
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any
import numpy as np


@dataclass
class StressResult:
    stress_score: float                # 0 to 100 (0 = relaxed, 100 = high autonomic load)
    stress_level: str                  # Low, Moderate, High
    baevsky_stress_index: float        # Classical Baevsky SI (norm 50-150)
    parasympathetic_score: float       # PNS tone & recovery score (0-100)
    pulse_respiration_quotient: float  # PRQ = HR / RR (norm ~4.0 - 5.0)
    ans_balance_ratio: float           # Sympathovagal balance proxy
    confidence: float                  # 0.0 to 1.0
    is_valid: bool
    disclaimer: str = "Physiological autonomic indicator — not a clinical diagnosis."
    rejection_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class StressEngine:
    """
    Evaluates autonomic stress load, Baevsky Index, PNS tone, and PRQ.
    """

    def compute_stress_index(
        self,
        hr_bpm: float,
        rmssd_ms: float,
        rr_rpm: float,
        confidence_hr: float = 0.8,
        confidence_prv: float = 0.8,
        ppi_intervals_ms: Optional[List[float]] = None,
    ) -> StressResult:
        if hr_bpm <= 0 or rmssd_ms <= 0:
            return StressResult(
                stress_score=0.0,
                stress_level="Unknown",
                baevsky_stress_index=0.0,
                parasympathetic_score=0.0,
                pulse_respiration_quotient=0.0,
                ans_balance_ratio=0.0,
                confidence=0.0,
                is_valid=False,
                rejection_reason="INSUFFICIENT_VITAL_METRICS_FOR_STRESS",
            )

        rmssd_clamped = max(5.0, min(120.0, rmssd_ms))
        hr_clamped = max(45.0, min(140.0, hr_bpm))

        # 1. Classical Baevsky Stress Index calculation
        if ppi_intervals_ms is not None and len(ppi_intervals_ms) >= 10:
            ppis = np.array(ppi_intervals_ms)
            bins = np.arange(np.min(ppis), np.max(ppis) + 50, 50)
            counts, bin_edges = np.histogram(ppis, bins=bins)
            max_bin_idx = np.argmax(counts)
            mo_s = (bin_edges[max_bin_idx] + 25.0) / 1000.0
            amo_pct = (counts[max_bin_idx] / len(ppis)) * 100.0
            mx_d_mn_s = (np.max(ppis) - np.min(ppis)) / 1000.0
            baevsky_si = amo_pct / (2.0 * max(0.3, mo_s) * max(0.05, mx_d_mn_s))
        else:
            approx_mo = 60.0 / hr_clamped
            approx_amo = min(80.0, max(20.0, 30.0 + (hr_clamped - 60.0) * 0.8))
            approx_range = max(0.06, min(0.40, rmssd_clamped / 250.0))
            baevsky_si = approx_amo / (2.0 * approx_mo * approx_range)

        baevsky_si = float(np.clip(baevsky_si, 15.0, 950.0))

        # 2. Parasympathetic Recovery Score (PNS Index 0-100)
        pns_score = float(np.clip((math.log(rmssd_clamped) - 1.6) / 2.8 * 100.0, 5.0, 98.0))

        # 3. Pulse-Respiration Quotient (PRQ = HR / RR)
        if rr_rpm > 0:
            prq = float(round(hr_bpm / rr_rpm, 2))
        else:
            prq = 4.2

        # 4. Composite Physiological Stress (0-100)
        ln_rmssd = math.log(rmssd_clamped)
        rmssd_factor = max(0.0, min(1.0, (4.4 - ln_rmssd) / 2.8))
        hr_factor = max(0.0, min(1.0, (hr_clamped - 55.0) / 45.0))

        if rr_rpm > 0:
            rr_factor = max(0.0, min(1.0, (rr_rpm - 12.0) / 14.0))
            composite_stress = 0.45 * rmssd_factor + 0.35 * hr_factor + 0.20 * rr_factor
        else:
            composite_stress = 0.60 * rmssd_factor + 0.40 * hr_factor

        stress_score = float(round(composite_stress * 100.0, 1))

        if stress_score < 35.0:
            level = "Low"
        elif stress_score < 68.0:
            level = "Moderate"
        else:
            level = "High"

        confidence = float(round(min(confidence_hr, confidence_prv), 3))

        return StressResult(
            stress_score=stress_score,
            stress_level=level,
            baevsky_stress_index=float(round(baevsky_si, 1)),
            parasympathetic_score=float(round(pns_score, 1)),
            pulse_respiration_quotient=prq,
            ans_balance_ratio=round(float(rmssd_clamped / (hr_clamped + 1e-4)), 3),
            confidence=confidence,
            is_valid=True,
            disclaimer="Physiological autonomic indicator — not a clinical diagnosis.",
            rejection_reason=None,
        )
