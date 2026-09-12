"""
Physiological Stress Index Engine for AuraPulse.
Computes an autonomic nervous system balance indicator derived from resting HR,
PRV (RMSSD/SDNN), and respiratory rate.

SCIENTIFIC PRINCIPLE:
This metric evaluates acute physiological autonomic arousal. It is NOT a clinical,
psychological, or psychiatric diagnosis.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class StressResult:
    stress_score: float          # 0 to 100 (0 = very relaxed, 100 = high autonomic load)
    stress_level: str            # "Low", "Moderate", "High"
    confidence: float            # 0.0 to 1.0
    ans_balance_ratio: float     # Sympathovagal balance proxy
    is_valid: bool
    disclaimer: str = "Physiological autonomic indicator — not a medical or psychological diagnosis."
    rejection_reason: Optional[str] = None


class StressEngine:
    """
    Evaluates autonomic stress load from fused physiological biomarkers.
    """

    def compute_stress_index(
        self,
        hr_bpm: float,
        rmssd_ms: float,
        rr_rpm: float,
        confidence_hr: float,
        confidence_prv: float,
    ) -> StressResult:
        """
        Computes Physiological Stress Index (0 to 100).
        High parasympathetic tone (high RMSSD, low resting HR) -> Low stress score.
        High sympathetic tone (low RMSSD, elevated HR) -> High stress score.
        """
        if hr_bpm <= 0 or rmssd_ms <= 0:
            return StressResult(
                stress_score=0.0,
                stress_level="Unknown",
                confidence=0.0,
                ans_balance_ratio=0.0,
                is_valid=False,
                rejection_reason="INSUFFICIENT_VITAL_METRICS_FOR_STRESS",
            )

        # Baseline reference parameters
        rmssd_clamped = max(5.0, min(120.0, rmssd_ms))
        hr_clamped = max(45.0, min(140.0, hr_bpm))

        # RMSSD contribution (inverted: low RMSSD gives high score)
        ln_rmssd = math.log(rmssd_clamped)
        # ln(5) ~ 1.6 (High stress), ln(80) ~ 4.4 (Low stress)
        rmssd_factor = max(0.0, min(1.0, (4.4 - ln_rmssd) / 2.8))

        # HR contribution: resting 60 -> 0, resting 100 -> 1.0
        hr_factor = max(0.0, min(1.0, (hr_clamped - 55.0) / 45.0))

        # RR contribution if available
        if rr_rpm > 0:
            rr_factor = max(0.0, min(1.0, (rr_rpm - 12.0) / 14.0))
            composite_stress = 0.50 * rmssd_factor + 0.35 * hr_factor + 0.15 * rr_factor
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
            confidence=confidence,
            ans_balance_ratio=round(float(rmssd_clamped / (hr_clamped + 1e-4)), 3),
            is_valid=True,
            disclaimer="Physiological autonomic indicator — not a medical or psychological diagnosis.",
            rejection_reason=None,
        )
