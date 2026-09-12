"""
Hemoglobin (Hb) Estimation Engine for AuraPulse.
Implements optical density differential absorption model from multi-spectral rPPG signals.

SCIENTIFIC PRINCIPLE:
Oxy- and deoxy-hemoglobin exhibit strong differential light absorption characteristics
in the green (~520-560 nm) and red (~620-660 nm) optical bands. By computing the pulsatile-to-static
ratio (AC/DC) across multi-spectral channels, total blood hemoglobin concentration is estimated.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, Optional, Any
import numpy as np


@dataclass
class HemoglobinResult:
    hb_g_dl: float               # Estimated total hemoglobin in g/dL
    reference_range: str         # e.g. "14.0 - 18.0 g/dL (Male)" or "12.0 - 16.0 g/dL (Female)"
    category: str                # "Normal", "Low (Anemia Warning)", "Elevated"
    attenuation_ratio: float     # Optical differential extinction ratio
    confidence: float            # 0.0 to 1.0
    is_valid: bool
    disclaimer: str = "Research & wellness indicator — not intended for clinical anemia diagnosis."
    rejection_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class HemoglobinEngine:
    """
    Non-invasive multi-spectral optical hemoglobin concentration estimator.
    """

    def __init__(self, calib_slope: float = 4.2, calib_intercept: float = 10.8):
        self.calib_slope = calib_slope
        self.calib_intercept = calib_intercept

    def estimate_hemoglobin(
        self,
        rgb_temporal: np.ndarray,
        is_male: bool = True,
        sqi_score: float = 80.0,
    ) -> HemoglobinResult:
        """
        Estimates blood hemoglobin concentration from facial ROI RGB temporal matrix.
        rgb_temporal shape: (N, 3) where columns are [R, G, B].
        """
        if rgb_temporal is None or len(rgb_temporal) < 60:
            return HemoglobinResult(
                hb_g_dl=0.0,
                reference_range="14.0 - 18.0 g/dL" if is_male else "12.0 - 16.0 g/dL",
                category="Unknown",
                attenuation_ratio=0.0,
                confidence=0.0,
                is_valid=False,
                rejection_reason="INSUFFICIENT_SAMPLES_FOR_HEMOGLOBIN",
            )

        # 1. Compute AC and DC components per channel
        dc_r = np.mean(rgb_temporal[:, 0]) + 1e-6
        ac_r = np.std(rgb_temporal[:, 0])
        dc_g = np.mean(rgb_temporal[:, 1]) + 1e-6
        ac_g = np.std(rgb_temporal[:, 1])

        # Pulsatile modulation ratios
        mod_r = ac_r / dc_r
        mod_g = ac_g / dc_g

        # Differential attenuation extinction ratio
        ratio = mod_g / (mod_r + 1e-6)
        ratio_clamped = float(np.clip(ratio, 0.4, 2.2))

        # Sex-adjusted baseline calibration
        gender_offset = 1.6 if is_male else 0.0
        hb_est = self.calib_intercept + self.calib_slope * (ratio_clamped - 0.75) + gender_offset
        
        # Physiological bounding (8.0 to 20.0 g/dL)
        hb_final = float(np.clip(hb_est, 8.5, 19.5))

        # Reference ranges: Men 14-18 g/dL, Women 12-16 g/dL
        low_thresh = 13.5 if is_male else 12.0
        high_thresh = 18.0 if is_male else 16.0
        ref_str = "14.0 - 18.0 g/dL (Male)" if is_male else "12.0 - 16.0 g/dL (Female)"

        if hb_final < low_thresh:
            cat = "Low (Anemia Warning)"
        elif hb_final > high_thresh:
            cat = "Elevated"
        else:
            cat = "Normal"

        conf = min(0.92, max(0.20, (sqi_score / 100.0) * 0.90))

        return HemoglobinResult(
            hb_g_dl=float(round(hb_final, 1)),
            reference_range=ref_str,
            category=cat,
            attenuation_ratio=float(round(ratio_clamped, 3)),
            confidence=float(round(conf, 3)),
            is_valid=True,
            disclaimer="Research & wellness indicator — not intended for clinical anemia diagnosis.",
            rejection_reason=None,
        )
