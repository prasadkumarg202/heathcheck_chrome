"""
Contactless Oxygen Saturation (SpO2) Estimation Engine for AuraPulse.
Derives SpO2 using multi-spectral Chromatic Ratio-of-Ratios:
R = (AC_red / DC_red) / (AC_blue / DC_blue)
SpO2 = A - B * R
"""

from __future__ import annotations
import math
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any
import numpy as np


@dataclass
class SpO2Result:
    spo2_pct: float                    # SpO2 in % (Normal 95-100%)
    category: str                      # Normal, Mild Hypoxia, Hypoxemia
    ratio_of_ratios: float             # Optical R value
    confidence: float                  # 0.0 to 1.0
    is_valid: bool
    rejection_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SpO2Engine:
    """
    On-device Contactless Oxygen Saturation (SpO2) estimation engine.
    """

    def __init__(self, calib_a: float = 110.0, calib_b: float = 14.5):
        self.calib_a = calib_a
        self.calib_b = calib_b

    def estimate_spo2(
        self,
        rgb_temporal: np.ndarray,
        fs: float = 30.0,
        sqi_score: float = 75.0,
    ) -> SpO2Result:
        if rgb_temporal is None or len(rgb_temporal) < int(fs * 4) or sqi_score < 25.0:
            return SpO2Result(
                spo2_pct=0.0,
                category="Unknown",
                ratio_of_ratios=0.0,
                confidence=0.0,
                is_valid=False,
                rejection_reason="INSUFFICIENT_SIGNAL_FOR_SPO2",
            )

        r_chan = rgb_temporal[:, 0].astype(np.float64)
        g_chan = rgb_temporal[:, 1].astype(np.float64)
        b_chan = rgb_temporal[:, 2].astype(np.float64)

        dc_r = np.mean(r_chan)
        dc_g = np.mean(g_chan)
        dc_b = np.mean(b_chan)

        if dc_r < 10 or dc_g < 10 or dc_b < 10:
            return SpO2Result(
                spo2_pct=0.0,
                category="Unknown",
                ratio_of_ratios=0.0,
                confidence=0.0,
                is_valid=False,
                rejection_reason="LOW_ILLUMINATION_FOR_SPO2",
            )

        ac_r = np.std(r_chan - dc_r)
        ac_g = np.std(g_chan - dc_g)
        ac_b = np.std(b_chan - dc_b)

        norm_ac_dc_r = ac_r / (dc_r + 1e-6)
        norm_ac_dc_bg = (ac_b / (dc_b + 1e-6) + ac_g / (dc_g + 1e-6)) / 2.0

        if norm_ac_dc_bg < 1e-5:
            r_ratio = 1.0
        else:
            r_ratio = norm_ac_dc_r / norm_ac_dc_bg

        raw_spo2 = self.calib_a - self.calib_b * r_ratio
        spo2_val = float(np.clip(raw_spo2, 88.0, 99.5))

        if spo2_val >= 95.0:
            cat = "Normal"
        elif spo2_val >= 90.0:
            cat = "Mild Hypoxia"
        else:
            cat = "Hypoxemia"

        confidence = float(np.clip((sqi_score / 100.0) * 0.85 + 0.10, 0.20, 0.95))

        return SpO2Result(
            spo2_pct=float(round(spo2_val, 1)),
            category=cat,
            ratio_of_ratios=float(round(r_ratio, 3)),
            confidence=float(round(confidence, 3)),
            is_valid=True,
            rejection_reason=None,
        )
