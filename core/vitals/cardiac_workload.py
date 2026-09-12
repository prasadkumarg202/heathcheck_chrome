"""
Cardiac Workload (Rate-Pressure Product / RPP) Engine for AuraPulse.
Evaluates Myocardial Oxygen Consumption and Cardiac Stress:
RPP = (HR * SBP) / 100
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, Optional, Any


@dataclass
class CardiacWorkloadResult:
    rpp_score: float                   # Rate-Pressure Product
    workload_category: str             # Low, Normal, Moderate, High, Very High
    myocardial_load_index: float       # 0-100 normalized
    confidence: float
    is_valid: bool
    rejection_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class CardiacWorkloadEngine:
    """
    Computes Myocardial Workload and Oxygen Demand from fused HR and SBP.
    """

    def compute_workload(
        self,
        hr_bpm: float,
        systolic_bp: float,
        confidence_hr: float = 0.8,
        confidence_bp: float = 0.8,
    ) -> CardiacWorkloadResult:
        if hr_bpm <= 0 or systolic_bp <= 0:
            return CardiacWorkloadResult(
                rpp_score=0.0,
                workload_category="Unknown",
                myocardial_load_index=0.0,
                confidence=0.0,
                is_valid=False,
                rejection_reason="INSUFFICIENT_METRICS_FOR_WORKLOAD",
            )

        rpp = (hr_bpm * systolic_bp) / 100.0

        if rpp < 75.0:
            cat = "Low"
        elif rpp <= 100.0:
            cat = "Normal"
        elif rpp <= 125.0:
            cat = "Moderate"
        elif rpp <= 150.0:
            cat = "High"
        else:
            cat = "Very High"

        norm_load = max(0.0, min(100.0, (rpp - 60.0) / 1.0))
        conf = min(confidence_hr, confidence_bp)

        return CardiacWorkloadResult(
            rpp_score=float(round(rpp, 1)),
            workload_category=cat,
            myocardial_load_index=float(round(norm_load, 1)),
            confidence=float(round(conf, 3)),
            is_valid=True,
            rejection_reason=None,
        )
