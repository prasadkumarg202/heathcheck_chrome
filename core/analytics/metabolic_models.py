"""
Metabolic Risk Analytics Suite for AuraPulse.
Features:
1. Multi-variate Fasting Blood Glucose Risk Estimation (Normal <100, Impaired 100-125, Elevated >125 mg/dL)
2. Multi-variate Glycated Hemoglobin (HbA1c) Risk Estimation (Normal <5.7%, Prediabetes 5.7-6.4%, Diabetes >6.4%)
3. Autonomic-Hemodynamic Metabolic Score derived from [RMSSD, LF/HF, SBP, Age, BMI, Stiffness Index]
"""

from __future__ import annotations
import math
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
import numpy as np


@dataclass
class MetabolicRiskResult:
    estimated_fbg_mg_dl: float         # Estimated fasting blood glucose proxy in mg/dL
    fbg_category: str                  # "Normal (<100 mg/dL)", "Impaired (100-125 mg/dL)", "Elevated (>125 mg/dL)"
    fbg_status: str                    # "Normal", "Impaired", "Elevated"
    estimated_hba1c_pct: float         # Estimated HbA1c in %
    hba1c_category: str                # "Normal (<5.7%)", "Prediabetes (5.7-6.4%)", "Diabetes (>6.4%)"
    hba1c_status: str                  # "Normal", "Prediabetes", "Diabetes"
    metabolic_score: float             # 0 to 100 composite metabolic risk score
    risk_level: str                    # "Low Risk", "Moderate Risk", "High Risk"
    confidence: float                  # 0.0 to 1.0
    key_factors: List[str]
    is_valid: bool
    disclaimer: str = "Wellness & physiological risk indicator — not a clinical lab diagnosis for diabetes."
    rejection_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MetabolicRiskEngine:
    """
    Multi-variate predictive engine evaluating glycemic status from autonomic tone,
    arterial stiffness, hemodynamics, and anthropometrics.
    """

    def estimate_metabolic_risks(
        self,
        rmssd_ms: float = 35.0,
        lf_hf_ratio: float = 1.5,
        systolic_bp: float = 120.0,
        age: float = 35.0,
        bmi: float = 23.5,
        stiffness_index: float = 7.5,
        is_diabetic_history: bool = False,
        confidence_inputs: float = 0.85,
    ) -> MetabolicRiskResult:
        """
        Synthesizes autonomic impairment (depressed RMSSD, elevated LF/HF),
        vascular stiffness, systolic hypertension, and BMI into glycemic risk projections.
        """
        # Feature normalizations
        age_z = max(0.0, (age - 25.0) / 40.0)
        bmi_z = max(0.0, (bmi - 21.0) / 10.0)
        bp_z = max(0.0, (systolic_bp - 115.0) / 30.0)
        stiff_z = max(0.0, (stiffness_index - 6.5) / 4.0)
        hrv_penalty = max(0.0, (40.0 - min(80.0, rmssd_ms)) / 30.0)
        symp_penalty = max(0.0, (lf_hf_ratio - 1.2) / 2.0)

        # Multi-variate risk index (0.0 to 1.0)
        linear_risk = (
            0.25 * bmi_z +
            0.20 * age_z +
            0.18 * bp_z +
            0.14 * stiff_z +
            0.13 * hrv_penalty +
            0.10 * symp_penalty
        )

        if is_diabetic_history:
            linear_risk += 0.45

        # 1. Fasting Blood Glucose (FBG) Estimation (mg/dL)
        # Baseline normal: ~90 mg/dL. Up to ~165 mg/dL.
        fbg_est = 88.0 + linear_risk * 48.0
        fbg_est = float(np.clip(fbg_est, 75.0, 195.0))

        if fbg_est < 100.0:
            fbg_cat = "Normal (<100 mg/dL)"
            fbg_stat = "Normal"
        elif fbg_est <= 125.0:
            fbg_cat = "Impaired (100-125 mg/dL)"
            fbg_stat = "Impaired"
        else:
            fbg_cat = "Elevated (>125 mg/dL)"
            fbg_stat = "Elevated"

        # 2. HbA1c Estimation (%)
        # Baseline normal: ~5.1%. Up to ~8.5%.
        # Relationship: ADAG formula eAG = 28.7 * A1c - 46.7  =>  A1c = (eAG + 46.7) / 28.7
        hba1c_est = (fbg_est + 46.7) / 28.7
        hba1c_est = float(np.clip(hba1c_est, 4.5, 9.5))

        if hba1c_est < 5.7:
            hba1c_cat = "Normal (<5.7%)"
            hba1c_stat = "Normal"
        elif hba1c_est <= 6.4:
            hba1c_cat = "Prediabetes (5.7-6.4%)"
            hba1c_stat = "Prediabetes"
        else:
            hba1c_cat = "Diabetes (>6.4%)"
            hba1c_stat = "Diabetes"

        # Composite metabolic score (0 - 100)
        met_score = float(np.clip(linear_risk * 100.0, 5.0, 95.0))

        if met_score < 30.0:
            risk_lvl = "Low Risk"
        elif met_score < 60.0:
            risk_lvl = "Moderate Risk"
        else:
            risk_lvl = "High Risk"

        factors = []
        if bmi >= 25.0:
            factors.append(f"Elevated BMI ({bmi:.1f})")
        if systolic_bp >= 130.0:
            factors.append(f"Elevated Systolic BP ({systolic_bp:.0f} mmHg)")
        if rmssd_ms < 25.0:
            factors.append("Low Autonomic Vagal Activity")
        if lf_hf_ratio > 2.0:
            factors.append("Sympathetic Hyperarousal")
        if stiffness_index > 8.5:
            factors.append("Arterial Wall Stiffening")
        if is_diabetic_history:
            factors.append("Known History of Glycemic Imbalance")
        if not factors:
            factors.append("Optimal Metabolic & Autonomic Profile")

        return MetabolicRiskResult(
            estimated_fbg_mg_dl=float(round(fbg_est, 1)),
            fbg_category=fbg_cat,
            fbg_status=fbg_stat,
            estimated_hba1c_pct=float(round(hba1c_est, 1)),
            hba1c_category=hba1c_cat,
            hba1c_status=hba1c_stat,
            metabolic_score=float(round(met_score, 1)),
            risk_level=risk_lvl,
            confidence=float(round(confidence_inputs * 0.88, 3)),
            key_factors=factors,
            is_valid=True,
            disclaimer="Wellness & physiological risk indicator — not a clinical lab diagnosis for diabetes.",
            rejection_reason=None,
        )


# Alias for backward compatibility
MetabolicAnalyticsEngine = MetabolicRiskEngine

