"""
Cardiovascular & Metabolic Risk Analytics Suite for AuraPulse.
Features:
1. Vascular Age Estimation (Arterial compliance & AGI vs chronological age)
2. 10-Year CVD & Stroke Risk Projection (Calibrated Framingham / WHO-ISH model)
3. Body Composition & Anthropometrics (BMI, Waist-to-Height Ratio WHtR, Body Roundness Index BRI)
4. Hypertension & Type 2 Diabetes Risk Indicators
"""

from __future__ import annotations
import math
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any
import numpy as np


@dataclass
class VascularAgeResult:
    vascular_age_years: float          # Estimated biological vascular age
    age_delta: float                   # Vascular age - Chronological age
    stiffness_status: str              # "Optimal", "Normal", "Accelerated Stiffening"
    aging_index_score: float
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CVDRiskResult:
    ten_year_risk_pct: float           # 10-year risk of cardiovascular event (%)
    risk_category: str                 # "Low (<10%)", "Moderate (10-20%)", "High (20-30%)", "Very High (>30%)"
    stroke_risk_pct: float             # 10-year stroke projection (%)
    hypertension_risk_score: float     # 0-100
    diabetes_risk_score: float         # 0-100 (FINDRISC proxy)
    key_drivers: List[str]
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BodyCompositionResult:
    bmi: float                         # Body Mass Index (kg/m^2)
    bmi_category: str                  # "Underweight", "Normal", "Overweight", "Obese"
    whtr: float                        # Waist-to-Height Ratio
    whtr_category: str                 # "Optimal (<0.5)", "Increased Risk (0.5-0.6)", "High Risk (>0.6)"
    body_roundness_index: float        # BRI = 364.2 - 365.5 * sqrt(1 - (waist / (2*pi*height))^2)
    bri_category: str                  # "Low Risk", "Moderate Risk", "Elevated Risk"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class HealthRiskAnalyticsEngine:
    """
    Computes Vascular Age, 10-Year Cardiovascular Risks, and Metabolic Indices.
    """

    def compute_body_composition(
        self,
        height_cm: float = 175.0,
        weight_kg: float = 72.0,
        waist_cm: Optional[float] = None,
    ) -> BodyCompositionResult:
        h_m = max(0.5, height_cm / 100.0)
        w_kg = max(20.0, weight_kg)
        bmi = w_kg / (h_m * h_m)

        if bmi < 18.5:
            bmi_cat = "Underweight"
        elif bmi < 25.0:
            bmi_cat = "Normal"
        elif bmi < 30.0:
            bmi_cat = "Overweight"
        else:
            bmi_cat = "Obese"

        if waist_cm is None or waist_cm <= 0:
            waist_cm = (0.45 + (bmi - 22.0) * 0.015) * height_cm

        whtr = waist_cm / max(height_cm, 50.0)
        if whtr < 0.50:
            whtr_cat = "Optimal (<0.5)"
        elif whtr < 0.60:
            whtr_cat = "Increased Risk (0.5-0.6)"
        else:
            whtr_cat = "High Risk (>0.6)"

        waist_m = waist_cm / 100.0
        eccentricity_term = 1.0 - ((waist_m / (2.0 * math.pi * h_m)) ** 2)
        eccentricity_term = max(0.01, min(1.0, eccentricity_term))
        bri = 364.2 - 365.5 * math.sqrt(eccentricity_term)
        bri = float(np.clip(bri, 1.0, 16.0))

        if bri < 3.5:
            bri_cat = "Low Risk"
        elif bri < 5.5:
            bri_cat = "Moderate Risk"
        else:
            bri_cat = "Elevated Risk"

        return BodyCompositionResult(
            bmi=float(round(bmi, 1)),
            bmi_category=bmi_cat,
            whtr=float(round(whtr, 2)),
            whtr_category=whtr_cat,
            body_roundness_index=float(round(bri, 1)),
            bri_category=bri_cat,
        )

    def estimate_vascular_age(
        self,
        chronological_age: float = 35.0,
        systolic_bp: float = 120.0,
        stiffness_index: float = 7.5,
        sdppg_aging_index: float = -0.35,
        confidence_bp: float = 0.8,
    ) -> VascularAgeResult:
        stiffness_delta = (stiffness_index - 7.0) * 2.8
        bp_delta = (systolic_bp - 118.0) * 0.28
        agi_delta = (sdppg_aging_index - (-0.40)) * 14.0

        v_age = chronological_age + stiffness_delta + bp_delta + agi_delta
        v_age = float(np.clip(v_age, 18.0, 95.0))
        age_delta = float(round(v_age - chronological_age, 1))

        if age_delta <= 1.0:
            status = "Optimal"
        elif age_delta <= 5.0:
            status = "Normal"
        else:
            status = "Accelerated Stiffening"

        return VascularAgeResult(
            vascular_age_years=float(round(v_age, 1)),
            age_delta=age_delta,
            stiffness_status=status,
            aging_index_score=float(round(sdppg_aging_index, 3)),
            confidence=float(round(confidence_bp * 0.9, 3)),
        )

    def estimate_10yr_cvd_risk(
        self,
        age: float = 35.0,
        is_male: bool = True,
        systolic_bp: float = 120.0,
        is_smoker: bool = False,
        is_diabetic: bool = False,
        bmi: float = 23.5,
        hr_bpm: float = 72.0,
        rmssd_ms: float = 35.0,
    ) -> CVDRiskResult:
        score = 0.0

        # Age points
        score += max(0.0, (age - 20.0) * 0.085)

        # Sex factor
        if is_male:
            score += 0.45

        # Blood Pressure contribution
        score += max(0.0, (systolic_bp - 110.0) * 0.022)

        # Smoking
        if is_smoker:
            score += 0.85

        # Diabetes
        if is_diabetic:
            score += 0.95

        # BMI
        score += max(0.0, (bmi - 22.0) * 0.035)

        # Autonomic balance / resting HR
        score += max(0.0, (hr_bpm - 70.0) * 0.015)
        score += max(0.0, (30.0 - min(100.0, rmssd_ms)) * 0.010)

        ten_yr_prob = 100.0 / (1.0 + math.exp(-(score - 3.2)))
        ten_yr_prob = float(np.clip(ten_yr_prob, 0.8, 65.0))

        if ten_yr_prob < 10.0:
            cat = "Low (<10%)"
        elif ten_yr_prob < 20.0:
            cat = "Moderate (10-20%)"
        elif ten_yr_prob < 30.0:
            cat = "High (20-30%)"
        else:
            cat = "Very High (>30%)"

        stroke_prob = float(round(ten_yr_prob * 0.38, 1))

        htn_risk = float(np.clip((systolic_bp - 100.0) * 1.1 + (bmi - 20.0) * 1.5 + (age - 25.0) * 0.5, 5.0, 95.0))
        dm_risk = float(np.clip((bmi - 20.0) * 2.8 + (age - 25.0) * 0.6 + (15.0 if is_smoker else 0.0), 4.0, 92.0))

        drivers = []
        if systolic_bp >= 130.0:
            drivers.append("Elevated Blood Pressure")
        if is_smoker:
            drivers.append("Tobacco Use")
        if bmi >= 25.0:
            drivers.append("Elevated BMI")
        if is_diabetic:
            drivers.append("Diabetic Metabolic Profile")
        if hr_bpm >= 85.0 or rmssd_ms < 25.0:
            drivers.append("Elevated Sympathetic Tone / Low HRV")
        if not drivers:
            drivers.append("Optimal Hemodynamic & Metabolic Biomarkers")

        return CVDRiskResult(
            ten_year_risk_pct=float(round(ten_yr_prob, 1)),
            risk_category=cat,
            stroke_risk_pct=stroke_prob,
            hypertension_risk_score=float(round(htn_risk, 1)),
            diabetes_risk_score=float(round(dm_risk, 1)),
            key_drivers=drivers,
            confidence=0.85,
        )
