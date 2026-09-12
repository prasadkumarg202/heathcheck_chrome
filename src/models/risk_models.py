from core.analytics.risk_models import (
    HealthRiskAnalyticsEngine,
    VascularAgeResult,
    CVDRiskResult,
    BodyCompositionResult
)

def compute_body_composition(height_cm: float = 175.0, weight_kg: float = 72.0, waist_cm: float = None) -> BodyCompositionResult:
    engine = HealthRiskAnalyticsEngine()
    return engine.compute_body_composition(height_cm=height_cm, weight_kg=weight_kg, waist_cm=waist_cm)

def estimate_vascular_age(chronological_age: float = 35.0, systolic_bp: float = 120.0, stiffness_index: float = 7.5, sdppg_aging_index: float = -0.35, confidence_bp: float = 0.8) -> VascularAgeResult:
    engine = HealthRiskAnalyticsEngine()
    return engine.estimate_vascular_age(chronological_age=chronological_age, systolic_bp=systolic_bp, stiffness_index=stiffness_index, sdppg_aging_index=sdppg_aging_index, confidence_bp=confidence_bp)

def estimate_10yr_cvd_risk(age: float = 35.0, is_male: bool = True, systolic_bp: float = 120.0, is_smoker: bool = False, is_diabetic: bool = False, bmi: float = 23.5, hr_bpm: float = 72.0, rmssd_ms: float = 35.0) -> CVDRiskResult:
    engine = HealthRiskAnalyticsEngine()
    return engine.estimate_10yr_cvd_risk(age=age, is_male=is_male, systolic_bp=systolic_bp, is_smoker=is_smoker, is_diabetic=is_diabetic, bmi=bmi, hr_bpm=hr_bpm, rmssd_ms=rmssd_ms)

__all__ = [
    'HealthRiskAnalyticsEngine',
    'VascularAgeResult',
    'CVDRiskResult',
    'BodyCompositionResult',
    'compute_body_composition',
    'estimate_vascular_age',
    'estimate_10yr_cvd_risk'
]
