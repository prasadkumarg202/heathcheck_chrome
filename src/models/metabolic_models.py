from core.analytics.metabolic_models import (
    MetabolicRiskEngine,
    MetabolicRiskResult
)

def estimate_metabolic_risks(
    rmssd_ms: float = 35.0,
    lf_hf_ratio: float = 1.5,
    systolic_bp: float = 120.0,
    age: float = 35.0,
    bmi: float = 23.5,
    stiffness_index: float = 7.5,
    is_diabetic_history: bool = False,
    confidence_inputs: float = 0.85
) -> MetabolicRiskResult:
    engine = MetabolicRiskEngine()
    return engine.estimate_metabolic_risks(
        rmssd_ms=rmssd_ms,
        lf_hf_ratio=lf_hf_ratio,
        systolic_bp=systolic_bp,
        age=age,
        bmi=bmi,
        stiffness_index=stiffness_index,
        is_diabetic_history=is_diabetic_history,
        confidence_inputs=confidence_inputs
    )

__all__ = [
    'MetabolicRiskEngine',
    'MetabolicRiskResult',
    'estimate_metabolic_risks'
]
