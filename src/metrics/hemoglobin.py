import numpy as np
from core.vitals.hemoglobin import HemoglobinEngine, HemoglobinResult

def estimate_hemoglobin(rgb_temporal: np.ndarray, is_male: bool = True, sqi_score: float = 80.0) -> HemoglobinResult:
    engine = HemoglobinEngine()
    return engine.estimate_hemoglobin(rgb_temporal, is_male=is_male, sqi_score=sqi_score)

__all__ = ['HemoglobinEngine', 'HemoglobinResult', 'estimate_hemoglobin']
