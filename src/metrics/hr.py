import numpy as np
from core.vitals.hr import HeartRateEngine, HRResult

def estimate_heart_rate(bvp_signal: np.ndarray, fs: float = 30.0, sqi_score: float = 100.0) -> HRResult:
    engine = HeartRateEngine()
    return engine.estimate_hr(bvp_signal, fs=fs, sqi_score=sqi_score)

__all__ = ['HeartRateEngine', 'HRResult', 'estimate_heart_rate']
