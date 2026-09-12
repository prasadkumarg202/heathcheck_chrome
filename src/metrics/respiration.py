import numpy as np
from core.vitals.respiration import RespirationEngine, RespirationResult

def estimate_respiration_rate(signal: np.ndarray, fs: float = 30.0, sqi_score: float = 100.0) -> RespirationResult:
    engine = RespirationEngine()
    return engine.estimate_rr(signal, fs=fs, sqi_score=sqi_score)

__all__ = ['RespirationEngine', 'RespirationResult', 'estimate_respiration_rate']
