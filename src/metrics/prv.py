import numpy as np
from core.vitals.prv import PRVEngine, PRVResult

def compute_prv_metrics(bvp_signal: np.ndarray, fs: float = 30.0, sqi_score: float = 100.0) -> PRVResult:
    engine = PRVEngine(min_duration_s=4.0)
    return engine.compute_prv(bvp_signal, fs=fs, sqi_score=sqi_score)

__all__ = ['PRVEngine', 'PRVResult', 'compute_prv_metrics']
