import numpy as np
from core.quality.sqi import SignalQualityEngine, SQIResult

def evaluate_sqi(bvp_signal: np.ndarray, fs: float = 30.0) -> SQIResult:
    engine = SignalQualityEngine()
    return engine.compute_sqi(bvp_signal, fs=fs)

__all__ = ['SignalQualityEngine', 'SQIResult', 'evaluate_sqi']
