import numpy as np
from core.vitals.spo2 import SpO2Engine, SpO2Result

def estimate_spo2(rgb_temporal: np.ndarray, fs: float = 30.0) -> SpO2Result:
    engine = SpO2Engine()
    return engine.estimate_spo2(rgb_temporal, fs=fs)

__all__ = ['SpO2Engine', 'SpO2Result', 'estimate_spo2']
