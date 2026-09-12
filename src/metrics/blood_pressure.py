import numpy as np
from core.vitals.blood_pressure import BloodPressureEngine, BloodPressureResult

def estimate_blood_pressure(bvp_signal: np.ndarray, fs: float = 30.0, hr_bpm: float = 72.0) -> BloodPressureResult:
    engine = BloodPressureEngine()
    return engine.estimate_blood_pressure(bvp_signal, fs=fs, hr_bpm=hr_bpm, rmssd_ms=40.0)

__all__ = ['BloodPressureEngine', 'BloodPressureResult', 'estimate_blood_pressure']
