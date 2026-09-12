import numpy as np
from core.vitals.stress import StressEngine, StressResult

def compute_stress_index(bvp_signal: np.ndarray, fs: float = 30.0, hr_bpm: float = 72.0, rmssd_ms: float = 40.0) -> StressResult:
    engine = StressEngine()
    return engine.compute_stress_index(hr_bpm=hr_bpm, rmssd_ms=rmssd_ms, rr_rpm=16.0)

__all__ = ['StressEngine', 'StressResult', 'compute_stress_index']
