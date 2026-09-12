import numpy as np
from core.rppg.ica import ICAAlgorithm

def extract_bvp_ica(rgb_series: np.ndarray, fs: float = 30.0) -> np.ndarray:
    algo = ICAAlgorithm()
    return algo.extract(rgb_series, fs=fs)

__all__ = ['ICAAlgorithm', 'extract_bvp_ica']
