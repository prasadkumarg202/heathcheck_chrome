import numpy as np
from core.rppg.pos import POSAlgorithm

def extract_bvp_pos(rgb_series: np.ndarray, fs: float = 30.0) -> np.ndarray:
    algo = POSAlgorithm()
    return algo.extract(rgb_series, fs=fs)

__all__ = ['POSAlgorithm', 'extract_bvp_pos']
