import numpy as np
from core.rppg.pbv import PBVAlgorithm

def extract_bvp_pbv(rgb_series: np.ndarray, fs: float = 30.0) -> np.ndarray:
    algo = PBVAlgorithm()
    return algo.extract(rgb_series, fs=fs)

__all__ = ['PBVAlgorithm', 'extract_bvp_pbv']
