import numpy as np
from core.rppg.chrom import CHROMAlgorithm

def extract_bvp_chrom(rgb_series: np.ndarray, fs: float = 30.0) -> np.ndarray:
    algo = CHROMAlgorithm()
    return algo.extract(rgb_series, fs=fs)

__all__ = ['CHROMAlgorithm', 'extract_bvp_chrom']
