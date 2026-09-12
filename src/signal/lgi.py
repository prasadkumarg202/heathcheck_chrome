import numpy as np
from core.rppg.lgi import LGIAlgorithm

def extract_bvp_lgi(rgb_series: np.ndarray, fs: float = 30.0) -> np.ndarray:
    algo = LGIAlgorithm()
    return algo.extract(rgb_series, fs=fs)

__all__ = ['LGIAlgorithm', 'extract_bvp_lgi']
