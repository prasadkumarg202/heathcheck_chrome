import numpy as np
from core.rppg.green import GreenAlgorithm

def extract_bvp_green(rgb_series: np.ndarray) -> np.ndarray:
    algo = GreenAlgorithm()
    return algo.extract_green(rgb_series)

__all__ = ['GreenAlgorithm', 'extract_bvp_green']
