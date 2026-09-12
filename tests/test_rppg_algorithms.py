"""
Unit tests for rPPG algorithms (POS, CHROM, Green, ICA).
"""

import numpy as np
import pytest
from core.rppg.pos import POSAlgorithm
from core.rppg.chrom import CHROMAlgorithm
from core.rppg.green import GreenAlgorithm
from core.rppg.ica import ICAAlgorithm
from research.synthetic.generator import SyntheticSignalGenerator


def test_pos_algorithm_extraction():
    gen = SyntheticSignalGenerator(fs=30.0)
    gt = gen.generate_signal(duration_s=15.0, hr_bpm=75.0, snr_db=20.0)
    rgb = gt.simulated_rgb["forehead"]

    pos = POSAlgorithm()
    bvp = pos.extract(rgb, fs=30.0)

    assert len(bvp) == len(rgb)
    assert not np.all(bvp == 0)
    # Correlation with true BVP
    corr = np.corrcoef(bvp[30:-30], gt.true_bvp[30:-30])[0, 1]
    assert abs(corr) > 0.70  # Strong correlation (POS may be inverted or non-inverted)


def test_chrom_algorithm_extraction():
    gen = SyntheticSignalGenerator(fs=30.0)
    gt = gen.generate_signal(duration_s=15.0, hr_bpm=80.0, snr_db=20.0)
    rgb = gt.simulated_rgb["forehead"]

    chrom = CHROMAlgorithm()
    bvp = chrom.extract(rgb, fs=30.0)

    assert len(bvp) == len(rgb)
    assert not np.all(bvp == 0)
    corr = np.corrcoef(bvp[30:-30], gt.true_bvp[30:-30])[0, 1]
    assert abs(corr) > 0.70


def test_green_channel_extraction():
    gen = SyntheticSignalGenerator(fs=30.0)
    gt = gen.generate_signal(duration_s=15.0, hr_bpm=70.0, snr_db=25.0)
    rgb = gt.simulated_rgb["forehead"]

    bvp = GreenAlgorithm.extract_green(rgb)
    assert len(bvp) == len(rgb)
    corr = np.corrcoef(bvp[30:-30], gt.true_bvp[30:-30])[0, 1]
    assert corr > 0.80


def test_ica_algorithm_extraction():
    gen = SyntheticSignalGenerator(fs=30.0)
    gt = gen.generate_signal(duration_s=15.0, hr_bpm=72.0, snr_db=20.0)
    rgb = gt.simulated_rgb["forehead"]

    ica = ICAAlgorithm()
    bvp = ica.extract(rgb, fs=30.0)
    assert len(bvp) == len(rgb)
    assert not np.all(bvp == 0)
