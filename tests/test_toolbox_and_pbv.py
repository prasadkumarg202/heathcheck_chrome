import numpy as np
import pytest
from core.rppg.pbv import PBVAlgorithm
from core.rppg.lgi import LGIAlgorithm
from core.config import AuraPulseConfig
from research.benchmarks.evaluator import VitalsBenchmarkEvaluator


def test_pbv_and_lgi_algorithms():
    t = np.linspace(0, 10, 300)
    pulse = 0.02 * np.sin(2 * np.pi * 1.25 * t)  # 75 BPM

    red = 140.0 + 4.0 * pulse + np.random.normal(0, 0.05, len(t))
    green = 110.0 + 18.0 * pulse + np.random.normal(0, 0.05, len(t))
    blue = 90.0 + 3.0 * pulse + np.random.normal(0, 0.05, len(t))

    rgb_matrix = np.column_stack([red, green, blue])

    # 1. Test PBV Algorithm
    pbv = PBVAlgorithm()
    bvp_pbv = pbv.extract(rgb_matrix, fs=30.0)
    assert len(bvp_pbv) == len(t)
    assert np.std(bvp_pbv) > 0.0

    # 2. Test LGI Algorithm
    lgi = LGIAlgorithm()
    bvp_lgi = lgi.extract(rgb_matrix, fs=30.0)
    assert len(bvp_lgi) == len(t)
    assert np.std(bvp_lgi) > 0.0


def test_config_system():
    cfg = AuraPulseConfig()
    d = cfg.to_dict()
    assert "toolbox_mode" in d
    assert "model" in d
    assert "unsupervised" in d
    assert "POS" in cfg.unsupervised.methods
    assert "PBV" in cfg.unsupervised.methods
    assert "LGI" in cfg.unsupervised.methods


def test_vitals_benchmark_evaluator():
    gt = np.array([72.0, 75.0, 80.0, 85.0, 90.0])
    pred = np.array([72.5, 74.8, 81.0, 84.5, 89.2])

    report = VitalsBenchmarkEvaluator.evaluate_predictions(gt, pred)
    assert report.mae < 1.0
    assert report.rmse < 1.0
    assert report.pearson_r > 0.98
    d = report.to_dict()
    assert "MAE" in d
    assert "RMSE" in d
