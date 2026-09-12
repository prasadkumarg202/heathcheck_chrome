"""
Unit tests for vital extraction engines: HR, Respiration, PRV, Stress.
"""

import numpy as np
import pytest
from core.vitals.hr import HeartRateEngine
from core.vitals.respiration import RespirationEngine
from core.vitals.prv import PRVEngine
from core.vitals.stress import StressEngine
from research.synthetic.generator import SyntheticSignalGenerator


def test_heart_rate_estimation_accuracy():
    hr_engine = HeartRateEngine()
    fs = 30.0

    for target_bpm in [60.0, 75.0, 90.0, 120.0]:
        t = np.linspace(0, 15, int(15 * fs), endpoint=False)
        sig = np.sin(2 * np.pi * (target_bpm / 60.0) * t) + 0.3 * np.sin(4 * np.pi * (target_bpm / 60.0) * t)
        
        res = hr_engine.estimate_hr(sig, fs=fs, sqi_score=90.0)
        assert res.is_valid
        # Error must be within 1.5 BPM on clean synthetic signals
        assert abs(res.hr_bpm - target_bpm) <= 1.5
        assert res.confidence > 0.60
        hr_engine.reset_tracking()


def test_respiration_rate_estimation():
    resp_engine = RespirationEngine()
    fs = 30.0
    duration_s = 20.0
    t = np.linspace(0, duration_s, int(duration_s * fs), endpoint=False)

    for target_rpm in [12.0, 16.0, 20.0]:
        # Cardiac carrier + slow respiratory modulation
        cardiac = np.sin(2 * np.pi * 1.2 * t)
        resp = 0.5 * np.sin(2 * np.pi * (target_rpm / 60.0) * t)
        combined = cardiac + resp

        res = resp_engine.estimate_rr(combined, fs=fs, sqi_score=85.0)
        assert res.is_valid
        # Error must be within 1.5 breaths/min
        assert abs(res.rr_rpm - target_rpm) <= 1.5


def test_prv_metrics_computation():
    prv_engine = PRVEngine(min_duration_s=15.0)
    fs = 30.0
    duration_s = 20.0
    t = np.linspace(0, duration_s, int(duration_s * fs), endpoint=False)

    # 72 BPM = 1.2 Hz -> ~24 peaks in 20s
    sig = np.sin(2 * np.pi * 1.2 * t)
    res = prv_engine.compute_prv(sig, fs=fs, sqi_score=90.0, hr_bpm_hint=72.0)

    assert res.is_valid
    assert res.mean_ppi_ms > 700.0 and res.mean_ppi_ms < 900.0  # Approx 833 ms
    assert res.label == "Pulse Rate Variability (PRV - Optical BVP)"
    assert res.num_peaks_detected >= 15


def test_stress_index_computation():
    stress_engine = StressEngine()

    # Case 1: High RMSSD (relaxed, high vagal tone) + Low HR -> Low stress
    res_relaxed = stress_engine.compute_stress_index(
        hr_bpm=60.0,
        rmssd_ms=65.0,
        rr_rpm=12.0,
        confidence_hr=0.9,
        confidence_prv=0.85
    )
    assert res_relaxed.is_valid
    assert res_relaxed.stress_level == "Low"

    # Case 2: Low RMSSD (high sympathetic arousal) + High HR -> High stress
    res_stressed = stress_engine.compute_stress_index(
        hr_bpm=110.0,
        rmssd_ms=12.0,
        rr_rpm=24.0,
        confidence_hr=0.9,
        confidence_prv=0.85
    )
    assert res_stressed.is_valid
    assert res_stressed.stress_level == "High"
    assert res_stressed.stress_score > res_relaxed.stress_score
