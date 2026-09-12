"""
Unit tests for ArrhythmiaEngine and FitnessAnalyticsEngine.
"""

import numpy as np
import pytest
from core.vitals.arrhythmia import ArrhythmiaEngine
from core.analytics.fitness_models import FitnessAnalyticsEngine


def test_regular_sinus_rhythm():
    engine = ArrhythmiaEngine(min_intervals=10)
    np.random.seed(42)
    ibis = [850.0 + float(np.random.normal(0, 15)) for _ in range(30)]
    res = engine.evaluate_rhythm(ibis, sqi_score=85.0, hr_bpm_hint=70.0)

    assert res.is_valid is True
    assert res.is_irregular is False
    assert res.rhythm_status == "Regular Sinus Rhythm"
    assert res.cv_pct < 8.0
    assert res.sd1_ms > 0.0
    assert res.sd2_ms > 0.0
    assert res.shannon_entropy > 0.0


def test_afib_irregular_pulse_detection():
    engine = ArrhythmiaEngine(min_intervals=10)
    np.random.seed(42)
    afib_ibis = [float(np.random.uniform(500.0, 1100.0)) for _ in range(35)]
    res = engine.evaluate_rhythm(afib_ibis, sqi_score=85.0)

    assert res.is_valid is True
    assert res.is_irregular is True
    assert res.rhythm_status == "Irregular / Potential AFib"
    assert res.cv_pct >= 12.0
    assert res.sd1_sd2_ratio >= 0.80
    assert "physician" in res.recommendation.lower() or "ecg" in res.recommendation.lower()


def test_arrhythmia_insufficient_intervals():
    engine = ArrhythmiaEngine(min_intervals=10)
    res = engine.evaluate_rhythm([800.0, 810.0, 790.0], sqi_score=90.0)
    assert res.is_valid is False
    assert res.rhythm_status == "Indeterminate"
    assert res.rejection_reason == "INSUFFICIENT_BEAT_INTERVALS"


def test_non_exercise_vo2max_estimation():
    fitness_engine = FitnessAnalyticsEngine()

    # Active 28-year-old male, BMI 22.0, RMSSD 45ms, activity level 4
    res_male = fitness_engine.estimate_vo2_max(
        age=28.0,
        is_male=True,
        bmi=22.0,
        rmssd_ms=45.0,
        activity_level=4
    )
    assert res_male.is_valid is True
    assert 38.0 <= res_male.vo2_max_ml_kg_min <= 65.0
    assert res_male.fitness_tier in ["Fair", "Good", "Excellent", "Superior"]

    # Sedentary 55-year-old female, BMI 30.0, RMSSD 18ms, activity level 1
    res_female = fitness_engine.estimate_vo2_max(
        age=55.0,
        is_male=False,
        bmi=30.0,
        rmssd_ms=18.0,
        activity_level=1
    )
    assert res_female.is_valid is True
    assert 12.0 <= res_female.vo2_max_ml_kg_min <= 30.0
    assert res_female.fitness_tier in ["Poor", "Fair"]


def test_cardiorespiratory_coherence():
    fitness_engine = FitnessAnalyticsEngine()
    fs = 30.0
    duration_s = 15.0
    t = np.linspace(0, duration_s, int(duration_s * fs))

    bvp = np.sin(2 * np.pi * 1.2 * t) * (1.0 + 0.3 * np.sin(2 * np.pi * 0.25 * t))

    res = fitness_engine.compute_cardiorespiratory_coherence(
        bvp_signal=bvp,
        rr_rpm=15.0,
        fs=fs,
        sqi_score=85.0
    )
    assert res.is_valid is True
    assert res.coherence_score > 30.0
    assert res.coherence_status in ["High Coherence", "Moderate Coherence"]
