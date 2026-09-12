"""
Comprehensive automated tests for AuraPulse Health & Risk Analytics Platform.
Validates Blood Pressure, SpO2, Cardiac Workload, Baevsky Stress Index,
Vascular Age, 10-Year CVD Risk, and Anthropometrics.
"""

import math
import numpy as np
import pytest

from core.vitals.blood_pressure import BloodPressureEngine, BloodPressureResult
from core.vitals.spo2 import SpO2Engine, SpO2Result
from core.vitals.cardiac_workload import CardiacWorkloadEngine, CardiacWorkloadResult
from core.vitals.stress import StressEngine, StressResult
from core.analytics.risk_models import HealthRiskAnalyticsEngine, VascularAgeResult, CVDRiskResult, BodyCompositionResult
from core.pipeline import AuraPulseEngine


def generate_synthetic_bvp(duration_s=10.0, fs=30.0, hr_bpm=72.0):
    t = np.linspace(0, duration_s, int(duration_s * fs))
    f_cardiac = hr_bpm / 60.0
    # Primary systolic wave + secondary dicrotic reflection wave
    bvp = np.sin(2 * np.pi * f_cardiac * t) + 0.35 * np.sin(4 * np.pi * f_cardiac * t + np.pi / 4)
    return bvp


def test_blood_pressure_estimation():
    bp_engine = BloodPressureEngine()
    fs = 30.0
    bvp = generate_synthetic_bvp(duration_s=12.0, fs=fs, hr_bpm=72.0)

    res = bp_engine.estimate_blood_pressure(
        bvp_signal=bvp,
        fs=fs,
        hr_bpm=72.0,
        rmssd_ms=38.0,
        age=32.0,
        is_male=True,
        bmi=23.0,
        sqi_score=85.0
    )

    assert res.is_valid is True
    assert 90.0 <= res.systolic_bp <= 140.0
    assert 60.0 <= res.diastolic_bp <= 90.0
    assert res.pulse_pressure == pytest.approx(res.systolic_bp - res.diastolic_bp, 0.1)
    assert res.mean_arterial_pressure > res.diastolic_bp
    assert res.aha_category in ["Normal", "Elevated", "Stage 1 HTN", "Stage 2 HTN"]
    assert res.stiffness_index_ms > 0.0


def test_blood_pressure_insufficient_signal():
    bp_engine = BloodPressureEngine()
    res = bp_engine.estimate_blood_pressure(
        bvp_signal=np.array([0.1, 0.2]),
        fs=30.0,
        hr_bpm=70.0,
        sqi_score=80.0
    )
    assert res.is_valid is False
    assert res.rejection_reason == "INSUFFICIENT_SIGNAL_FOR_BP"


def test_spo2_estimation():
    spo2_engine = SpO2Engine()
    fs = 30.0
    t = np.linspace(0, 10.0, int(10.0 * fs))

    # Construct realistic RGB temporal signals: DC baseline ~120, AC modulation ~1.5
    r = 130.0 + 1.2 * np.sin(2 * np.pi * 1.2 * t)
    g = 115.0 + 2.0 * np.sin(2 * np.pi * 1.2 * t)
    b = 100.0 + 1.4 * np.sin(2 * np.pi * 1.2 * t)
    rgb = np.column_stack([r, g, b])

    res = spo2_engine.estimate_spo2(rgb, fs=fs, sqi_score=85.0)

    assert res.is_valid is True
    assert 90.0 <= res.spo2_pct <= 100.0
    assert res.category in ["Normal", "Mild Hypoxia", "Hypoxemia"]
    assert res.ratio_of_ratios > 0.0


def test_cardiac_workload_computation():
    cw_engine = CardiacWorkloadEngine()

    res = cw_engine.compute_workload(hr_bpm=75.0, systolic_bp=120.0)
    assert res.is_valid is True
    # RPP = 75 * 120 / 100 = 90.0
    assert res.rpp_score == pytest.approx(90.0, 0.1)
    assert res.workload_category == "Normal"

    res_high = cw_engine.compute_workload(hr_bpm=110.0, systolic_bp=150.0)
    # RPP = 110 * 150 / 100 = 165.0
    assert res_high.rpp_score == pytest.approx(165.0, 0.1)
    assert res_high.workload_category == "Very High"


def test_baevsky_stress_and_wellness():
    stress_engine = StressEngine()

    res = stress_engine.compute_stress_index(
        hr_bpm=68.0,
        rmssd_ms=45.0,
        rr_rpm=16.0,
        confidence_hr=0.9,
        confidence_prv=0.9,
    )

    assert res.is_valid is True
    assert 0.0 <= res.stress_score <= 100.0
    assert res.baevsky_stress_index > 0.0
    assert 0.0 <= res.parasympathetic_score <= 100.0
    # PRQ = 68 / 16 = 4.25
    assert res.pulse_respiration_quotient == pytest.approx(4.25, 0.1)


def test_body_composition_and_anthropometrics():
    risk_engine = HealthRiskAnalyticsEngine()

    res = risk_engine.compute_body_composition(
        height_cm=180.0,
        weight_kg=75.0,
        waist_cm=82.0
    )

    # BMI = 75 / (1.8^2) = 23.15
    assert res.bmi == pytest.approx(23.1, 0.2)
    assert res.bmi_category == "Normal"
    # WHtR = 82 / 180 = 0.455
    assert res.whtr == pytest.approx(0.46, 0.05)
    assert res.whtr_category == "Optimal (<0.5)"
    assert res.body_roundness_index > 0.0


def test_vascular_age_estimation():
    risk_engine = HealthRiskAnalyticsEngine()

    res = risk_engine.estimate_vascular_age(
        chronological_age=40.0,
        systolic_bp=122.0,
        stiffness_index=7.2,
        sdppg_aging_index=-0.38
    )

    assert 20.0 <= res.vascular_age_years <= 80.0
    assert isinstance(res.stiffness_status, str)


def test_10yr_cvd_risk_model():
    risk_engine = HealthRiskAnalyticsEngine()

    res = risk_engine.estimate_10yr_cvd_risk(
        age=45.0,
        is_male=True,
        systolic_bp=135.0,
        is_smoker=True,
        is_diabetic=False,
        bmi=27.0,
        hr_bpm=78.0,
        rmssd_ms=28.0
    )

    assert res.ten_year_risk_pct > 5.0
    assert res.stroke_risk_pct > 0.0
    assert "Tobacco Use" in res.key_drivers or "Elevated Blood Pressure" in res.key_drivers


def test_master_pipeline_full_synthesis():
    engine = AuraPulseEngine(min_measurement_duration_s=4.0, window_duration_s=10.0, target_fs=30.0)
    engine.start_session()

    fs = 30.0
    duration_s = 6.0
    num_frames = int(duration_s * fs)

    # Ingest synthetic spatial signals
    for i in range(num_frames):
        t = i / fs
        # Synthetic high perfusion malar & forehead signals
        pulse_g = 120.0 + 3.0 * np.sin(2 * np.pi * 1.25 * t)
        pulse_r = 135.0 + 2.0 * np.sin(2 * np.pi * 1.25 * t)
        pulse_b = 105.0 + 1.5 * np.sin(2 * np.pi * 1.25 * t)

        dummy_poly = np.array([[10, 10], [50, 10], [50, 50], [10, 50]])
        from core.roi.extractor import ROIData
        rois = {
            "forehead": ROIData("forehead", dummy_poly, (pulse_r, pulse_g, pulse_b), (pulse_r, pulse_g, pulse_b), (1.0, 1.0, 1.0), 95.0, 200, True),
            "left_cheek": ROIData("left_cheek", dummy_poly, (pulse_r, pulse_g, pulse_b), (pulse_r, pulse_g, pulse_b), (1.0, 1.0, 1.0), 95.0, 200, True),
            "right_cheek": ROIData("right_cheek", dummy_poly, (pulse_r, pulse_g, pulse_b), (pulse_r, pulse_g, pulse_b), (1.0, 1.0, 1.0), 95.0, 200, True),
        }
        engine.signal_buffer.append(t, rois)

    user_meta = {
        "age": 38.0,
        "is_male": True,
        "height_cm": 178.0,
        "weight_kg": 74.0,
        "waist_cm": 83.0,
        "is_smoker": False,
        "is_diabetic": False
    }

    result = engine.compute_vitals(user_metadata=user_meta)

    assert result.status == "VALID"
    assert result.heart_rate is not None
    assert result.heart_rate["value"] is not None
    assert result.blood_pressure is not None
    assert result.blood_pressure["systolic"] is not None
    assert result.spo2 is not None
    assert result.spo2["value"] is not None
    assert result.cardiac_workload is not None
    assert result.vascular_age is not None
    assert result.cvd_risk is not None
    assert result.body_composition is not None
    assert result.wellness_indices is not None
