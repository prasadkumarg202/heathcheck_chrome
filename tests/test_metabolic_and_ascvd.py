import numpy as np
import pytest
from core.vitals.hemoglobin import HemoglobinEngine, HemoglobinResult
from core.analytics.metabolic_models import MetabolicRiskEngine, MetabolicRiskResult
from core.analytics.risk_models import HealthRiskAnalyticsEngine, CVDRiskResult, VascularAgeResult
from core.vitals.stress import StressEngine, StressResult
from core.pipeline import AuraPulseEngine, HealthMeasurementResult
from core.roi.extractor import ROIData


def test_hemoglobin_estimation_male_and_female():
    engine = HemoglobinEngine()
    
    # Synthetic RGB temporal signal (300 frames)
    t = np.linspace(0, 10, 300)
    dc_r, dc_g, dc_b = 140.0, 100.0, 80.0
    ac_r = 2.0 * np.sin(2 * np.pi * 1.2 * t)
    ac_g = 3.5 * np.sin(2 * np.pi * 1.2 * t)
    ac_b = 1.0 * np.sin(2 * np.pi * 1.2 * t)
    
    rgb_temporal = np.column_stack([dc_r + ac_r, dc_g + ac_g, dc_b + ac_b])

    # Male
    res_m = engine.estimate_hemoglobin(rgb_temporal, is_male=True, sqi_score=85.0)
    assert res_m.is_valid is True
    assert 13.0 <= res_m.hb_g_dl <= 18.5
    assert "Male" in res_m.reference_range
    assert res_m.confidence > 0.5

    # Female
    res_f = engine.estimate_hemoglobin(rgb_temporal, is_male=False, sqi_score=85.0)
    assert res_f.is_valid is True
    assert 11.5 <= res_f.hb_g_dl <= 17.5
    assert "Female" in res_f.reference_range
    assert res_m.hb_g_dl > res_f.hb_g_dl  # Male baseline is higher


def test_metabolic_risk_models():
    engine = MetabolicRiskEngine()

    # Healthy profile
    res_healthy = engine.estimate_metabolic_risks(
        rmssd_ms=65.0,
        lf_hf_ratio=1.1,
        systolic_bp=116.0,
        age=28.0,
        bmi=21.5,
        stiffness_index=6.8,
        is_diabetic_history=False,
        confidence_inputs=0.9
    )
    assert res_healthy.is_valid is True
    assert res_healthy.estimated_fbg_mg_dl < 100.0
    assert res_healthy.hba1c_status == "Normal"
    assert res_healthy.risk_level == "Low Risk"

    # Elevated metabolic profile
    res_elevated = engine.estimate_metabolic_risks(
        rmssd_ms=18.0,
        lf_hf_ratio=3.8,
        systolic_bp=145.0,
        age=58.0,
        bmi=32.0,
        stiffness_index=11.2,
        is_diabetic_history=True,
        confidence_inputs=0.9
    )
    assert res_elevated.is_valid is True
    assert res_elevated.estimated_fbg_mg_dl > res_healthy.estimated_fbg_mg_dl
    assert res_elevated.estimated_hba1c_pct > res_healthy.estimated_hba1c_pct
    assert res_elevated.risk_level in ["Moderate Risk", "High Risk"]


def test_framingham_ascvd_cox_model():
    engine = HealthRiskAnalyticsEngine()

    # Young healthy male
    res_young = engine.estimate_10yr_cvd_risk(
        age=30.0,
        is_male=True,
        systolic_bp=118.0,
        is_smoker=False,
        is_diabetic=False,
        bmi=22.5,
        hr_bpm=68.0,
        rmssd_ms=55.0
    )
    assert res_young.ten_year_risk_pct < 5.0
    assert res_young.risk_category == "Low (<10%)"

    # Older high-risk smoker male
    res_risk = engine.estimate_10yr_cvd_risk(
        age=62.0,
        is_male=True,
        systolic_bp=152.0,
        is_smoker=True,
        is_diabetic=True,
        bmi=29.5,
        hr_bpm=88.0,
        rmssd_ms=16.0
    )
    assert res_risk.ten_year_risk_pct > 15.0
    assert res_risk.risk_category in ["Moderate (10-20%)", "High (20-30%)", "Very High (>30%)"]


def test_stress_engine_with_ppis():
    engine = StressEngine()
    
    # Synthetic clean PPI sequence around 800ms (75 bpm) with healthy respiratory sinus arrhythmia (range ~160ms)
    ppis = [720.0, 760.0, 810.0, 860.0, 880.0, 840.0, 790.0, 740.0, 730.0, 770.0, 820.0, 870.0, 850.0, 800.0, 750.0]
    res = engine.compute_stress_index(
        hr_bpm=75.0,
        rmssd_ms=48.0,
        rr_rpm=15.0,
        ppi_intervals_ms=ppis
    )
    assert res.is_valid is True
    assert 30.0 <= res.baevsky_stress_index <= 250.0
    assert 3.5 <= res.pulse_respiration_quotient <= 5.5
    assert res.parasympathetic_score > 0.0
    assert res.sympathetic_score > 0.0


def test_master_pipeline_full_integration():
    engine = AuraPulseEngine(min_measurement_duration_s=1.0, window_duration_s=10.0, target_fs=30.0)
    engine.start_session()

    dummy_poly = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    t_vals = np.linspace(0, 6, 180)

    # Ingest 6 seconds of multi-spectral signal
    for i, t in enumerate(t_vals):
        pulsatile = np.sin(2 * np.pi * 1.2 * t)
        rois = {
            "forehead": ROIData(
                name="forehead",
                polygon=dummy_poly,
                mean_rgb=(130.0 + 2.0 * pulsatile, 100.0 + 3.0 * pulsatile, 85.0 + 1.0 * pulsatile),
                median_rgb=(130.0, 100.0, 85.0),
                std_rgb=(1.0, 1.0, 1.0),
                skin_pixel_pct=95.0,
                num_valid_pixels=200,
                is_valid=True
            ),
            "left_cheek": ROIData(
                name="left_cheek",
                polygon=dummy_poly,
                mean_rgb=(128.0 + 2.0 * pulsatile, 98.0 + 3.0 * pulsatile, 84.0 + 1.0 * pulsatile),
                median_rgb=(128.0, 98.0, 84.0),
                std_rgb=(1.0, 1.0, 1.0),
                skin_pixel_pct=95.0,
                num_valid_pixels=200,
                is_valid=True
            )
        }
        engine.signal_buffer.append(t, rois)

    vitals = engine.compute_vitals({
        "age": 35,
        "is_male": True,
        "height_cm": 178,
        "weight_kg": 75,
        "waist_cm": 82,
        "is_smoker": False,
        "is_diabetic": False
    })

    assert vitals.status == "VALID"
    assert vitals.heart_rate is not None
    assert 65.0 <= vitals.heart_rate["value"] <= 78.0
    assert vitals.respiration_rate is not None
    assert vitals.pulse_rate_variability is not None
    assert vitals.blood_pressure is not None
    assert vitals.spo2 is not None
    assert vitals.hemoglobin is not None
    assert vitals.hemoglobin["value"] is not None
    assert vitals.metabolic_risks is not None
    assert vitals.metabolic_risks["fbgMgDl"] is not None
    assert vitals.metabolic_risks["hba1cPct"] is not None
    assert vitals.vascular_age is not None
    assert vitals.cvd_risk is not None
    assert vitals.body_composition is not None
    assert vitals.wellness_indices is not None
    assert vitals.algorithm_version == "0.3.0"
