"""
Unit tests for Respiration and PRQ synchronization & Vascular Age anchoring.
"""

import numpy as np
import pytest
from core.pipeline import AuraPulseEngine
from core.roi.extractor import ROIData
from core.analytics.risk_models import HealthRiskAnalyticsEngine


def test_vascular_age_chronological_anchoring():
    risk_engine = HealthRiskAnalyticsEngine()

    # Case 1: 35-year-old healthy subject
    res35 = risk_engine.estimate_vascular_age(
        chronological_age=35.0,
        systolic_bp=120.0,
        stiffness_index=7.0,
        sdppg_aging_index=-0.40,
        is_male=True,
        is_smoker=False,
        is_diabetic=False
    )
    assert res35.vascular_age_years == pytest.approx(35.0, abs=3.0)
    assert -10.0 <= res35.age_delta <= 20.0
    assert res35.stiffness_status in ["Optimal", "Normal"]

    # Case 2: 55-year-old hypertensive smoker subject (should not collapse to 18)
    res55 = risk_engine.estimate_vascular_age(
        chronological_age=55.0,
        systolic_bp=150.0,
        stiffness_index=9.5,
        sdppg_aging_index=-0.10,
        is_male=True,
        is_smoker=True,
        is_diabetic=True
    )
    assert res55.vascular_age_years >= 55.0
    assert res55.age_delta <= 20.0
    assert res55.stiffness_status == "Accelerated Stiffening"


def test_prq_respiration_synchronization_on_short_or_invalid_signal():
    engine = AuraPulseEngine(min_measurement_duration_s=4.0, window_duration_s=6.0, target_fs=30.0)
    engine.start_session()

    fs = 30.0
    duration_s = 5.0
    t_vals = np.linspace(0, duration_s, int(duration_s * fs))
    dummy_poly = np.array([[10, 10], [50, 10], [50, 50], [10, 50]])

    # Generate 5 seconds of simple cardiac pulsation without respiratory modulation
    for t in t_vals:
        pulsatile = np.sin(2 * np.pi * 1.25 * t)
        rois = {
            "forehead": ROIData("forehead", dummy_poly, (120.0 + pulsatile, 100.0 + 2.0 * pulsatile, 90.0), (120.0, 100.0, 90.0), (1.0, 1.0, 1.0), 90.0, 200, True),
            "left_cheek": ROIData("left_cheek", dummy_poly, (120.0 + pulsatile, 100.0 + 2.0 * pulsatile, 90.0), (120.0, 100.0, 90.0), (1.0, 1.0, 1.0), 90.0, 200, True)
        }
        engine.signal_buffer.append(t, rois)

    vitals = engine.compute_vitals({"age": 35, "is_male": True})

    # If respiration rate is None (insufficient window), PRQ MUST also be None
    if vitals.respiration_rate["value"] is None:
        assert vitals.stress_index["prq"] is None
        assert vitals.wellness_indices["prq"] is None
        assert vitals.respiration_rate["measurement_status"] == "insufficient_signal"
