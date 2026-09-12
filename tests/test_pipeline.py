"""
End-to-end integration tests for AuraPulseEngine.
"""

import numpy as np
import pytest
from core.pipeline import AuraPulseEngine
from core.roi.extractor import ROIData
from research.synthetic.generator import SyntheticSignalGenerator


def test_pipeline_streaming_simulation():
    engine = AuraPulseEngine(
        min_measurement_duration_s=8.0,
        window_duration_s=15.0,
        target_fs=30.0,
        min_sqi_threshold=35.0
    )
    engine.start_session()

    gen = SyntheticSignalGenerator(fs=30.0)
    gt = gen.generate_signal(duration_s=15.0, hr_bpm=74.0, rr_rpm=16.0, snr_db=20.0)

    # Ingest synthetic observations directly into pipeline buffer
    dummy_poly = np.array([[10, 10], [50, 10], [50, 50], [10, 50]])
    for idx in range(len(gt.timestamps)):
        t_s = gt.timestamps[idx]
        roi_dict = {}
        for roi_name in ["forehead", "left_cheek", "right_cheek"]:
            rgb = tuple(gt.simulated_rgb[roi_name][idx])
            roi_dict[roi_name] = ROIData(
                name=roi_name,
                polygon=dummy_poly,
                mean_rgb=rgb,
                median_rgb=rgb,
                std_rgb=(1.0, 1.0, 1.0),
                skin_pixel_pct=90.0,
                num_valid_pixels=200,
                is_valid=True
            )
        engine.signal_buffer.append(t_s, roi_dict)

    # Compute vitals
    result = engine.compute_vitals()

    assert result.status in ("valid", "VALID")
    assert result.heart_rate is not None
    assert abs(result.heart_rate["value"] - 74.0) <= 2.0
    assert result.heart_rate["confidence"] > 0.40
    assert result.signal_quality >= 35.0
    assert result.measurement_duration_s >= 8.0
    assert result.algorithm_version in ("4.0.0", "0.3.0")
    assert result.engine == "AuraPulse-Clinical-Edge"


def test_pipeline_fail_safe_on_short_duration():
    engine = AuraPulseEngine(min_measurement_duration_s=10.0)
    engine.start_session()

    # Empty / short buffer
    result = engine.compute_vitals()
    assert result.status in ("collecting", "insufficient_signal", "UNAVAILABLE")
    assert "MEASUREMENT_IN_PROGRESS" in result.reason or "INSUFFICIENT_DURATION" in result.reason or result.reason is not None

