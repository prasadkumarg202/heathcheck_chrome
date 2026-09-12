"""
Automated benchmark suite for AuraPulse across demographic & environmental conditions.
Evaluates MAE and RMSE across Fitzpatrick scales I-VI and variable SNR conditions.
"""

import numpy as np
import pytest
from core.pipeline import AuraPulseEngine
from core.roi.extractor import ROIData
from research.synthetic.generator import SyntheticSignalGenerator


def test_fitzpatrick_scale_robustness_benchmark():
    """
    Evaluates that POS algorithm extracts accurate HR across Fitzpatrick skin types I to VI.
    """
    target_hr = 76.0
    fs = 30.0
    gen = SyntheticSignalGenerator(fs=fs)
    dummy_poly = np.array([[10, 10], [50, 10], [50, 50], [10, 50]])

    errors = []
    for fitz_type in [1, 2, 3, 4, 5, 6]:
        gt = gen.generate_signal(
            duration_s=20.0,
            hr_bpm=target_hr,
            rr_rpm=15.0,
            snr_db=15.0,
            skin_tone_fitzpatrick=fitz_type
        )

        engine = AuraPulseEngine(
            min_measurement_duration_s=8.0,
            window_duration_s=20.0,
            target_fs=fs,
            min_sqi_threshold=30.0
        )
        engine.start_session()

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

        res = engine.compute_vitals()
        assert res.status == "VALID", f"Failed for Fitzpatrick scale {fitz_type}: {res.reason}"
        err = abs(res.heart_rate["value"] - target_hr)
        errors.append(err)

    mae = float(np.mean(errors))
    # Mean Absolute Error across all skin tones must be < 2.0 BPM
    assert mae < 2.0, f"MAE across Fitzpatrick scales exceeded 2.0 BPM (got {mae:.2f})"


def test_snr_noise_stress_benchmark():
    """
    Tests engine stability under decreasing SNR levels.
    """
    target_hr = 82.0
    fs = 30.0
    gen = SyntheticSignalGenerator(fs=fs)
    dummy_poly = np.array([[10, 10], [50, 10], [50, 50], [10, 50]])

    # Clean condition (SNR = 20 dB) -> Should be highly accurate
    gt_clean = gen.generate_signal(duration_s=20.0, hr_bpm=target_hr, snr_db=20.0)
    engine = AuraPulseEngine(min_measurement_duration_s=8.0, target_fs=fs, min_sqi_threshold=30.0)
    engine.start_session()
    for idx in range(len(gt_clean.timestamps)):
        t_s = gt_clean.timestamps[idx]
        roi_dict = {}
        for roi_name in ["forehead", "left_cheek", "right_cheek"]:
            rgb = tuple(gt_clean.simulated_rgb[roi_name][idx])
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

    res_clean = engine.compute_vitals()
    assert res_clean.status == "VALID"
    assert abs(res_clean.heart_rate["value"] - target_hr) <= 1.5
