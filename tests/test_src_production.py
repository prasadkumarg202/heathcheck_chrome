import pytest
import numpy as np
from src.signal.pos import extract_bvp_pos
from src.signal.chrom import extract_bvp_chrom
from src.signal.green import extract_bvp_green
from src.signal.pbv import extract_bvp_pbv
from src.signal.lgi import extract_bvp_lgi
from src.signal.filters import butter_bandpass_filter, detrend_signal
from src.cv.face import FaceDetector
from src.cv.roi import ROIExtractor
from src.cv.skin import SkinSegmenter
from src.sqi.quality import SignalQualityEngine
from src.metrics.hr import estimate_heart_rate
from src.metrics.respiration import estimate_respiration_rate
from src.metrics.prv import compute_prv_metrics
from src.metrics.stress import compute_stress_index
from src.metrics.blood_pressure import estimate_blood_pressure
from src.metrics.spo2 import estimate_spo2
from src.metrics.cardiac_workload import compute_cardiac_workload
from src.models.risk_models import compute_body_composition, estimate_vascular_age, estimate_10yr_cvd_risk
from src.reporting.report import generate_health_report

def test_src_production_signal_extraction():
    fps = 30.0
    duration = 10.0
    t = np.linspace(0, duration, int(fps * duration))
    
    # Synthetic RGB pulse signals
    r = 150.0 + 2.0 * np.sin(2 * np.pi * 1.2 * t)
    g = 120.0 + 4.0 * np.sin(2 * np.pi * 1.2 * t)
    b = 100.0 + 1.0 * np.sin(2 * np.pi * 1.2 * t)
    rgb_means = np.column_stack([r, g, b])
    
    bvp_pos = extract_bvp_pos(rgb_means, fps)
    assert len(bvp_pos) == len(t)
    
    bvp_chrom = extract_bvp_chrom(rgb_means, fps)
    assert len(bvp_chrom) == len(t)
    
    bvp_green = extract_bvp_green(rgb_means)
    assert len(bvp_green) == len(t)
    
    bvp_pbv = extract_bvp_pbv(rgb_means)
    assert len(bvp_pbv) == len(t)
    
    bvp_lgi = extract_bvp_lgi(rgb_means)
    assert len(bvp_lgi) == len(t)

def test_src_production_metrics_and_reporting():
    fps = 30.0
    t = np.linspace(0, 10.0, 300)
    bvp = np.sin(2 * np.pi * 1.2 * t) # 72 BPM
    
    hr_res = estimate_heart_rate(bvp, fps)
    assert hr_res.hr_bpm is not None
    assert 68.0 <= hr_res.hr_bpm <= 76.0
    
    resp_res = estimate_respiration_rate(bvp, fps)
    assert resp_res is not None
    
    prv_res = compute_prv_metrics(bvp, fps)
    assert prv_res is not None
    
    stress_res = compute_stress_index(bvp, fps)
    assert stress_res is not None
    
    report = generate_health_report(
        patient_id="TEST-001",
        session_id="SESS-001",
        duration_s=60.0,
        vitals_data={"heart_rate": hr_res.hr_bpm, "respiration_rate": resp_res.rr_rpm},
        sqi_score=92.0
    )
    assert report["patient_id"] == "TEST-001"
    assert report["signal_quality"]["overall_sqi"] == 92.0
    assert "POS" in report["engine_provenance"]["algorithms"]
