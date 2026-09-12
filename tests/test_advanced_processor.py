import numpy as np
import pytest
from core.rppg.advanced_processor import AdvancedRPPGProcessor
from core.skin.segmenter import SkinSegmenter

def test_advanced_rppg_chrom_and_pos_extraction():
    processor = AdvancedRPPGProcessor(fps=30.0)
    
    # Generate 10 seconds of synthetic pulsatile RGB trace (HR = 72 BPM => 1.2 Hz)
    t = np.linspace(0, 10, 300)
    hr_hz = 1.2
    
    # Blood Volume Pulse modulation
    pulse = 0.02 * np.sin(2 * np.pi * hr_hz * t)
    
    # R, G, B channels with green carrying strongest cardiac absorption
    r_trace = 145.0 + 5.0 * pulse + np.random.normal(0, 0.1, len(t))
    g_trace = 110.0 + 20.0 * pulse + np.random.normal(0, 0.1, len(t))
    b_trace = 95.0 + 3.0 * pulse + np.random.normal(0, 0.1, len(t))
    
    rgb_seq = np.column_stack([r_trace, g_trace, b_trace])
    
    # 1. Test CHROM extraction
    bvp_chrom = processor.process_color_time_series(rgb_seq)
    assert bvp_chrom is not None
    assert len(bvp_chrom) == len(t)
    
    # 2. Test POS extraction
    bvp_pos = processor.process_pos_time_series(rgb_seq)
    assert bvp_pos is not None
    assert len(bvp_pos) == len(t)
    
    # 3. Test Ensemble Vitals Computation
    bpm, rmssd = processor.compute_vitals(bvp_chrom)
    assert 68.0 <= bpm <= 76.0, f"Expected ~72 BPM, got {bpm}"
    assert rmssd >= 0.0

def test_semantic_skin_segmenter_with_glare_rejection():
    segmenter = SkinSegmenter(reject_specular_glare=True)
    
    # Create synthetic skin patch (BGR format: e.g. R=160, G=115, B=95)
    patch = np.zeros((50, 50, 3), dtype=np.uint8)
    patch[:, :] = [95, 115, 160]
    
    # Inject a specular glare spot in the middle (R=255, G=255, B=255)
    patch[20:30, 20:30] = [255, 255, 255]
    
    mask, pct = segmenter.segment(patch)
    assert mask.shape == (50, 50)
    assert pct > 70.0
    
    # Verify the specular glare region was rejected (set to 0)
    assert np.all(mask[22:28, 22:28] == 0)
