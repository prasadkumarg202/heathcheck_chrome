"""
Unit tests for Signal Quality Index (SQI) and Fail-Safe Rejection Engine.
"""

import numpy as np
import pytest
from core.quality.sqi import SignalQualityEngine


def test_sqi_on_clean_signal():
    sqi_engine = SignalQualityEngine(min_acceptable_sqi=45.0)
    fs = 30.0
    t = np.linspace(0, 15, int(15 * fs), endpoint=False)
    # Clean sine wave at 1.2 Hz (72 BPM)
    sig = np.sin(2 * np.pi * 1.2 * t)

    res = sqi_engine.compute_sqi(sig, fs=fs)
    assert res.is_acceptable
    assert res.sqi_score >= 70.0
    assert res.category in ["Good", "Excellent"]
    assert res.rejection_reason is None


def test_sqi_rejection_on_pure_noise():
    sqi_engine = SignalQualityEngine(min_acceptable_sqi=45.0)
    fs = 30.0
    np.random.seed(42)
    # High-entropy white noise without periodic cardiac structure
    noise = np.random.normal(0, 1.0, int(15 * fs))

    res = sqi_engine.compute_sqi(noise, fs=fs)
    # White noise should be rejected or have low SQI
    assert res.sqi_score < 55.0
    assert res.category in ["Invalid", "Poor", "Fair"]


def test_sqi_on_short_window():
    sqi_engine = SignalQualityEngine(min_acceptable_sqi=45.0)
    short_sig = np.sin(np.linspace(0, 3, 30))  # Only 1 second @ 30 FPS

    res = sqi_engine.compute_sqi(short_sig, fs=30.0)
    assert not res.is_acceptable
    assert res.rejection_reason == "WINDOW_TOO_SHORT"
