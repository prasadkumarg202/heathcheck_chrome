"""
Master AuraPulse Engine Pipeline.
Coordinates video frame ingestion, face/ROI tracking, multi-algorithm rPPG extraction,
signal quality evaluation, vital signs computation, and standardized API result dispatch.
"""

from __future__ import annotations
import time
from dataclasses import dataclass, asdict
from typing import Dict, Optional, Tuple, Any
import numpy as np

from core.face.detector import FaceDetector, FaceLandmarks, FaceQualityScore
from core.roi.extractor import ROIExtractor, ROIData
from core.signal.extractor import TemporalSignalBuffer
from core.filters.signal_filters import SignalPreprocessor
from core.rppg.pos import POSAlgorithm
from core.rppg.chrom import CHROMAlgorithm
from core.rppg.green import GreenAlgorithm
from core.rppg.ica import ICAAlgorithm
from core.quality.sqi import SignalQualityEngine, SQIResult
from core.fusion.fusion_engine import SignalFusionEngine
from core.vitals.hr import HeartRateEngine, HRResult
from core.vitals.respiration import RespirationEngine, RespirationResult
from core.vitals.prv import PRVEngine, PRVResult
from core.vitals.stress import StressEngine, StressResult


@dataclass
class HealthMeasurementResult:
    status: str                         # "VALID" or "UNAVAILABLE"
    heart_rate: Optional[Dict[str, Any]] = None
    respiration_rate: Optional[Dict[str, Any]] = None
    pulse_rate_variability: Optional[Dict[str, Any]] = None
    stress_index: Optional[Dict[str, Any]] = None
    pulse_waveform: Optional[list] = None
    signal_quality: float = 0.0
    quality_category: str = "Invalid"
    face_quality: float = 0.0
    measurement_duration_s: float = 0.0
    algorithm_version: str = "0.1.0"
    engine: str = "AuraPulse-Core"
    validation_status: str = "Research/Validated - On-Device Core"
    reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AuraPulseEngine:
    """
    Main on-device contactless physiological measurement engine.
    """

    def __init__(
        self,
        min_measurement_duration_s: float = 6.0,
        window_duration_s: float = 25.0,
        target_fs: float = 30.0,
        min_sqi_threshold: float = 35.0,
        primary_algorithm: str = "POS",
    ):
        self.min_measurement_duration_s = min_measurement_duration_s
        self.window_duration_s = window_duration_s
        self.target_fs = target_fs
        self.min_sqi_threshold = min_sqi_threshold
        self.primary_algorithm = primary_algorithm

        # Subsystems
        self.face_detector = FaceDetector()
        self.roi_extractor = ROIExtractor()
        self.signal_buffer = TemporalSignalBuffer(capacity=int(window_duration_s * target_fs * 2))
        self.preprocessor = SignalPreprocessor()

        self.pos_algo = POSAlgorithm()
        self.chrom_algo = CHROMAlgorithm()
        self.green_algo = GreenAlgorithm()
        self.ica_algo = ICAAlgorithm()

        self.sqi_engine = SignalQualityEngine(min_acceptable_sqi=min_sqi_threshold)
        self.fusion_engine = SignalFusionEngine()

        self.hr_engine = HeartRateEngine()
        self.respiration_engine = RespirationEngine()
        self.prv_engine = PRVEngine(min_duration_s=min_measurement_duration_s)
        self.stress_engine = StressEngine()

        self.session_start_time: Optional[float] = None
        self.frame_count = 0

    def start_session(self) -> None:
        """Starts a new measurement session and clears history."""
        self.signal_buffer = TemporalSignalBuffer(capacity=int(self.window_duration_s * self.target_fs * 2))
        self.hr_engine.reset_tracking()
        self.session_start_time = time.perf_counter()
        self.frame_count = 0

    def process_frame(
        self,
        frame: np.ndarray,
        timestamp_s: Optional[float] = None
    ) -> Tuple[bool, Optional[FaceLandmarks], Optional[Dict[str, ROIData]], FaceQualityScore]:
        """
        Processes a single video frame: face detection, dynamic ROI extraction, and buffer update.
        """
        if timestamp_s is None:
            if self.session_start_time is None:
                self.start_session()
            timestamp_s = time.perf_counter() - self.session_start_time

        self.frame_count += 1

        # 1. Face Detection & Quality Assessment
        has_face, landmarks, face_quality = self.face_detector.detect(frame)

        if not has_face or landmarks is None:
            return False, None, None, face_quality

        # 2. Dynamic Polygon ROI Extraction & Skin Masking
        rois = self.roi_extractor.extract_rois(frame, landmarks)

        # 3. Append to temporal buffer
        self.signal_buffer.append(timestamp_s, rois)

        return True, landmarks, rois, face_quality

    def compute_vitals(self) -> HealthMeasurementResult:
        """
        Executes signal processing, rPPG extraction, SQI, and vitals estimation
        over the current sliding buffer window.
        """
        t_uniform, roi_rgbs, duration = self.signal_buffer.get_window(
            duration_s=self.window_duration_s,
            target_fs=self.target_fs
        )

        if duration < self.min_measurement_duration_s or not roi_rgbs:
            return HealthMeasurementResult(
                status="UNAVAILABLE",
                measurement_duration_s=round(duration, 1),
                signal_quality=0.0,
                quality_category="Invalid",
                reason=f"MEASUREMENT_IN_PROGRESS_NEED_{int(self.min_measurement_duration_s)}S",
            )

        # 1. Extract raw rPPG per ROI using POS, CHROM, or Green on raw DC+AC signals
        roi_signals: Dict[str, np.ndarray] = {}
        roi_sqis: Dict[str, float] = {}

        for name, rgb in roi_rgbs.items():
            if rgb.shape[0] < 10 or np.all(rgb == 0):
                continue

            if self.primary_algorithm == "CHROM":
                bvp_raw = self.chrom_algo.extract(rgb, fs=self.target_fs)
            elif self.primary_algorithm == "GREEN":
                bvp_raw = self.green_algo.extract_green(rgb)
            else:  # Default POS
                bvp_raw = self.pos_algo.extract(rgb, fs=self.target_fs)

            # Bandpass Filter [0.7, 3.5] Hz (42 to 210 BPM)
            bvp_filtered = self.preprocessor.butterworth_bandpass(
                bvp_raw,
                lowcut=0.7,
                highcut=3.5,
                fs=self.target_fs,
                order=3
            )
            # Remove motion spikes
            bvp_clean = self.preprocessor.remove_motion_spikes(bvp_filtered)

            roi_signals[name] = bvp_clean

            # Compute individual ROI SQI
            sqi_item = self.sqi_engine.compute_sqi(bvp_clean, fs=self.target_fs)
            roi_sqis[name] = sqi_item.sqi_score

        if not roi_signals:
            return HealthMeasurementResult(
                status="UNAVAILABLE",
                measurement_duration_s=round(duration, 1),
                signal_quality=0.0,
                quality_category="Invalid",
                reason="ALL_ROIS_INVALID",
            )

        # 2. Multi-ROI Weighted Fusion
        fused_bvp, roi_weights = self.fusion_engine.fuse_roi_signals(roi_signals, roi_sqis)

        # 3. Composite Signal Quality Assessment
        overall_sqi = self.sqi_engine.compute_sqi(
            fused_bvp,
            fs=self.target_fs,
            other_roi_signals=roi_signals
        )

        # Fail-Safe Gate: If SQI is too low, reject measurement
        if not overall_sqi.is_acceptable:
            return HealthMeasurementResult(
                status="UNAVAILABLE",
                measurement_duration_s=round(duration, 1),
                signal_quality=overall_sqi.sqi_score,
                quality_category=overall_sqi.category,
                reason=f"SIGNAL_QUALITY_TOO_LOW ({overall_sqi.category})",
            )

        # 4. Heart Rate Computation
        hr_res = self.hr_engine.estimate_hr(
            fused_bvp,
            fs=self.target_fs,
            sqi_score=overall_sqi.sqi_score
        )

        if not hr_res.is_valid:
            return HealthMeasurementResult(
                status="UNAVAILABLE",
                measurement_duration_s=round(duration, 1),
                signal_quality=overall_sqi.sqi_score,
                quality_category=overall_sqi.category,
                reason=hr_res.rejection_reason or "HEART_RATE_INDETERMINATE",
            )

        # 5. Respiration Rate Computation
        fh_rgb = roi_rgbs.get("forehead", next(iter(roi_rgbs.values())))
        raw_green = fh_rgb[:, 1] if fh_rgb.ndim == 2 else fused_bvp
        rr_res = self.respiration_engine.estimate_rr(
            raw_green,
            fs=self.target_fs,
            sqi_score=overall_sqi.sqi_score
        )

        # 6. Pulse Rate Variability (PRV) Computation
        prv_res = self.prv_engine.compute_prv(
            fused_bvp,
            fs=self.target_fs,
            sqi_score=overall_sqi.sqi_score,
            hr_bpm_hint=hr_res.hr_bpm
        )

        # 7. Physiological Stress Index
        stress_res = self.stress_engine.compute_stress_index(
            hr_bpm=hr_res.hr_bpm,
            rmssd_ms=prv_res.rmssd_ms if prv_res.is_valid else 35.0,
            rr_rpm=rr_res.rr_rpm if rr_res.is_valid else 0.0,
            confidence_hr=hr_res.confidence,
            confidence_prv=prv_res.confidence if prv_res.is_valid else 0.5,
        )

        # Downsample waveform for API transmission (last 120 samples)
        waveform_sample = [round(float(v), 4) for v in fused_bvp[-120:]]

        return HealthMeasurementResult(
            status="VALID",
            heart_rate={
                "value": hr_res.hr_bpm,
                "unit": "bpm",
                "confidence": hr_res.confidence,
                "signalQuality": overall_sqi.sqi_score,
                "methods": hr_res.method_estimates,
            },
            respiration_rate={
                "value": rr_res.rr_rpm if rr_res.is_valid else None,
                "unit": "breaths/min",
                "confidence": rr_res.confidence if rr_res.is_valid else 0.0,
                "status": "VALID" if rr_res.is_valid else "INSUFFICIENT_DATA",
            },
            pulse_rate_variability={
                "rmssd": prv_res.rmssd_ms if prv_res.is_valid else None,
                "sdnn": prv_res.sdnn_ms if prv_res.is_valid else None,
                "meanPpi": prv_res.mean_ppi_ms if prv_res.is_valid else None,
                "pnn50": prv_res.pnn50_pct if prv_res.is_valid else None,
                "sd1": prv_res.sd1_ms if prv_res.is_valid else None,
                "sd2": prv_res.sd2_ms if prv_res.is_valid else None,
                "unit": "ms",
                "confidence": prv_res.confidence if prv_res.is_valid else 0.0,
                "label": prv_res.label,
                "status": "VALID" if prv_res.is_valid else "INSUFFICIENT_DATA",
            },
            stress_index={
                "score": stress_res.stress_score if stress_res.is_valid else None,
                "level": stress_res.stress_level if stress_res.is_valid else "Unknown",
                "confidence": stress_res.confidence if stress_res.is_valid else 0.0,
                "disclaimer": stress_res.disclaimer,
            },
            pulse_waveform=waveform_sample,
            signal_quality=overall_sqi.sqi_score,
            quality_category=overall_sqi.category,
            face_quality=100.0,
            measurement_duration_s=round(duration, 1),
            algorithm_version="0.1.0",
            engine="AuraPulse-Core",
            validation_status="Research/Validated - On-Device Core",
            reason=None,
        )
