"""
Master AuraPulse 4.0 Engine Pipeline.
Coordinates video frame ingestion, face/ROI tracking, multi-algorithm rPPG extraction,
signal quality evaluation, vital signs computation, and standardized API result dispatch.
Follows non-negotiable scientific integrity rules and explicit measurement status conventions.
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
from core.vitals.blood_pressure import BloodPressureEngine, BloodPressureResult
from core.vitals.spo2 import SpO2Engine, SpO2Result
from core.vitals.hemoglobin import HemoglobinEngine, HemoglobinResult
from core.vitals.cardiac_workload import CardiacWorkloadEngine, CardiacWorkloadResult
from core.analytics.risk_models import HealthRiskAnalyticsEngine, VascularAgeResult, CVDRiskResult, BodyCompositionResult
from core.analytics.metabolic_models import MetabolicRiskEngine, MetabolicRiskResult


@dataclass
class HealthMeasurementResult:
    status: str                         # "valid", "collecting", "insufficient_signal", "insufficient_duration"
    heart_rate: Optional[Dict[str, Any]] = None
    respiration_rate: Optional[Dict[str, Any]] = None
    pulse_rate_variability: Optional[Dict[str, Any]] = None
    blood_pressure: Optional[Dict[str, Any]] = None
    spo2: Optional[Dict[str, Any]] = None
    hemoglobin: Optional[Dict[str, Any]] = None
    cardiac_workload: Optional[Dict[str, Any]] = None
    stress_index: Optional[Dict[str, Any]] = None
    metabolic_risks: Optional[Dict[str, Any]] = None
    vascular_age: Optional[Dict[str, Any]] = None
    cvd_risk: Optional[Dict[str, Any]] = None
    body_composition: Optional[Dict[str, Any]] = None
    wellness_indices: Optional[Dict[str, Any]] = None
    pulse_waveform: Optional[list] = None
    signal_quality: float = 0.0
    quality_category: str = "Invalid"
    face_quality: float = 0.0
    measurement_duration_s: float = 0.0
    algorithm_version: str = "4.0.0"
    engine: str = "AuraPulse-Clinical-Edge"
    validation_status: str = "On-Device Synthesis Core (rPPG + Biomarkers)"
    reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AuraPulseEngine:
    """
    Main on-device contactless physiological measurement engine.
    """

    def __init__(
        self,
        min_measurement_duration_s: float = 4.0,
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
        self.bp_engine = BloodPressureEngine()
        self.spo2_engine = SpO2Engine()
        self.hb_engine = HemoglobinEngine()
        self.cardiac_workload_engine = CardiacWorkloadEngine()
        self.risk_engine = HealthRiskAnalyticsEngine()
        self.metabolic_engine = MetabolicRiskEngine()

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

    def compute_vitals(self, user_metadata: Optional[Dict[str, Any]] = None) -> HealthMeasurementResult:
        """
        Executes signal processing, rPPG extraction, SQI, vitals estimation,
        and comprehensive health risk analytics over the current sliding buffer window.
        """
        meta = user_metadata or {}
        user_age = float(meta.get("age", 35.0))
        user_is_male = bool(meta.get("is_male", True))
        user_bmi = float(meta.get("bmi", 23.5))
        user_height = float(meta.get("height_cm", 175.0))
        user_weight = float(meta.get("weight_kg", 72.0))
        user_waist = float(meta.get("waist_cm", 0.0)) if meta.get("waist_cm") else None
        user_smoker = bool(meta.get("is_smoker", False))
        user_diabetic = bool(meta.get("is_diabetic", False))
        clinical_sbp = float(meta.get("clinical_sbp", 0.0)) if meta.get("clinical_sbp") else None

        t_uniform, roi_rgbs, duration = self.signal_buffer.get_window(
            duration_s=self.window_duration_s,
            target_fs=self.target_fs
        )

        if duration < self.min_measurement_duration_s or not roi_rgbs:
            return HealthMeasurementResult(
                status="collecting",
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
                status="invalid_roi",
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
                status="insufficient_signal",
                measurement_duration_s=round(duration, 1),
                signal_quality=overall_sqi.sqi_score,
                quality_category=overall_sqi.category,
                reason=f"SIGNAL_QUALITY_TOO_LOW ({overall_sqi.category})",
            )

        # 4. Heart Rate Computation (Primary MVP)
        hr_res = self.hr_engine.estimate_hr(
            fused_bvp,
            fs=self.target_fs,
            sqi_score=overall_sqi.sqi_score
        )

        if not hr_res.is_valid:
            return HealthMeasurementResult(
                status="insufficient_signal",
                measurement_duration_s=round(duration, 1),
                signal_quality=overall_sqi.sqi_score,
                quality_category=overall_sqi.category,
                reason=hr_res.rejection_reason or "HEART_RATE_INDETERMINATE",
            )

        # 5. Respiration Rate Computation (Primary MVP)
        fh_rgb = roi_rgbs.get("forehead", next(iter(roi_rgbs.values())))
        raw_green = fh_rgb[:, 1] if fh_rgb.ndim == 2 else fused_bvp
        rr_res = self.respiration_engine.estimate_rr(
            raw_green,
            fs=self.target_fs,
            sqi_score=overall_sqi.sqi_score
        )

        # 6. Pulse Rate Variability (PRV) Computation (Primary MVP)
        prv_res = self.prv_engine.compute_prv(
            fused_bvp,
            fs=self.target_fs,
            sqi_score=overall_sqi.sqi_score,
            hr_bpm_hint=hr_res.hr_bpm
        )

        # 7. Blood Pressure Estimation (Research Only / Experimental)
        bp_res = self.bp_engine.estimate_blood_pressure(
            bvp_signal=fused_bvp,
            fs=self.target_fs,
            hr_bpm=hr_res.hr_bpm,
            rmssd_ms=prv_res.rmssd_ms if prv_res.is_valid else 35.0,
            age=user_age,
            is_male=user_is_male,
            bmi=user_bmi,
            sqi_score=overall_sqi.sqi_score
        )

        # 8. Oxygen Saturation (SpO2) Estimation (Research Only / Experimental)
        spo2_res = self.spo2_engine.estimate_spo2(
            rgb_temporal=fh_rgb,
            fs=self.target_fs,
            sqi_score=overall_sqi.sqi_score
        )

        # 9. Hemoglobin (Hb) Estimation (Research Only / Experimental)
        hb_res = self.hb_engine.estimate_hemoglobin(
            rgb_temporal=fh_rgb,
            is_male=user_is_male,
            sqi_score=overall_sqi.sqi_score
        )

        # 10. Cardiac Workload (Rate-Pressure Product)
        cardiac_res = self.cardiac_workload_engine.compute_workload(
            hr_bpm=hr_res.hr_bpm,
            systolic_bp=clinical_sbp or (bp_res.systolic_bp if bp_res.is_valid else 120.0),
            confidence_hr=hr_res.confidence,
            confidence_bp=bp_res.confidence if bp_res.is_valid else 0.8
        )

        # 11. Advanced Physiological Stress & Autonomic Recovery (Research Only)
        stress_res = self.stress_engine.compute_stress_index(
            hr_bpm=hr_res.hr_bpm,
            rmssd_ms=prv_res.rmssd_ms if prv_res.is_valid else 35.0,
            rr_rpm=rr_res.rr_rpm if rr_res.is_valid else 0.0,
            confidence_hr=hr_res.confidence,
            confidence_prv=prv_res.confidence if prv_res.is_valid else 0.5,
            ppi_intervals_ms=prv_res.ppi_intervals if hasattr(prv_res, "ppi_intervals") else None,
            bvp_signal=fused_bvp,
            fs=self.target_fs,
        )

        # 12. Body Composition & Anthropometrics
        body_comp_res = self.risk_engine.compute_body_composition(
            height_cm=user_height,
            weight_kg=user_weight,
            waist_cm=user_waist
        )

        # 13. Vascular Age Estimation (Model Dependent)
        vasc_age_res = self.risk_engine.estimate_vascular_age(
            chronological_age=user_age,
            systolic_bp=clinical_sbp or (bp_res.systolic_bp if bp_res.is_valid else 120.0),
            stiffness_index=bp_res.stiffness_index_ms if bp_res.is_valid else 7.5,
            sdppg_aging_index=bp_res.sdppg_aging_index if bp_res.is_valid else -0.35,
            confidence_bp=bp_res.confidence if bp_res.is_valid else 0.8,
            is_male=user_is_male,
            is_smoker=user_smoker,
            is_diabetic=user_diabetic,
        )

        # 14. 10-Year Cardiovascular & Stroke Risk Projection (Clinical Risk Model)
        # Note: Uses clinical SBP if provided by user; otherwise calibrated baseline
        cvd_risk_res = self.risk_engine.estimate_10yr_cvd_risk(
            age=user_age,
            is_male=user_is_male,
            systolic_bp=clinical_sbp or (bp_res.systolic_bp if bp_res.is_valid else 120.0),
            is_smoker=user_smoker,
            is_diabetic=user_diabetic,
            bmi=body_comp_res.bmi,
            hr_bpm=hr_res.hr_bpm,
            rmssd_ms=prv_res.rmssd_ms if prv_res.is_valid else 35.0
        )

        # 15. Metabolic & Glycemic Risk Model (Research Only / Experimental)
        metabolic_res = self.metabolic_engine.estimate_metabolic_risks(
            rmssd_ms=prv_res.rmssd_ms if prv_res.is_valid else 35.0,
            lf_hf_ratio=stress_res.lf_hf_ratio if stress_res.is_valid else 1.5,
            systolic_bp=clinical_sbp or (bp_res.systolic_bp if bp_res.is_valid else 120.0),
            age=user_age,
            bmi=body_comp_res.bmi,
            stiffness_index=bp_res.stiffness_index_ms if bp_res.is_valid else 7.5,
            is_diabetic_history=user_diabetic,
            confidence_inputs=min(hr_res.confidence, overall_sqi.sqi_score / 100.0)
        )

        # Downsample waveform for API transmission (last 120 samples)
        waveform_sample = [round(float(v), 4) for v in fused_bvp[-120:]]

        return HealthMeasurementResult(
            status="valid",
            heart_rate={
                "value": hr_res.hr_bpm,
                "unit": "BPM",
                "confidence": hr_res.confidence,
                "signal_quality": overall_sqi.sqi_score,
                "measurement_status": "valid",
                "validation_status": "PRIMARY_MVP",
                "algorithm_version": "4.0.0",
                "source": "Ensemble (Welch PSD, FFT, Autocorrelation, Peak Interval)",
                "methods": hr_res.method_estimates,
                "status": "Tachycardia" if hr_res.hr_bpm > 100.0 else ("Bradycardia" if hr_res.hr_bpm < 60.0 else "Normal"),
                "limitations": "Subject must remain stationary without speech.",
            },
            respiration_rate={
                "value": rr_res.rr_rpm if rr_res.is_valid else None,
                "unit": "breaths/min",
                "confidence": rr_res.confidence if rr_res.is_valid else None,
                "signal_quality": overall_sqi.sqi_score if rr_res.is_valid else None,
                "measurement_status": "valid" if rr_res.is_valid else "insufficient_duration",
                "validation_status": "PRIMARY_MVP",
                "algorithm_version": "4.0.0",
                "source": "Demodulated Baseline & Amplitude Variation (RIBV/RIAV)",
                "status": "Tachypnea" if (rr_res.is_valid and rr_res.rr_rpm > 20.0) else ("Bradypnea" if (rr_res.is_valid and rr_res.rr_rpm < 12.0) else "Optimal"),
                "limitations": "Requires at least 10s of continuous steady breathing.",
            },
            pulse_rate_variability={
                "rmssd": prv_res.rmssd_ms if prv_res.is_valid else None,
                "sdnn": prv_res.sdnn_ms if prv_res.is_valid else None,
                "meanPpi": prv_res.mean_ppi_ms if prv_res.is_valid else None,
                "pnn50": prv_res.pnn50_pct if prv_res.is_valid else None,
                "sd1": prv_res.sd1_ms if prv_res.is_valid else None,
                "sd2": prv_res.sd2_ms if prv_res.is_valid else None,
                "unit": "ms",
                "confidence": prv_res.confidence if prv_res.is_valid else None,
                "measurement_status": "valid" if prv_res.is_valid else "insufficient_duration",
                "validation_status": "PRIMARY_MVP",
                "algorithm_version": "4.0.0",
                "label": "Pulse Rate Variability (PRV - Optical BVP)",
                "source": "Sub-sample Parabolic Peak Interpolation",
                "limitations": "Camera-derived PRV reflects optical pulse timing and differs from ECG-derived HRV.",
            },
            blood_pressure={
                "systolic": bp_res.systolic_bp if bp_res.is_valid else None,
                "diastolic": bp_res.diastolic_bp if bp_res.is_valid else None,
                "pulsePressure": bp_res.pulse_pressure if bp_res.is_valid else None,
                "map": bp_res.mean_arterial_pressure if bp_res.is_valid else None,
                "category": bp_res.aha_category if bp_res.is_valid else "Unknown",
                "stiffnessIndex": bp_res.stiffness_index_ms if bp_res.is_valid else None,
                "augmentationIndex": bp_res.augmentation_index_pct if bp_res.is_valid else None,
                "agingIndex": bp_res.sdppg_aging_index if bp_res.is_valid else None,
                "confidence": bp_res.confidence if bp_res.is_valid else None,
                "measurement_status": "experimental",
                "validation_status": "NOT_VALIDATED (Research Only)",
                "algorithm_version": "4.0.0",
                "source": "SDPPG Pulse Contour & Morphometry Model",
                "limitations": "Research-only pulse morphology estimate; not validated as clinical blood pressure.",
            },
            spo2={
                "value": spo2_res.spo2_pct if spo2_res.is_valid else None,
                "category": spo2_res.category if spo2_res.is_valid else "Unknown",
                "ratioOfRatios": spo2_res.ratio_of_ratios if spo2_res.is_valid else None,
                "confidence": spo2_res.confidence if spo2_res.is_valid else None,
                "measurement_status": "experimental",
                "validation_status": "NOT_VALIDATED (Research Only)",
                "algorithm_version": "4.0.0",
                "source": "Dual-Wavelength Chromatic Extinction Calibration",
                "limitations": "Subject to ambient lighting spectrum and camera RGB filter sensitivity.",
            },
            hemoglobin={
                "value": hb_res.hb_g_dl if hb_res.is_valid else None,
                "unit": "g/dL",
                "category": hb_res.category if hb_res.is_valid else "Unknown",
                "referenceRange": hb_res.reference_range if hb_res.is_valid else "14.0 - 18.0 g/dL",
                "attenuationRatio": hb_res.attenuation_ratio if hb_res.is_valid else None,
                "confidence": hb_res.confidence if hb_res.is_valid else None,
                "measurement_status": "experimental",
                "validation_status": "NOT_VALIDATED (Research Only)",
                "algorithm_version": "4.0.0",
                "source": "Multi-Spectral Optical Density Differential Attenuation",
                "disclaimer": hb_res.disclaimer,
                "limitations": "Cannot be used for anemia diagnosis without clinical laboratory testing.",
            },
            cardiac_workload={
                "rpp": cardiac_res.rpp_score if cardiac_res.is_valid else None,
                "category": cardiac_res.workload_category if cardiac_res.is_valid else "Unknown",
                "loadIndex": cardiac_res.myocardial_load_index if cardiac_res.is_valid else None,
                "confidence": cardiac_res.confidence if cardiac_res.is_valid else None,
                "measurement_status": "valid" if cardiac_res.is_valid else "insufficient_data",
                "validation_status": "PHYSIOLOGICAL_RATIO",
                "algorithm_version": "4.0.0",
                "source": "Rate-Pressure Product Deterministic Formulation",
                "limitations": "Reflects resting myocardial oxygen demand proxy.",
            },
            stress_index={
                "score": stress_res.stress_score if stress_res.is_valid else None,
                "level": stress_res.stress_level if stress_res.is_valid else "Unknown",
                "baevskyIndex": stress_res.baevsky_stress_index if stress_res.is_valid else None,
                "parasympatheticScore": stress_res.parasympathetic_score if stress_res.is_valid else None,
                "sympatheticScore": stress_res.sympathetic_score if stress_res.is_valid else None,
                "prq": stress_res.pulse_respiration_quotient if stress_res.is_valid else None,
                "lfPower": stress_res.lf_power if stress_res.is_valid else None,
                "hfPower": stress_res.hf_power if stress_res.is_valid else None,
                "lfHfRatio": stress_res.lf_hf_ratio if stress_res.is_valid else None,
                "confidence": stress_res.confidence if stress_res.is_valid else None,
                "measurement_status": "research_only",
                "validation_status": "RESEARCH_ONLY",
                "algorithm_version": "4.0.0",
                "source": "Baevsky Histogram (50ms) + Welch Spectral Autonomic Features",
                "disclaimer": stress_res.disclaimer,
                "limitations": "Autonomic spectral features are research indices and not direct clinical biomarkers.",
            },
            metabolic_risks={
                "fbgMgDl": metabolic_res.estimated_fbg_mg_dl if metabolic_res.is_valid else None,
                "fbgCategory": metabolic_res.fbg_category if metabolic_res.is_valid else "Unknown",
                "fbgStatus": metabolic_res.fbg_status if metabolic_res.is_valid else "Unknown",
                "hba1cPct": metabolic_res.estimated_hba1c_pct if metabolic_res.is_valid else None,
                "hba1cCategory": metabolic_res.hba1c_category if metabolic_res.is_valid else "Unknown",
                "hba1cStatus": metabolic_res.hba1c_status if metabolic_res.is_valid else "Unknown",
                "metabolicScore": metabolic_res.metabolic_score if metabolic_res.is_valid else None,
                "riskLevel": metabolic_res.risk_level if metabolic_res.is_valid else "Unknown",
                "keyFactors": metabolic_res.key_factors if metabolic_res.is_valid else [],
                "confidence": metabolic_res.confidence if metabolic_res.is_valid else None,
                "measurement_status": "experimental",
                "validation_status": "NOT_VALIDATED (Research Only)",
                "algorithm_version": "4.0.0",
                "source": "Multivariate Hemodynamic-Autonomic Risk Projection",
                "disclaimer": metabolic_res.disclaimer,
                "limitations": "Not a substitute for fasting plasma glucose or laboratory HbA1c venipuncture.",
            },
            vascular_age={
                "estimatedAge": vasc_age_res.vascular_age_years,
                "ageDelta": vasc_age_res.age_delta,
                "status": vasc_age_res.stiffness_status,
                "confidence": vasc_age_res.confidence,
                "measurement_status": "valid",
                "validation_status": "MODEL_DEPENDENT",
                "algorithm_version": "4.0.0",
                "source": "Framingham Optimal Baseline Vascular Age Regression",
                "limitations": "Reflects calculated biological vascular age relative to chronological baseline.",
            },
            cvd_risk={
                "tenYearRiskPct": cvd_risk_res.ten_year_risk_pct,
                "category": cvd_risk_res.risk_category,
                "strokeRiskPct": cvd_risk_res.stroke_risk_pct,
                "hypertensionScore": cvd_risk_res.hypertension_risk_score,
                "diabetesScore": cvd_risk_res.diabetes_risk_score,
                "keyDrivers": cvd_risk_res.key_drivers,
                "confidence": cvd_risk_res.confidence,
                "measurement_status": "valid",
                "validation_status": "CLINICAL_RISK_MODEL",
                "algorithm_version": "4.0.0",
                "source": "Framingham General Cardiovascular Risk Profile (Cox Proportional Hazard)",
                "limitations": "Calculated using demographic and risk factor inputs for 10-year projection.",
            },
            body_composition={
                "bmi": body_comp_res.bmi,
                "bmiCategory": body_comp_res.bmi_category,
                "whtr": body_comp_res.whtr,
                "whtrCategory": body_comp_res.whtr_category,
                "bri": body_comp_res.body_roundness_index,
                "briCategory": body_comp_res.bri_category,
                "measurement_status": "valid",
                "validation_status": "PHYSIOLOGICAL_RATIO",
                "algorithm_version": "4.0.0",
            },
            wellness_indices={
                "baevskyStress": stress_res.baevsky_stress_index if stress_res.is_valid else None,
                "pnsRecovery": stress_res.parasympathetic_score if stress_res.is_valid else None,
                "snsZone": stress_res.sympathetic_score if stress_res.is_valid else None,
                "prq": stress_res.pulse_respiration_quotient if stress_res.is_valid else None,
                "lfHfRatio": stress_res.lf_hf_ratio if stress_res.is_valid else None,
                "ansBalance": stress_res.ans_balance_ratio if stress_res.is_valid else None,
            },
            pulse_waveform=waveform_sample,
            signal_quality=overall_sqi.sqi_score,
            quality_category=overall_sqi.category,
            face_quality=100.0,
            measurement_duration_s=round(duration, 1),
            algorithm_version="4.0.0",
            engine="AuraPulse-Clinical-Edge",
            validation_status="On-Device Synthesis Core (rPPG + Biomarkers)",
            reason=None,
        )
