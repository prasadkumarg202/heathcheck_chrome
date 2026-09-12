"""
High-Fidelity Synthetic Physiological Signal and Video Generator for AuraPulse.
Generates ground-truth photoplethysmogram waveforms, respiratory modulation,
ambient lighting fluctuations, motion artifacts, and simulated camera frames.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Dict, Optional
import numpy as np
import cv2


@dataclass
class SyntheticGroundTruth:
    target_hr_bpm: float
    target_rr_rpm: float
    target_rmssd_ms: float
    snr_db: float
    duration_s: float
    fs: float
    timestamps: np.ndarray
    true_bvp: np.ndarray
    true_respiration: np.ndarray
    simulated_rgb: Dict[str, np.ndarray]  # ROI -> (N, 3) array


class SyntheticSignalGenerator:
    """
    Simulates realistic optical absorption curves in human skin based on modified
    Lambert-Beer law and two-chamber cardiac pulse modeling.
    """

    def __init__(self, fs: float = 30.0):
        self.fs = fs

    def generate_signal(
        self,
        duration_s: float = 30.0,
        hr_bpm: float = 72.0,
        rr_rpm: float = 16.0,
        rmssd_ms: float = 35.0,
        snr_db: float = 15.0,
        skin_tone_fitzpatrick: int = 3,
        motion_artifact_level: float = 0.0,
        light_flicker_level: float = 0.0,
    ) -> SyntheticGroundTruth:
        """
        Generates clean ground-truth physiological signals with customizable noise and optical dynamics.
        """
        n_samples = int(duration_s * self.fs)
        t = np.linspace(0, duration_s, n_samples, endpoint=False)

        # 1. Cardiac Fundamental & Harmonics
        f_hr = hr_bpm / 60.0
        # Realistic PPG pulse waveform (systolic + dicrotic notch)
        cardiac_wave = (
            1.0 * np.sin(2 * np.pi * f_hr * t) +
            0.4 * np.sin(4 * np.pi * f_hr * t + 0.6) +
            0.15 * np.sin(6 * np.pi * f_hr * t + 1.2)
        )

        # 2. Respiration Modulation (RIBV - Baseline & RIAV - Amplitude)
        f_rr = rr_rpm / 60.0
        resp_wave = np.sin(2 * np.pi * f_rr * t)
        
        # Combined modulated BVP
        modulated_bvp = (1.0 + 0.15 * resp_wave) * cardiac_wave + 0.25 * resp_wave

        # 3. Base Skin Reflectance depending on Fitzpatrick scale (I = light, VI = dark)
        # Base RGB levels (RGB order)
        fitzpatrick_bases = {
            1: np.array([230.0, 190.0, 175.0]),  # Type I: Very Fair
            2: np.array([215.0, 170.0, 150.0]),  # Type II: Fair
            3: np.array([195.0, 150.0, 125.0]),  # Type III: Medium/Olive
            4: np.array([170.0, 125.0, 95.0]),   # Type IV: Olive/Brown
            5: np.array([130.0, 90.0, 65.0]),    # Type V: Dark Brown
            6: np.array([90.0, 60.0, 45.0]),     # Type VI: Very Dark
        }
        base_rgb = fitzpatrick_bases.get(skin_tone_fitzpatrick, fitzpatrick_bases[3])

        # Hemoglobin absorption coefficient: Green absorbs strongest, Red least
        # Modulation coefficients for [R, G, B]
        absorption_coeffs = np.array([0.002, 0.012, 0.005])

        # Perfusion attenuation for darker skin types (due to melanin absorption)
        melanin_factor = 1.0 - (skin_tone_fitzpatrick - 1) * 0.08
        effective_absorption = absorption_coeffs * melanin_factor

        # 4. Generate Multi-ROI RGB signals
        simulated_rgb: Dict[str, np.ndarray] = {}
        rois = {
            "forehead": 1.0,      # Strongest signal
            "left_cheek": 0.85,   # Moderate signal
            "right_cheek": 0.85,  # Moderate signal
        }

        # Noise calculation from SNR
        sig_power = np.var(modulated_bvp)
        noise_power = sig_power / (10.0 ** (snr_db / 10.0))
        noise_std = np.sqrt(max(noise_power, 1e-8))

        for roi_name, amplitude_mult in rois.items():
            roi_rgb = np.zeros((n_samples, 3), dtype=np.float32)
            
            # Base color
            for c in range(3):
                # Optical modulation: I_c(t) = I_0 * (1 - coeff * bvp)
                channel_signal = base_rgb[c] * (1.0 - amplitude_mult * effective_absorption[c] * modulated_bvp)
                
                # Add Gaussian noise
                noise = np.random.normal(0, noise_std * base_rgb[c] * 0.005, size=n_samples)
                channel_signal += noise

                # Add slow ambient light drift / flicker
                if light_flicker_level > 0:
                    drift = light_flicker_level * np.sin(2 * np.pi * 0.05 * t) * base_rgb[c] * 0.05
                    flicker_50hz = light_flicker_level * 0.02 * np.sin(2 * np.pi * 50.0 * t) * base_rgb[c]
                    channel_signal += drift + flicker_50hz

                # Add motion artifacts (step baseline shift or sudden sway)
                if motion_artifact_level > 0:
                    motion_sway = motion_artifact_level * np.sin(2 * np.pi * 1.5 * t) * base_rgb[c] * 0.1
                    channel_signal += motion_sway

                roi_rgb[:, c] = channel_signal

            simulated_rgb[roi_name] = roi_rgb

        return SyntheticGroundTruth(
            target_hr_bpm=hr_bpm,
            target_rr_rpm=rr_rpm,
            target_rmssd_ms=rmssd_ms,
            snr_db=snr_db,
            duration_s=duration_s,
            fs=self.fs,
            timestamps=t,
            true_bvp=cardiac_wave,
            true_respiration=resp_wave,
            simulated_rgb=simulated_rgb,
        )

    def generate_synthetic_face_frame(
        self,
        ground_truth: SyntheticGroundTruth,
        sample_idx: int,
        width: int = 640,
        height: int = 480
    ) -> np.ndarray:
        """
        Creates a synthetic video frame with an anthropometric face and dynamic skin color modulation.
        """
        frame = np.ones((height, width, 3), dtype=np.uint8) * 220  # Neutral gray background

        # Face center & dimensions
        fx, fy = width // 2, height // 2
        fw, fh = int(width * 0.40), int(height * 0.55)

        # Get instantaneous modulated color from forehead ROI (RGB -> BGR)
        inst_rgb = ground_truth.simulated_rgb["forehead"][sample_idx]
        bgr_face = (int(np.clip(inst_rgb[2], 0, 255)),
                    int(np.clip(inst_rgb[1], 0, 255)),
                    int(np.clip(inst_rgb[0], 0, 255)))

        # Draw face ellipse
        cv2.ellipse(frame, (fx, fy), (fw // 2, fh // 2), 0, 0, 360, bgr_face, -1)

        # Draw eyes
        eye_y = int(fy - fh * 0.12)
        eye_lx = int(fx - fw * 0.22)
        eye_rx = int(fx + fw * 0.22)
        cv2.circle(frame, (eye_lx, eye_y), 12, (50, 50, 50), -1)
        cv2.circle(frame, (eye_rx, eye_y), 12, (50, 50, 50), -1)

        # Draw mouth
        mouth_y = int(fy + fh * 0.25)
        cv2.ellipse(frame, (fx, mouth_y), (int(fw * 0.18), 8), 0, 0, 360, (70, 70, 180), -1)

        return frame
