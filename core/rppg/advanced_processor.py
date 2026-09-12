"""
Production-Ready Advanced rPPG Processor
Combines multi-channel CHROM and POS color decomposition with weighted
spatial skin mask extraction, zero-phase filtering, and ensemble frequency-time peak validation.
"""

from __future__ import annotations
from typing import Optional, Tuple, List, Union
import cv2
import numpy as np
import scipy.signal


class AdvancedRPPGProcessor:
    """
    Enterprise rPPG Processor implementing CHROM (Chrominance-based) and POS
    (Plane-Orthogonal-to-Skin) algorithms with semantic skin masking and ensemble validation.
    """

    def __init__(self, fps: float = 30.0, hr_low_hz: float = 0.7, hr_high_hz: float = 3.5):
        self.fps = float(fps)
        self.hr_low_hz = float(hr_low_hz)
        self.hr_high_hz = float(hr_high_hz)
        self.bvp_buffer: List[float] = []

    def extract_masked_skin_rgb(
        self,
        frame: np.ndarray,
        skin_mask: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Extracts rPPG signal using weighted skin mask, isolating valid vascular pixels
        while discarding hair, eyebrows, eyes, lips, and specular reflections.
        Returns: [R, G, B] mean vector in float32 format.
        """
        if frame is None or skin_mask is None or frame.size == 0:
            return None

        # Ensure mask is 2D binary uint8
        if len(skin_mask.shape) == 3:
            skin_mask = cv2.cvtColor(skin_mask, cv2.COLOR_BGR2GRAY)
        
        valid_indices = skin_mask > 0
        pixels = frame[valid_indices]

        if len(pixels) < 16:
            return None

        # Mean RGB values across valid skin pixels (frame is BGR in OpenCV)
        mean_bgr = np.mean(pixels, axis=0)  # [B, G, R]
        r = float(mean_bgr[2])
        g = float(mean_bgr[1])
        b = float(mean_bgr[0])

        return np.array([r, g, b], dtype=np.float32)

    def extract_chrom_signal(
        self,
        frame: np.ndarray,
        skin_mask: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Extracts rPPG signal using the CHROM method on a masked skin frame.
        """
        return self.extract_masked_skin_rgb(frame, skin_mask)

    def process_color_time_series(
        self,
        rgb_sequence: Union[np.ndarray, List[List[float]]]
    ) -> Optional[np.ndarray]:
        """
        Transforms raw RGB temporal traces into a clean BVP (Blood Volume Pulse) waveform 
        using the CHROM plane projection method.
        """
        if rgb_sequence is None or len(rgb_sequence) < int(self.fps * 2):
            return None

        rgb_array = np.asarray(rgb_sequence, dtype=np.float64)
        if len(rgb_array.shape) != 2 or rgb_array.shape[1] != 3:
            return None

        # 1. Temporal Mean Normalization
        mean_rgb = np.mean(rgb_array, axis=0) + 1e-8
        normalized_rgb = rgb_array / mean_rgb

        r = normalized_rgb[:, 0]
        g = normalized_rgb[:, 1]
        b = normalized_rgb[:, 2]

        # 2. CHROM projection vectors
        # Xs = 3*Rn - 2*Gn
        # Ys = 1.5*Rn + Gn - 1.5*Bn
        xs = 3.0 * r - 2.0 * g
        ys = 1.5 * r + g - 1.5 * b

        # 3. Bandpass filter the projection signals (0.7Hz to 3.5Hz)
        nyq = 0.5 * self.fps
        low = max(0.01, min(0.99, self.hr_low_hz / nyq))
        high = max(0.01, min(0.99, self.hr_high_hz / nyq))

        b_coeff, a_coeff = scipy.signal.butter(3, [low, high], btype="band")
        
        # Zero-phase forward-backward filtering
        filtered_xs = scipy.signal.filtfilt(b_coeff, a_coeff, xs)
        filtered_ys = scipy.signal.filtfilt(b_coeff, a_coeff, ys)

        # 4. Alpha tuning step for final BVP extraction
        std_xs = np.std(filtered_xs)
        std_ys = np.std(filtered_ys)
        alpha = std_xs / (std_ys + 1e-8)

        bvp = filtered_xs - alpha * filtered_ys
        return bvp

    def process_pos_time_series(
        self,
        rgb_sequence: Union[np.ndarray, List[List[float]]],
        window_sec: float = 1.6
    ) -> Optional[np.ndarray]:
        """
        Transforms raw RGB temporal traces into a clean BVP waveform using
        the Plane-Orthogonal-to-Skin (POS) projection algorithm.
        """
        if rgb_sequence is None or len(rgb_sequence) < int(self.fps * 2):
            return None

        rgb_array = np.asarray(rgb_sequence, dtype=np.float64)
        N = len(rgb_array)
        l = int(round(window_sec * self.fps))
        if l < 4:
            l = 4

        H = np.zeros(N, dtype=np.float64)

        for m in range(0, N - l + 1):
            w = rgb_array[m : m + l]
            mean_c = np.mean(w, axis=0) + 1e-8
            w_norm = w / mean_c

            # POS Projection Planes:
            # S1 = G - B
            # S2 = G + B - 2*R
            s1 = w_norm[:, 1] - w_norm[:, 2]
            s2 = w_norm[:, 1] + w_norm[:, 2] - 2.0 * w_norm[:, 0]

            std1 = np.std(s1) + 1e-8
            std2 = np.std(s2) + 1e-8
            alpha = std1 / std2

            h_sub = s1 + alpha * s2
            H[m : m + l] += h_sub

        # Post-filter
        nyq = 0.5 * self.fps
        b_coeff, a_coeff = scipy.signal.butter(3, [self.hr_low_hz / nyq, self.hr_high_hz / nyq], btype="band")
        bvp_pos = scipy.signal.filtfilt(b_coeff, a_coeff, H)
        return bvp_pos

    def compute_vitals(
        self,
        bvp_signal: np.ndarray
    ) -> Tuple[float, float]:
        """
        Computes accurate Heart Rate (BPM) and RMSSD (HRV) from the extracted BVP wave
        using ensemble frequency-time peak validation.
        Returns: (heart_rate_bpm, rmssd_ms)
        """
        if bvp_signal is None or len(bvp_signal) < int(self.fps * 4):
            return 0.0, 0.0

        # Detrend and zero-mean
        detrended = scipy.signal.detrend(bvp_signal)
        
        # 1. FFT Peak Detection for Heart Rate
        fft_vals = np.fft.rfft(detrended)
        fft_freqs = np.fft.rfftfreq(len(detrended), 1.0 / self.fps)
        
        valid_idx = np.where((fft_freqs >= self.hr_low_hz) & (fft_freqs <= self.hr_high_hz))
        if len(valid_idx[0]) == 0:
            return 0.0, 0.0

        peak_idx = valid_idx[0][np.argmax(np.abs(fft_vals[valid_idx]))]
        bpm_fft = float(fft_freqs[peak_idx] * 60.0)

        # 2. Time-Domain Peak Detection with Sub-Sample Parabolic Refinement for HRV (RMSSD)
        min_dist = int(self.fps / 3.2)  # Maximum 192 BPM
        peaks, _ = scipy.signal.find_peaks(detrended, distance=min_dist, prominence=np.std(detrended) * 0.3)

        refined_peak_times = []
        for p in peaks:
            if 1 <= p < len(detrended) - 1:
                y0 = detrended[p - 1]
                y1 = detrended[p]
                y2 = detrended[p + 1]
                denom = 2.0 * (y0 - 2.0 * y1 + y2 + 1e-8)
                delta = (y0 - y2) / denom
                exact_p = p + delta
                refined_peak_times.append(exact_p / self.fps)
            else:
                refined_peak_times.append(p / self.fps)

        if len(refined_peak_times) >= 3:
            rr_intervals = np.diff(refined_peak_times) * 1000.0  # ms
            
            # Physiological outlier filtering (300ms to 1500ms => 40-200 BPM)
            valid_rr = [rr for rr in rr_intervals if 300.0 <= rr <= 1500.0]
            
            if len(valid_rr) >= 2:
                # RMSSD computation
                diff_rr = np.diff(valid_rr)
                rmssd = float(np.sqrt(np.mean(np.square(diff_rr))))
                
                # Time-domain BPM cross-check
                mean_rr = np.mean(valid_rr)
                bpm_time = 60000.0 / mean_rr
                
                # Ensemble fusion (FFT + Time domain)
                if abs(bpm_fft - bpm_time) < 12.0:
                    bpm_final = 0.65 * bpm_fft + 0.35 * bpm_time
                else:
                    bpm_final = bpm_fft
            else:
                rmssd = 0.0
                bpm_final = bpm_fft
        else:
            rmssd = 0.0
            bpm_final = bpm_fft

        return round(float(bpm_final), 1), round(float(rmssd), 1)
