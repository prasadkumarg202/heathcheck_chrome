"""
Standard Physiological Benchmark Evaluator for Contactless Vitals.
Computes MAE, RMSE, MAPE, Pearson r, Signal-to-Noise Ratio (SNR dB),
and Bland-Altman statistical analysis against ECG / PPG ground-truth.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np
from scipy import signal, stats


@dataclass
class BenchmarkReport:
    mae: float
    rmse: float
    mape: float
    pearson_r: float
    p_value: float
    mean_error: float
    std_error: float
    bland_altman_lower_loa: float
    bland_altman_upper_loa: float
    snr_db_mean: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "MAE": round(self.mae, 3),
            "RMSE": round(self.rmse, 3),
            "MAPE_pct": round(self.mape, 3),
            "Pearson_r": round(self.pearson_r, 3),
            "p_value": self.p_value,
            "Mean_Error": round(self.mean_error, 3),
            "Std_Error": round(self.std_error, 3),
            "BlandAltman_LowerLoA": round(self.bland_altman_lower_loa, 3),
            "BlandAltman_UpperLoA": round(self.bland_altman_upper_loa, 3),
            "SNR_dB_Mean": round(self.snr_db_mean, 2),
        }


class VitalsBenchmarkEvaluator:
    """
    Evaluator comparing estimated contactless vitals vs ground-truth contact sensors.
    """

    @staticmethod
    def evaluate_predictions(
        ground_truth: np.ndarray,
        predictions: np.ndarray,
        snr_values: Optional[np.ndarray] = None
    ) -> BenchmarkReport:
        gt = np.asarray(ground_truth, dtype=np.float64)
        pred = np.asarray(predictions, dtype=np.float64)

        if len(gt) != len(pred) or len(gt) == 0:
            raise ValueError("Ground-truth and predictions must have identical non-zero length.")

        # 1. Error metrics
        errors = pred - gt
        abs_errors = np.abs(errors)
        mae = float(np.mean(abs_errors))
        rmse = float(np.sqrt(np.mean(errors ** 2)))
        mape = float(np.mean(abs_errors / (gt + 1e-8)) * 100.0)

        # 2. Pearson Correlation
        if len(gt) > 1 and np.std(gt) > 1e-6 and np.std(pred) > 1e-6:
            r_val, p_val = stats.pearsonr(gt, pred)
        else:
            r_val, p_val = 1.0, 0.0

        # 3. Bland-Altman Limits of Agreement
        mean_err = float(np.mean(errors))
        std_err = float(np.std(errors, ddof=1) if len(errors) > 1 else 0.0)
        lower_loa = mean_err - 1.96 * std_err
        upper_loa = mean_err + 1.96 * std_err

        # 4. SNR mean
        snr_mean = float(np.mean(snr_values)) if snr_values is not None and len(snr_values) > 0 else 0.0

        return BenchmarkReport(
            mae=mae,
            rmse=rmse,
            mape=mape,
            pearson_r=float(r_val),
            p_value=float(p_val),
            mean_error=mean_err,
            std_error=std_err,
            bland_altman_lower_loa=lower_loa,
            bland_altman_upper_loa=upper_loa,
            snr_db_mean=snr_mean
        )

    @staticmethod
    def compute_bvp_snr_db(bvp_signal: np.ndarray, fs: float = 30.0, hr_bpm_gt: float = 75.0) -> float:
        """
        Computes SNR (in dB) of a BVP pulse waveform relative to ground truth heart rate frequency.
        """
        N = len(bvp_signal)
        if N < int(fs * 2):
            return 0.0

        freqs, psd = signal.welch(bvp_signal, fs=fs, nperseg=min(N, int(fs * 8)))
        f_hr = hr_bpm_gt / 60.0

        # Band: [0.5, 4.0] Hz
        band_mask = (freqs >= 0.5) & (freqs <= 4.0)
        signal_mask = (np.abs(freqs - f_hr) <= 0.18) | (np.abs(freqs - 2 * f_hr) <= 0.18)

        signal_power = np.sum(psd[band_mask & signal_mask])
        noise_power = np.sum(psd[band_mask & (~signal_mask)]) + 1e-8

        snr = float(10.0 * np.log10(max(signal_power / noise_power, 1e-4)))
        return snr
