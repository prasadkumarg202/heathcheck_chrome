"""
Multi-ROI and Multi-Algorithm Signal Fusion Engine for AuraPulse.
Combines signals and vital estimates from Forehead, Left Cheek, and Right Cheek
and algorithms (POS, CHROM, Green) weighted by individual SQI and spatial signal strength.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class FusedSignalResult:
    fused_bvp: np.ndarray
    roi_weights: Dict[str, float]
    algorithm_weights: Dict[str, float]
    composite_sqi: float
    is_valid: bool


class SignalFusionEngine:
    """
    Weighted ensemble fusion across anatomical ROIs and extraction algorithms.
    Aligns phase polarity across ROIs to prevent destructive signal cancellation.
    """

    def fuse_roi_signals(
        self,
        roi_signals: Dict[str, np.ndarray],
        roi_sqis: Dict[str, float]
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Fuses 1D rPPG signals across Forehead, Left Cheek, and Right Cheek.
        """
        valid_rois = [name for name, sqi in roi_sqis.items() if sqi >= 20.0 and len(roi_signals.get(name, [])) > 0]
        if not valid_rois:
            first_sig = next(iter(roi_signals.values())) if roi_signals else np.array([])
            return np.zeros_like(first_sig), {k: 0.0 for k in roi_signals}

        # Select anchor ROI with highest SQI
        anchor_roi = max(valid_rois, key=lambda name: roi_sqis[name])
        anchor_sig = roi_signals[anchor_roi]
        std_anchor = np.std(anchor_sig)
        anchor_norm = (anchor_sig - np.mean(anchor_sig)) / (std_anchor + 1e-8)

        # Softmax / normalized exponential weighting on SQI
        sqi_vals = np.array([roi_sqis[name] for name in valid_rois])
        exp_weights = np.exp((sqi_vals - np.max(sqi_vals)) / 20.0)
        norm_weights = exp_weights / np.sum(exp_weights)

        sig_len = len(anchor_sig)
        fused = np.zeros(sig_len, dtype=np.float32)
        weight_dict: Dict[str, float] = {k: 0.0 for k in roi_signals}

        for idx, name in enumerate(valid_rois):
            w = float(norm_weights[idx])
            weight_dict[name] = round(w, 3)
            sig = roi_signals[name]
            std = np.std(sig)
            if std > 1e-6:
                norm_sig = (sig - np.mean(sig)) / std
                
                # Phase alignment with anchor ROI
                corr_matrix = np.corrcoef(norm_sig, anchor_norm)
                corr_val = corr_matrix[0, 1] if not np.isnan(corr_matrix[0, 1]) else 1.0
                if corr_val < 0.0:
                    norm_sig = -norm_sig

                fused += w * norm_sig

        return fused, weight_dict

    def fuse_vital_estimates(
        self,
        estimates: List[Tuple[float, float, float]]
    ) -> Tuple[float, float]:
        """
        Ensemble voting and outlier rejection for vital numbers.
        """
        if not estimates:
            return 0.0, 0.0

        vals = np.array([e[0] for e in estimates if e[0] > 0])
        confs = np.array([e[1] for e in estimates if e[0] > 0])
        sqis = np.array([e[2] for e in estimates if e[0] > 0])

        if len(vals) == 0:
            return 0.0, 0.0

        if len(vals) == 1:
            return float(vals[0]), float(confs[0])

        med = np.median(vals)
        dev = np.abs(vals - med)
        valid_mask = dev <= 8.0

        if not np.any(valid_mask):
            valid_mask = np.ones(len(vals), dtype=bool)

        filt_vals = vals[valid_mask]
        filt_confs = confs[valid_mask]
        filt_sqis = sqis[valid_mask]

        w = filt_confs * (filt_sqis / 100.0)
        if np.sum(w) > 0:
            w_norm = w / np.sum(w)
            fused_val = float(np.sum(filt_vals * w_norm))
            fused_conf = float(np.mean(filt_confs))
        else:
            fused_val = float(np.mean(filt_vals))
            fused_conf = float(np.mean(filt_confs))

        return round(fused_val, 1), round(fused_conf, 3)
