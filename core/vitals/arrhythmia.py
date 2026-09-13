"""
Arrhythmia & Irregular Pulse Screening Engine for AuraPulse.
Features:
1. High-resolution Inter-Beat Interval (IBI) time series extraction.
2. Poincaré Plot geometric descriptors: SD1, SD2, SD1/SD2 ratio.
3. Pulse Rhythm Variability: Coefficient of Variation (CV%), Shannon Entropy, RMSSD.
4. Investigational Irregular Rhythm / Atrial Fibrillation (AFib) screening indicator.

SCIENTIFIC PRINCIPLE:
Atrial Fibrillation and frequent ectopic beats are characterized by irregular-irregular
pulse intervals leading to elevated CV (>12%) and dispersion along the identity line
in Poincaré scatter distributions (elevated SD1/SD2 ratio and Shannon Entropy).
"""

from __future__ import annotations
import math
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any
import numpy as np


@dataclass
class ArrhythmiaResult:
    rhythm_status: str               # "Regular Sinus Rhythm", "Irregular / Potential AFib", "Borderline Irregularity", "Indeterminate"
    is_irregular: bool               # True if irregularity criteria met
    cv_pct: float                    # Coefficient of Variation: (std(IBI) / mean(IBI)) * 100
    sd1_ms: float                    # Poincaré short-term orthogonal variability
    sd2_ms: float                    # Poincaré long-term identity line variability
    sd1_sd2_ratio: float             # SD1 / SD2 ratio
    shannon_entropy: float           # Shannon entropy of normalized IBI distribution
    rmssd_ms: float                  # Root Mean Square of Successive Differences
    ectopic_beat_count: int          # Count of premature/compensatory intervals
    pvc_pac_burden_pct: float = 0.0  # Percentage of premature/compensatory intervals
    poincare_points: Optional[List[List[float]]] = None # [[RR_n, RR_n+1], ...]
    num_intervals: int = 0           # Total beat intervals analyzed
    confidence: float = 0.0          # 0.0 to 1.0
    is_valid: bool = False
    recommendation: str = ""         # Clinical safe-harbor guidance
    disclaimer: str = (
        "Investigational contactless rhythm screening. Not certified as a primary diagnostic "
        "instrument for Atrial Fibrillation under FDA 510(k) or EU MDR."
    )
    rejection_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ArrhythmiaEngine:
    """
    Evaluates pulse rhythm regularity, Poincaré geometric distribution,
    and entropy for irregular pulse / AFib screening.
    """

    def __init__(
        self,
        min_intervals: int = 10,
        cv_threshold_pct: float = 12.0,
        sd_ratio_threshold: float = 0.85,
        entropy_threshold: float = 2.20
    ):
        self.min_intervals = min_intervals
        self.cv_threshold_pct = cv_threshold_pct
        self.sd_ratio_threshold = sd_ratio_threshold
        self.entropy_threshold = entropy_threshold

    def evaluate_rhythm(
        self,
        ppi_intervals_ms: Optional[List[float]],
        sqi_score: float = 75.0,
        hr_bpm_hint: Optional[float] = None
    ) -> ArrhythmiaResult:
        """
        Analyzes Inter-Beat Interval (IBI/PPI) series and returns comprehensive rhythm metrics.
        """
        if ppi_intervals_ms is None or len(ppi_intervals_ms) < self.min_intervals:
            return ArrhythmiaResult(
                rhythm_status="Indeterminate",
                is_irregular=False,
                cv_pct=0.0,
                sd1_ms=0.0,
                sd2_ms=0.0,
                sd1_sd2_ratio=0.0,
                shannon_entropy=0.0,
                rmssd_ms=0.0,
                ectopic_beat_count=0,
                num_intervals=len(ppi_intervals_ms) if ppi_intervals_ms else 0,
                confidence=0.0,
                is_valid=False,
                recommendation="Insufficient beat intervals captured. Maintain steady posture for at least 30 seconds.",
                rejection_reason="INSUFFICIENT_BEAT_INTERVALS",
            )

        ibis = np.array(ppi_intervals_ms, dtype=np.float64)

        # Filter extreme unphysiological outliers (< 300ms or > 2000ms, corresponding to 30-200 BPM)
        valid_mask = (ibis >= 300.0) & (ibis <= 2000.0)
        clean_ibis = ibis[valid_mask]

        if len(clean_ibis) < self.min_intervals:
            return ArrhythmiaResult(
                rhythm_status="Indeterminate",
                is_irregular=False,
                cv_pct=0.0,
                sd1_ms=0.0,
                sd2_ms=0.0,
                sd1_sd2_ratio=0.0,
                shannon_entropy=0.0,
                rmssd_ms=0.0,
                ectopic_beat_count=0,
                num_intervals=len(clean_ibis),
                confidence=0.0,
                is_valid=False,
                recommendation="Pulse intervals contain non-physiological artifacts. Ensure stable lighting and head position.",
                rejection_reason="UNPHYSIOLOGICAL_INTERVALS",
            )

        mean_ibi = float(np.mean(clean_ibis))
        std_ibi = float(np.std(clean_ibis, ddof=1)) if len(clean_ibis) > 1 else 0.0

        # 1. Coefficient of Variation (CV%)
        cv_pct = float((std_ibi / max(1.0, mean_ibi)) * 100.0)

        # 2. Successive Differences & RMSSD
        diffs = np.diff(clean_ibis)
        rmssd_ms = float(np.sqrt(np.mean(diffs ** 2))) if len(diffs) > 0 else 0.0

        # Ectopic beat detection (consecutive beat interval change > 20% from local median)
        median_ibi = float(np.median(clean_ibis))
        ectopic_count = int(np.count_nonzero(np.abs(clean_ibis - median_ibi) > (0.20 * median_ibi)))
        pvc_pac_burden = float(round((ectopic_count / max(1, len(clean_ibis))) * 100.0, 1))

        # 3. Poincaré Plot Analysis (SD1, SD2, SD1/SD2) & 2D Return Map Coordinates
        # SD1 = sqrt(0.5 * var(diffs))
        # SD2 = sqrt(2 * var(clean_ibis) - 0.5 * var(diffs))
        var_diff = float(np.var(diffs, ddof=1)) if len(diffs) > 1 else 0.0
        var_ibi = float(np.var(clean_ibis, ddof=1)) if len(clean_ibis) > 1 else 0.0

        sd1 = math.sqrt(max(0.0, 0.5 * var_diff))
        sd2_sq = max(0.0, 2.0 * var_ibi - 0.5 * var_diff)
        sd2 = math.sqrt(sd2_sq)

        sd1_sd2_ratio = float(sd1 / max(1e-4, sd2))

        # Generate [RR_n, RR_n+1] scatter coordinates for real-time visualization
        poincare_pairs = []
        if len(clean_ibis) >= 2:
            for i in range(len(clean_ibis) - 1):
                poincare_pairs.append([float(round(clean_ibis[i], 1)), float(round(clean_ibis[i+1], 1))])

        # 4. Shannon Entropy on 16-bin normalized histogram
        hist_bins = np.linspace(
            max(300.0, mean_ibi - 3.0 * max(10.0, std_ibi)),
            min(2000.0, mean_ibi + 3.0 * max(10.0, std_ibi)),
            17
        )
        counts, _ = np.histogram(clean_ibis, bins=hist_bins)
        probs = counts / max(1, len(clean_ibis))
        nonzero_probs = probs[probs > 0]
        shannon_ent = float(-np.sum(nonzero_probs * np.log2(nonzero_probs)))

        # 5. Diagnostic Classification Logic
        sqi_factor = min(1.0, max(0.0, sqi_score / 100.0))
        confidence = float(round(min(0.95, (len(clean_ibis) / 30.0) * sqi_factor), 2))

        # AFib / Irregular criteria: Elevated CV (> 12%), elevated SD1/SD2 (> 0.85)
        is_high_cv = cv_pct >= self.cv_threshold_pct
        is_high_sd_ratio = sd1_sd2_ratio >= self.sd_ratio_threshold

        if is_high_cv and is_high_sd_ratio:
            rhythm_status = "Irregular / Potential AFib"
            is_irregular = True
            recommendation = (
                "Elevated beat-to-beat variability detected. If persistent or accompanied by "
                "palpitations, dizziness, or shortness of breath, consult a physician for standard 12-lead ECG evaluation."
            )
        elif is_high_cv or (ectopic_count >= 3 and is_high_sd_ratio):
            rhythm_status = "Borderline Irregularity"
            is_irregular = False
            recommendation = (
                "Mild pulse irregularities or isolated ectopic beats detected. Ensure resting state and repeat measurement."
            )
        else:
            rhythm_status = "Regular Sinus Rhythm"
            is_irregular = False
            recommendation = "Normal rhythmic pulse intervals observed within expected physiological bounds."

        return ArrhythmiaResult(
            rhythm_status=rhythm_status,
            is_irregular=is_irregular,
            cv_pct=float(round(cv_pct, 2)),
            sd1_ms=float(round(sd1, 2)),
            sd2_ms=float(round(sd2, 2)),
            sd1_sd2_ratio=float(round(sd1_sd2_ratio, 3)),
            shannon_entropy=float(round(shannon_ent, 3)),
            rmssd_ms=float(round(rmssd_ms, 2)),
            ectopic_beat_count=ectopic_count,
            pvc_pac_burden_pct=pvc_pac_burden,
            poincare_points=poincare_pairs,
            num_intervals=len(clean_ibis),
            confidence=confidence,
            is_valid=True,
            recommendation=recommendation,
            rejection_reason=None,
        )
