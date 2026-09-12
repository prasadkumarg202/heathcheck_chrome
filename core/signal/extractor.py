"""
Signal Extractor and Temporal Buffer for AuraPulse.
Manages continuous multi-ROI RGB time-series buffers, timestamps,
sliding windows, and uniform interpolation across variable camera frame rates.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING
import numpy as np
from scipy import interpolate

if TYPE_CHECKING:
    from core.roi.extractor import ROIData


@dataclass
class TemporalSignalBuffer:
    """
    Sliding window buffer for raw color signals and timestamps.
    """
    capacity: int = 1800  # ~60 seconds @ 30 FPS
    timestamps: List[float] = field(default_factory=list)
    signals_rgb: Dict[str, List[Tuple[float, float, float]]] = field(
        default_factory=lambda: {
            "forehead": [],
            "left_cheek": [],
            "right_cheek": [],
        }
    )
    is_valid_mask: Dict[str, List[bool]] = field(
        default_factory=lambda: {
            "forehead": [],
            "left_cheek": [],
            "right_cheek": [],
        }
    )

    def append(
        self,
        timestamp_s: float,
        roi_data: Dict[str, any]
    ) -> None:
        """Appends one frame's worth of ROI observations."""
        self.timestamps.append(timestamp_s)
        if len(self.timestamps) > self.capacity:
            self.timestamps.pop(0)

        for roi_name in ["forehead", "left_cheek", "right_cheek"]:
            if roi_name in roi_data:
                data = roi_data[roi_name]
                self.signals_rgb[roi_name].append(data.mean_rgb)
                self.is_valid_mask[roi_name].append(data.is_valid)
            else:
                self.signals_rgb[roi_name].append((0.0, 0.0, 0.0))
                self.is_valid_mask[roi_name].append(False)

            if len(self.signals_rgb[roi_name]) > self.capacity:
                self.signals_rgb[roi_name].pop(0)
                self.is_valid_mask[roi_name].pop(0)

    def get_window(
        self,
        duration_s: float = 10.0,
        target_fs: float = 30.0
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray], float]:
        """
        Extracts the most recent `duration_s` window of RGB signals,
        interpolated to a uniform sampling frequency `target_fs`.
        Returns:
            uniform_time: 1D array of time steps
            uniform_rgb: Dict[roi_name -> 2D array of shape (N, 3) representing [R, G, B]]
            actual_duration_s: duration of valid data available
        """
        n_samples = len(self.timestamps)
        if n_samples < 15:
            return np.array([]), {}, 0.0

        t_arr = np.array(self.timestamps)
        t_end = t_arr[-1]
        t_start = max(t_arr[0], t_end - duration_s)
        actual_duration = t_end - t_start

        if actual_duration < 2.0:
            return np.array([]), {}, actual_duration

        # Indices in the window
        mask = t_arr >= t_start
        t_win = t_arr[mask]

        if len(t_win) < 2 or (t_win[-1] - t_win[0]) < 1.0:
            return np.array([]), {}, actual_duration

        # Deduplicate timestamps to guarantee strictly increasing time sequence
        unique_mask = np.concatenate(([True], np.diff(t_win) > 1e-5))
        t_win_uniq = t_win[unique_mask]

        if len(t_win_uniq) < 2 or (t_win_uniq[-1] - t_win_uniq[0]) < 1.0:
            return np.array([]), {}, actual_duration

        # Uniform time grid
        num_target_points = max(int(actual_duration * target_fs), 10)
        t_uniform = np.linspace(t_win_uniq[0], t_win_uniq[-1], num_target_points)

        interpolated_signals: Dict[str, np.ndarray] = {}

        for roi_name in ["forehead", "left_cheek", "right_cheek"]:
            raw_rgb = np.array(self.signals_rgb[roi_name])[mask][unique_mask]  # (M, 3)
            val_mask = np.array(self.is_valid_mask[roi_name])[mask][unique_mask]

            if np.count_nonzero(val_mask) < (len(val_mask) * 0.4):
                # ROI is largely invalid/corrupted
                interpolated_signals[roi_name] = np.zeros((num_target_points, 3), dtype=np.float32)
                continue

            # Linear interpolation for R, G, B channels
            interp_rgb = np.zeros((num_target_points, 3), dtype=np.float32)
            for c in range(3):
                channel_data = raw_rgb[:, c]
                # Replace invalid values with nearest valid
                if not np.all(val_mask):
                    valid_idx = np.where(val_mask)[0]
                    if len(valid_idx) > 0:
                        channel_data = np.interp(np.arange(len(channel_data)), valid_idx, channel_data[valid_idx])
                
                f_interp = interpolate.interp1d(
                    t_win_uniq,
                    channel_data,
                    kind='linear',
                    bounds_error=False,
                    fill_value="extrapolate"
                )
                interp_rgb[:, c] = f_interp(t_uniform)

            interpolated_signals[roi_name] = interp_rgb

        return t_uniform, interpolated_signals, actual_duration

    def clear(self) -> None:
        """Clears all stored samples."""
        self.timestamps.clear()
        for roi in self.signals_rgb:
            self.signals_rgb[roi].clear()
            self.is_valid_mask[roi].clear()


# Alias for backward compatibility
SignalBuffer = TemporalSignalBuffer

