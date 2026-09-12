"""
Video Input Engine for AuraPulse.
Handles video streams, real-time camera feeds, high-precision timestamping,
actual FPS calculation, illumination checking, and frame drop detection.
"""

from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Optional, Tuple, Generator
import cv2
import numpy as np


@dataclass
class FrameMetadata:
    frame_index: int
    timestamp_s: float
    dt_s: float
    actual_fps: float
    width: int
    height: int
    mean_brightness: float
    is_dropped: bool = False
    is_blurry: bool = False
    blur_score: float = 0.0


class VideoInputEngine:
    """
    Robust Video Input Engine that manages frame acquisition from webcams or video files,
    dynamically measures inter-frame delta-t, and assesses basic camera quality.
    """

    def __init__(
        self,
        source: int | str = 0,
        target_fps: float = 30.0,
        min_brightness: float = 30.0,
        max_brightness: float = 240.0,
        blur_threshold: float = 20.0,
    ):
        self.source = source
        self.target_fps = target_fps
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness
        self.blur_threshold = blur_threshold

        self.cap: Optional[cv2.VideoCapture] = None
        self.frame_index = 0
        self.last_timestamp: Optional[float] = None
        self.start_timestamp: Optional[float] = None
        self.timestamps_history: list[float] = []
        self.fps_estimate = target_fps

    def open(self) -> bool:
        if isinstance(self.source, int):
            self.cap = cv2.VideoCapture(self.source, cv2.CAP_DSHOW if cv2.CAP_DSHOW else cv2.CAP_ANY)
        else:
            self.cap = cv2.VideoCapture(self.source)

        if not self.cap or not self.cap.isOpened():
            return False

        self.frame_index = 0
        self.last_timestamp = None
        self.start_timestamp = time.perf_counter()
        self.timestamps_history.clear()
        return True

    def close(self) -> None:
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.cap = None

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray], Optional[FrameMetadata]]:
        """
        Reads next frame from source with exact monotonic timestamp and quality metrics.
        """
        if not self.cap or not self.cap.isOpened():
            return False, None, None

        ret, frame = self.cap.read()
        current_time = time.perf_counter()

        if not ret or frame is None:
            return False, None, None

        if self.last_timestamp is None:
            dt = 1.0 / self.target_fps
            self.last_timestamp = current_time
        else:
            dt = current_time - self.last_timestamp
            self.last_timestamp = current_time

        # Avoid zero or negative dt
        dt = max(dt, 1e-4)

        self.timestamps_history.append(current_time)
        if len(self.timestamps_history) > 60:
            self.timestamps_history.pop(0)

        # Dynamic FPS calculation over moving window
        if len(self.timestamps_history) >= 2:
            time_span = self.timestamps_history[-1] - self.timestamps_history[0]
            if time_span > 0:
                self.fps_estimate = (len(self.timestamps_history) - 1) / time_span

        h, w = frame.shape[:2]

        # Illumination and Blur Metrics
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        mean_brightness = float(np.mean(gray))

        # Laplacian variance for motion blur detection
        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        is_blurry = blur_score < self.blur_threshold

        # Frame drop detection (> 2.0x expected period)
        expected_dt = 1.0 / self.target_fps
        is_dropped = dt > (1.8 * expected_dt)

        meta = FrameMetadata(
            frame_index=self.frame_index,
            timestamp_s=current_time - (self.start_timestamp or current_time),
            dt_s=dt,
            actual_fps=self.fps_estimate,
            width=w,
            height=h,
            mean_brightness=mean_brightness,
            is_dropped=is_dropped,
            is_blurry=is_blurry,
            blur_score=blur_score,
        )

        self.frame_index += 1
        return True, frame, meta

    def frame_generator(self) -> Generator[Tuple[np.ndarray, FrameMetadata], None, None]:
        """Convenience generator for streaming frames."""
        try:
            if not self.open():
                return
            while True:
                success, frame, meta = self.read_frame()
                if not success or frame is None or meta is None:
                    break
                yield frame, meta
        finally:
            self.close()
