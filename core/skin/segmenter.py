"""
Multi-Color-Space Skin Segmentation Engine for AuraPulse.
Accurately segments facial skin across Fitzpatrick skin types I through VI
using fused YCbCr, HSV, and normalized RGB color-space boundaries.
"""

from __future__ import annotations
from typing import Tuple
import cv2
import numpy as np


class SkinSegmenter:
    """
    Robust skin color segmentation across diverse demographics and lighting.
    Fuses chromatic boundaries in YCbCr and HSV to reject hair, beard,
    glasses, eyes, lips, and background clutter.
    """

    def __init__(self, use_morphology: bool = True):
        self.use_morphology = use_morphology
        self.morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    def segment(self, bgr_image: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Takes a BGR image patch (e.g. ROI) and returns a binary skin mask (uint8: 0 or 255)
        and the percentage of valid skin pixels (0.0 to 100.0).
        """
        if bgr_image is None or bgr_image.size == 0:
            return np.zeros((0, 0), dtype=np.uint8), 0.0

        # Convert to YCbCr
        ycbcr = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = ycbcr[:, :, 0], ycbcr[:, :, 1], ycbcr[:, :, 2]

        # Standard YCrCb skin boundary (inclusive of Fitzpatrick I-VI under normal lighting)
        # Cr: [133, 175], Cb: [77, 130]
        # Darker skin tones (Fitzpatrick V-VI) typically exhibit slightly higher Cr and lower Cb relative to Y
        mask_ycbcr = (cr >= 130) & (cr <= 180) & (cb >= 75) & (cb <= 135) & (y >= 30)

        # Convert to HSV
        hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
        h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

        # Hue: 0 to 25 (reds-oranges-yellows), Saturation: 20 to 255, Value: 35 to 255
        mask_hsv = (h >= 0) & (h <= 28) & (s >= 20) & (s <= 250) & (v >= 35)

        # RGB Normalized Thresholding
        b = bgr_image[:, :, 0].astype(np.float32)
        g = bgr_image[:, :, 1].astype(np.float32)
        r = bgr_image[:, :, 2].astype(np.float32)

        # Skin Rule: R > G and R > B, and (R - G) >= 5
        mask_rgb = (r > g) & (g > b * 0.8) & ((r - g) >= 3) & (r > 40)

        # Fuse masks: YCbCr AND (HSV OR RGB)
        skin_bool = mask_ycbcr & (mask_hsv | mask_rgb)
        skin_mask = (skin_bool * 255).astype(np.uint8)

        if self.use_morphology and skin_mask.size > 0:
            skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, self.morph_kernel)
            skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, self.morph_kernel)

        total_pixels = skin_mask.size
        skin_pixels = int(np.count_nonzero(skin_mask))
        skin_pct = (skin_pixels / total_pixels * 100.0) if total_pixels > 0 else 0.0

        return skin_mask, skin_pct
