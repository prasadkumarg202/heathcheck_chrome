"""
Multi-Color-Space Semantic Skin Segmentation Engine for AuraPulse.
Features:
1. Fused YCbCr, HSV, and Normalized RGB chromatic boundaries across Fitzpatrick types I-VI.
2. Specular glare and over-saturation rejection (sweaty / direct reflection removal).
3. Semantic facial landmark exclusion (eyes, eyebrows, lips, and nostrils masking).
"""

from __future__ import annotations
from typing import Tuple, Optional, Any
import cv2
import numpy as np


class SkinSegmenter:
    """
    Enterprise skin color segmentation and semantic parsing engine.
    Fuses chromatic boundaries in YCbCr, HSV, and RGB to reject hair, beard,
    glasses, eyes, lips, and specular glare artifacts.
    """

    def __init__(self, use_morphology: bool = True, reject_specular_glare: bool = True):
        self.use_morphology = use_morphology
        self.reject_specular_glare = reject_specular_glare
        self.morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    def segment(self, bgr_image: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Takes a BGR image patch (e.g. ROI) and returns a binary skin mask (uint8: 0 or 255)
        and the percentage of valid skin pixels (0.0 to 100.0).
        """
        if bgr_image is None or bgr_image.size == 0:
            return np.zeros((0, 0), dtype=np.uint8), 0.0

        # 1. Convert to YCbCr
        ycbcr = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = ycbcr[:, :, 0], ycbcr[:, :, 1], ycbcr[:, :, 2]

        # Inclusive of Fitzpatrick I-VI under various ambient lighting
        mask_ycbcr = (cr >= 130) & (cr <= 180) & (cb >= 75) & (cb <= 135) & (y >= 30)

        # 2. Convert to HSV
        hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
        h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

        # Hue: 0 to 28 (reds-oranges-yellows), Saturation: 18 to 250, Value: 35 to 255
        mask_hsv = (h >= 0) & (h <= 28) & (s >= 18) & (s <= 250) & (v >= 35)

        # 3. Normalized RGB Boundaries
        b = bgr_image[:, :, 0].astype(np.float32)
        g = bgr_image[:, :, 1].astype(np.float32)
        r = bgr_image[:, :, 2].astype(np.float32)

        # Vascular skin rule: R > G and G > B*0.75, with minimum intensity
        mask_rgb = (r > g) & (g > b * 0.75) & ((r - g) >= 3) & (r > 38)

        # 4. Specular Glare & Direct Reflection Rejection
        # Specular reflections wash out pulsatile absorption (R,G,B saturated near 255 with low chroma)
        if self.reject_specular_glare:
            specular_mask = (v >= 245) & (s <= 20) | ((r >= 250) & (g >= 250) & (b >= 250))
        else:
            specular_mask = np.zeros(bgr_image.shape[:2], dtype=bool)

        # Fuse masks: (YCbCr AND (HSV OR RGB)) AND (NOT Specular)
        skin_bool = mask_ycbcr & (mask_hsv | mask_rgb) & (~specular_mask)
        skin_mask = (skin_bool * 255).astype(np.uint8)

        if self.use_morphology and skin_mask.size > 0:
            skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, self.morph_kernel)
            skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, self.morph_kernel)

        total_pixels = skin_mask.size
        skin_pixels = int(np.count_nonzero(skin_mask))
        skin_pct = (skin_pixels / total_pixels * 100.0) if total_pixels > 0 else 0.0

        return skin_mask, skin_pct

    def create_semantic_face_mask(
        self,
        frame: np.ndarray,
        landmarks: Optional[Any] = None
    ) -> np.ndarray:
        """
        Creates a high-precision semantic mask for the entire face,
        segmenting skin while explicitly zeroing out eyes, eyebrows, lips, and nostrils.
        """
        h, w = frame.shape[:2]
        base_skin_mask, _ = self.segment(frame)

        if landmarks is None or not hasattr(landmarks, "landmarks_68"):
            return base_skin_mask

        pts = landmarks.landmarks_68
        if pts is None or len(pts) < 68:
            return base_skin_mask

        exclusion_mask = np.ones((h, w), dtype=np.uint8) * 255

        # 1. Mask Eyes (Left: 36-41, Right: 42-47)
        left_eye = np.array(pts[36:42], dtype=np.int32)
        right_eye = np.array(pts[42:48], dtype=np.int32)
        cv2.fillPoly(exclusion_mask, [left_eye, right_eye], 0)

        # 2. Mask Eyebrows (Left: 17-21, Right: 22-26)
        left_eyebrow = np.array(pts[17:22], dtype=np.int32)
        right_eyebrow = np.array(pts[22:27], dtype=np.int32)
        cv2.fillPoly(exclusion_mask, [left_eyebrow, right_eyebrow], 0)

        # 3. Mask Lips (Outer: 48-59, Inner: 60-67)
        outer_lips = np.array(pts[48:60], dtype=np.int32)
        cv2.fillPoly(exclusion_mask, [outer_lips], 0)

        # 4. Mask Nostrils (31-35)
        nostrils = np.array(pts[31:36], dtype=np.int32)
        cv2.fillPoly(exclusion_mask, [nostrils], 0)

        # Combine chromatic skin segmentation with anatomical exclusion mask
        semantic_skin_mask = cv2.bitwise_and(base_skin_mask, exclusion_mask)
        return semantic_skin_mask
