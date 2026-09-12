"""
Dynamic Region of Interest (ROI) Extraction Engine for AuraPulse.
Extracts landmark-anchored polygonal ROIs for Forehead and Malar/Cheekbones,
optimized for vascular density and beard/hair rejection.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

from core.face.detector import FaceLandmarks
from core.skin.segmenter import SkinSegmenter


@dataclass
class ROIData:
    name: str
    polygon: np.ndarray             # Nx2 polygon vertices
    mean_rgb: Tuple[float, float, float]    # Mean R, G, B of skin pixels
    median_rgb: Tuple[float, float, float]  # Median R, G, B
    std_rgb: Tuple[float, float, float]     # Standard deviation
    skin_pixel_pct: float                   # Percentage of valid skin pixels
    num_valid_pixels: int
    is_valid: bool


class ROIExtractor:
    """
    Extracts dynamic polygon ROIs anchored to facial landmarks and computes
    skin-masked color statistics. Optimized for malar prominence and beard avoidance.
    """

    def __init__(self, min_skin_pct: float = 25.0, min_pixels: int = 60):
        self.min_skin_pct = min_skin_pct
        self.min_pixels = min_pixels
        self.skin_segmenter = SkinSegmenter()

    def extract_rois(
        self,
        frame: np.ndarray,
        landmarks: FaceLandmarks
    ) -> Dict[str, ROIData]:
        """
        Extracts high-perfusion ROIs: Expanded Forehead, Upper Left Malar Cheek, Upper Right Malar Cheek.
        """
        h, w = frame.shape[:2]
        bx, by, bw, bh = landmarks.bbox
        rois: Dict[str, ROIData] = {}

        lex, ley = landmarks.left_eye
        rex, rey = landmarks.right_eye
        nose_x, nose_y = landmarks.nose_tip
        eye_dist = max(10.0, float(np.hypot(lex - rex, ley - rey)))

        # 1. FOREHEAD POLYGON (Trapezoid: expanded across frontal bone, avoiding brows and hair)
        fh_top_y = max(0, int(min(ley, rey) - eye_dist * 0.72))
        fh_bot_y = max(0, int(min(ley, rey) - eye_dist * 0.22))
        fh_mid_x = (lex + rex) / 2.0
        fh_half_w = eye_dist * 0.55

        forehead_poly = np.array([
            [int(fh_mid_x - fh_half_w * 1.1), fh_bot_y],
            [int(fh_mid_x - fh_half_w * 0.85), fh_top_y],
            [int(fh_mid_x + fh_half_w * 0.85), fh_top_y],
            [int(fh_mid_x + fh_half_w * 1.1), fh_bot_y],
        ], dtype=np.int32)

        # 2. UPPER LEFT CHEEK (Malar / Zygomatic bone: high on cheekbone, well above beard/mouth)
        # Positioned directly below eye and lateral to nose
        lc_x = lex + eye_dist * 0.08
        lc_y = ley + (nose_y - ley) * 0.45
        lc_w = eye_dist * 0.28
        lc_h = (nose_y - ley) * 0.40

        left_cheek_poly = np.array([
            [int(lc_x - lc_w * 0.8), int(lc_y - lc_h * 0.5)],
            [int(lc_x + lc_w * 0.8), int(lc_y - lc_h * 0.5)],
            [int(lc_x + lc_w * 0.7), int(lc_y + lc_h * 0.5)],
            [int(lc_x - lc_w * 0.7), int(lc_y + lc_h * 0.5)],
        ], dtype=np.int32)

        # 3. UPPER RIGHT CHEEK (Malar / Zygomatic bone)
        rc_x = rex - eye_dist * 0.08
        rc_y = rey + (nose_y - rey) * 0.45
        rc_w = eye_dist * 0.28
        rc_h = (nose_y - rey) * 0.40

        right_cheek_poly = np.array([
            [int(rc_x - rc_w * 0.8), int(rc_y - rc_h * 0.5)],
            [int(rc_x + rc_w * 0.8), int(rc_y - rc_h * 0.5)],
            [int(rc_x + rc_w * 0.7), int(rc_y + rc_h * 0.5)],
            [int(rc_x - rc_w * 0.7), int(rc_y + rc_h * 0.5)],
        ], dtype=np.int32)

        candidate_polys = {
            "forehead": forehead_poly,
            "left_cheek": left_cheek_poly,
            "right_cheek": right_cheek_poly,
        }

        for name, poly in candidate_polys.items():
            roi_data = self._process_single_polygon(frame, poly, name)
            rois[name] = roi_data

        return rois

    def _process_single_polygon(
        self,
        frame: np.ndarray,
        poly: np.ndarray,
        name: str
    ) -> ROIData:
        h, w = frame.shape[:2]

        x, y, pw, ph = cv2.boundingRect(poly)
        x = max(0, min(w - 1, x))
        y = max(0, min(h - 1, y))
        pw = max(1, min(w - x, pw))
        ph = max(1, min(h - y, ph))

        crop = frame[y:y+ph, x:x+pw]
        if crop.size == 0 or crop.shape[0] == 0 or crop.shape[1] == 0:
            return ROIData(
                name=name,
                polygon=poly,
                mean_rgb=(0.0, 0.0, 0.0),
                median_rgb=(0.0, 0.0, 0.0),
                std_rgb=(0.0, 0.0, 0.0),
                skin_pixel_pct=0.0,
                num_valid_pixels=0,
                is_valid=False
            )

        poly_relative = poly - np.array([x, y])
        poly_mask = np.zeros((ph, pw), dtype=np.uint8)
        cv2.fillPoly(poly_mask, [poly_relative], 255)

        skin_mask, _ = self.skin_segmenter.segment(crop)
        combined_mask = cv2.bitwise_and(poly_mask, skin_mask)

        poly_pixels = np.count_nonzero(poly_mask)
        skin_pixels = np.count_nonzero(combined_mask)

        skin_pct = (skin_pixels / max(poly_pixels, 1)) * 100.0
        is_valid = (skin_pct >= self.min_skin_pct) and (skin_pixels >= self.min_pixels)

        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

        if is_valid:
            valid_pixels = crop_rgb[combined_mask > 0]
            mean_r = float(np.mean(valid_pixels[:, 0]))
            mean_g = float(np.mean(valid_pixels[:, 1]))
            mean_b = float(np.mean(valid_pixels[:, 2]))

            med_r = float(np.median(valid_pixels[:, 0]))
            med_g = float(np.median(valid_pixels[:, 1]))
            med_b = float(np.median(valid_pixels[:, 2]))

            std_r = float(np.std(valid_pixels[:, 0]))
            std_g = float(np.std(valid_pixels[:, 1]))
            std_b = float(np.std(valid_pixels[:, 2]))
        else:
            mean_r, mean_g, mean_b = 0.0, 0.0, 0.0
            med_r, med_g, med_b = 0.0, 0.0, 0.0
            std_r, std_g, std_b = 0.0, 0.0, 0.0

        return ROIData(
            name=name,
            polygon=poly,
            mean_rgb=(mean_r, mean_g, mean_b),
            median_rgb=(med_r, med_g, med_b),
            std_rgb=(std_r, std_g, std_b),
            skin_pixel_pct=skin_pct,
            num_valid_pixels=skin_pixels,
            is_valid=is_valid
        )
