"""
Dynamic Region of Interest (ROI) Extraction Engine for AuraPulse.
Extracts landmark-anchored polygonal ROIs for Forehead and Malar/Cheekbones,
with zero-phase homomorphic lighting normalization and beard/hair rejection.
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
    skin-masked color statistics with homomorphic lighting correction.
    """

    def __init__(self, min_skin_pct: float = 25.0, min_pixels: int = 60, enable_homomorphic: bool = True):
        self.min_skin_pct = min_skin_pct
        self.min_pixels = min_pixels
        self.skin_segmenter = SkinSegmenter(enable_homomorphic_norm=enable_homomorphic)
        self.enable_homomorphic = enable_homomorphic

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

        eye_left_x = min(lex, rex)
        eye_right_x = max(lex, rex)
        eye_y = (ley + rey) / 2.0
        eye_dist = max(10.0, float(np.hypot(lex - rex, ley - rey)))

        # 1. FOREHEAD POLYGON (Upper Face above eyebrows)
        fh_mid_x = (eye_left_x + eye_right_x) / 2.0
        fh_bot_y = int(eye_y - eye_dist * 0.22)
        fh_top_y = max(int(by + bh * 0.04), int(eye_y - eye_dist * 0.65))
        if fh_top_y >= fh_bot_y:
            fh_top_y = max(0, fh_bot_y - int(eye_dist * 0.35))
        fh_half_w = eye_dist * 0.44

        forehead_poly = np.array([
            [int(fh_mid_x - fh_half_w * 1.05), fh_bot_y],
            [int(fh_mid_x - fh_half_w * 0.80), fh_top_y],
            [int(fh_mid_x + fh_half_w * 0.80), fh_top_y],
            [int(fh_mid_x + fh_half_w * 1.05), fh_bot_y],
        ], dtype=np.int32)

        # 2. MALAR CHEEKBONES (Upper cheeks flanking nose bridge, above nose base)
        cheek_y = eye_y + (nose_y - eye_y) * 0.28
        cheek_w = eye_dist * 0.28
        cheek_h = max(6.0, (nose_y - eye_y) * 0.35)

        left_cheek_poly = np.array([
            [int(eye_left_x - eye_dist * 0.02 - cheek_w * 0.5), int(cheek_y - cheek_h * 0.5)],
            [int(eye_left_x - eye_dist * 0.02 + cheek_w * 0.5), int(cheek_y - cheek_h * 0.5)],
            [int(eye_left_x - eye_dist * 0.02 + cheek_w * 0.45), int(cheek_y + cheek_h * 0.5)],
            [int(eye_left_x - eye_dist * 0.02 - cheek_w * 0.45), int(cheek_y + cheek_h * 0.5)],
        ], dtype=np.int32)

        right_cheek_poly = np.array([
            [int(eye_right_x + eye_dist * 0.02 - cheek_w * 0.5), int(cheek_y - cheek_h * 0.5)],
            [int(eye_right_x + eye_dist * 0.02 + cheek_w * 0.5), int(cheek_y - cheek_h * 0.5)],
            [int(eye_right_x + eye_dist * 0.02 + cheek_w * 0.45), int(cheek_y + cheek_h * 0.5)],
            [int(eye_right_x + eye_dist * 0.02 - cheek_w * 0.45), int(cheek_y + cheek_h * 0.5)],
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

        # Equalize illumination if enabled
        if self.enable_homomorphic and crop.shape[0] >= 8 and crop.shape[1] >= 8:
            crop_for_color = self.skin_segmenter.homomorphic_equalizer(crop, sigma=15.0)
        else:
            crop_for_color = crop

        crop_rgb = cv2.cvtColor(crop_for_color, cv2.COLOR_BGR2RGB)

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
