"""
Dynamic Region of Interest (ROI) Extraction Engine for AuraPulse.
Extracts landmark-anchored polygonal ROIs for Forehead, Left Cheek, and Right Cheek,
applies skin segmentation masks, and computes robust spatial statistics (mean, median, std).
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
    skin-masked color statistics.
    """

    def __init__(self, min_skin_pct: float = 30.0, min_pixels: int = 100):
        self.min_skin_pct = min_skin_pct
        self.min_pixels = min_pixels
        self.skin_segmenter = SkinSegmenter()

    def extract_rois(
        self,
        frame: np.ndarray,
        landmarks: FaceLandmarks
    ) -> Dict[str, ROIData]:
        """
        Extracts Forehead, Left Cheek, and Right Cheek dynamic polygon ROIs.
        """
        h, w = frame.shape[:2]
        bx, by, bw, bh = landmarks.bbox
        rois: Dict[str, ROIData] = {}

        # 1. FOREHEAD POLYGON (Trapezoid above eyebrows, below hairline)
        # Scaled relative to eye positions and face bounding box
        lex, ley = landmarks.left_eye
        rex, rey = landmarks.right_eye
        eye_dist = abs(rex - lex)

        fh_top_y = max(0, int(by + bh * 0.04))
        fh_bot_y = max(0, int(min(ley, rey) - bh * 0.12))
        fh_left_x = max(0, int(bx + bw * 0.25))
        fh_right_x = min(w, int(bx + bw * 0.75))

        forehead_poly = np.array([
            [fh_left_x, fh_bot_y],
            [int(bx + bw * 0.28), fh_top_y],
            [int(bx + bw * 0.72), fh_top_y],
            [fh_right_x, fh_bot_y],
        ], dtype=np.int32)

        # 2. LEFT CHEEK POLYGON (Below left eye, left of nose, avoiding lips and beard)
        lc_x, lc_y = landmarks.left_cheek
        lc_half_w = int(eye_dist * 0.28)
        lc_half_h = int(bh * 0.10)

        left_cheek_poly = np.array([
            [int(lc_x - lc_half_w), int(lc_y - lc_half_h)],
            [int(lc_x + lc_half_w), int(lc_y - lc_half_h)],
            [int(lc_x + lc_half_w * 0.9), int(lc_y + lc_half_h)],
            [int(lc_x - lc_half_w * 0.9), int(lc_y + lc_half_h)],
        ], dtype=np.int32)

        # 3. RIGHT CHEEK POLYGON (Below right eye, right of nose)
        rc_x, rc_y = landmarks.right_cheek
        rc_half_w = int(eye_dist * 0.28)
        rc_half_h = int(bh * 0.10)

        right_cheek_poly = np.array([
            [int(rc_x - rc_half_w), int(rc_y - rc_half_h)],
            [int(rc_x + rc_half_w), int(rc_y - rc_half_h)],
            [int(rc_x + rc_half_w * 0.9), int(rc_y + rc_half_h)],
            [int(rc_x - rc_half_w * 0.9), int(rc_y + rc_half_h)],
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

        # Bounding box of the polygon
        x, y, pw, ph = cv2.boundingRect(poly)
        x = max(0, min(w - 1, x))
        y = max(0, min(h - 1, y))
        pw = max(1, min(w - x, pw))
        ph = max(1, min(h - y, ph))

        # Crop sub-image
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

        # Polygon mask relative to the crop
        poly_relative = poly - np.array([x, y])
        poly_mask = np.zeros((ph, pw), dtype=np.uint8)
        cv2.fillPoly(poly_mask, [poly_relative], 255)

        # Skin segmentation mask on the crop
        skin_mask, _ = self.skin_segmenter.segment(crop)

        # Combined mask: polygon boundary AND valid skin pixels
        combined_mask = cv2.bitwise_and(poly_mask, skin_mask)

        poly_pixels = np.count_nonzero(poly_mask)
        skin_pixels = np.count_nonzero(combined_mask)

        skin_pct = (skin_pixels / max(poly_pixels, 1)) * 100.0
        is_valid = (skin_pct >= self.min_skin_pct) and (skin_pixels >= self.min_pixels)

        # Convert crop from BGR to RGB
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
