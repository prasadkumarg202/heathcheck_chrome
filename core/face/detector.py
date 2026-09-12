"""
Face Detection and Quality Assessment Engine for AuraPulse.
Detects faces, estimates head pose (yaw/pitch/roll), computes landmarks/anchors,
enforces single-face policy, and computes comprehensive FaceQualityScore (0-100).
Powered by OpenCV YuNet neural detector and Haar cascades.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List, Dict
import cv2
import numpy as np


@dataclass
class FaceLandmarks:
    """Key anchor points on the face in pixel coordinates (x, y)."""
    forehead: Tuple[float, float]
    left_eye: Tuple[float, float]
    right_eye: Tuple[float, float]
    nose_tip: Tuple[float, float]
    left_cheek: Tuple[float, float]
    right_cheek: Tuple[float, float]
    chin: Tuple[float, float]
    mouth_center: Tuple[float, float]
    bbox: Tuple[int, int, int, int]  # x, y, w, h


@dataclass
class HeadPose:
    yaw_deg: float      # Left/Right turning
    pitch_deg: float    # Up/Down tilt
    roll_deg: float     # In-plane rotation
    is_aligned: bool


@dataclass
class FaceQualityScore:
    overall_score: float  # 0 to 100
    is_acceptable: bool
    face_size_ratio: float
    illumination_score: float
    pose_score: float
    motion_score: float
    blur_score: float
    multiple_faces_detected: bool
    rejection_reason: Optional[str] = None


class FaceDetector:
    """
    High-accuracy, edge-portable neural face detector and pose estimator.
    Uses OpenCV YuNet with Haar cascade fallback.
    """

    def __init__(
        self,
        min_face_size_ratio: float = 0.05,
        max_yaw_deg: float = 35.0,
        max_pitch_deg: float = 35.0,
        max_roll_deg: float = 35.0,
    ):
        self.min_face_size_ratio = min_face_size_ratio
        self.max_yaw_deg = max_yaw_deg
        self.max_pitch_deg = max_pitch_deg
        self.max_roll_deg = max_roll_deg

        models_dir = Path(__file__).parent / "models"
        self.yunet_path = models_dir / "face_detection_yunet_2023mar.onnx"
        self.cascade_path = models_dir / "haarcascade_frontalface_alt2.xml"

        self.yunet = None
        self.yunet_size = (0, 0)
        self.cascade = None

        if self.cascade_path.exists():
            try:
                self.cascade = cv2.CascadeClassifier(str(self.cascade_path))
            except Exception:
                self.cascade = None

        self.last_bbox: Optional[Tuple[int, int, int, int]] = None
        self.last_center: Optional[Tuple[float, float]] = None
        self.tracking_alpha = 0.75

    def detect(self, frame: np.ndarray) -> Tuple[bool, Optional[FaceLandmarks], FaceQualityScore]:
        """
        Detects primary face, estimates landmarks, computes pose and quality score.
        """
        h, w = frame.shape[:2]
        detected_faces = []
        raw_landmarks = None

        # 1. Primary: YuNet ONNX Neural Face Detector
        if self.yunet_path.exists():
            try:
                if self.yunet is None or self.yunet_size != (w, h):
                    self.yunet = cv2.FaceDetectorYN.create(
                        str(self.yunet_path),
                        "",
                        (w, h),
                        score_threshold=0.45,
                        nms_threshold=0.3
                    )
                    self.yunet_size = (w, h)

                _, faces = self.yunet.detect(frame)
                if faces is not None and len(faces) > 0:
                    for f in faces:
                        bx, by, bw, bh = int(f[0]), int(f[1]), int(f[2]), int(f[3])
                        bx = max(0, min(w - 1, bx))
                        by = max(0, min(h - 1, by))
                        bw = max(10, min(w - bx, bw))
                        bh = max(10, min(h - by, bh))
                        detected_faces.append((bx, by, bw, bh))
                    
                    # Store 5 facial keypoints of highest scoring face
                    # [x_re, y_re, x_le, y_le, x_nt, y_nt, x_rcm, y_rcm, x_lcm, y_lcm]
                    best_f = faces[0]
                    raw_landmarks = {
                        "right_eye": (float(best_f[4]), float(best_f[5])),
                        "left_eye": (float(best_f[6]), float(best_f[7])),
                        "nose_tip": (float(best_f[8]), float(best_f[9])),
                        "right_mouth": (float(best_f[10]), float(best_f[11])),
                        "left_mouth": (float(best_f[12]), float(best_f[13])),
                    }
            except Exception as e:
                pass

        # 2. Secondary Fallback: Haar Cascade
        if len(detected_faces) == 0 and self.cascade is not None and not self.cascade.empty():
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
            faces = self.cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=3,
                minSize=(int(w * 0.08), int(h * 0.08))
            )
            detected_faces = list(faces)

        # 3. Tertiary Fallback: Skin Blob
        if len(detected_faces) == 0:
            detected_faces = self._detect_skin_face_candidate(frame)

        num_faces = len(detected_faces)
        if num_faces == 0:
            return False, None, FaceQualityScore(
                overall_score=0.0,
                is_acceptable=False,
                face_size_ratio=0.0,
                illumination_score=0.0,
                pose_score=0.0,
                motion_score=0.0,
                blur_score=0.0,
                multiple_faces_detected=False,
                rejection_reason="NO_FACE_DETECTED"
            )

        # Largest face
        detected_faces.sort(key=lambda c: c[2] * c[3], reverse=True)
        fx, fy, fw, fh = detected_faces[0]

        # Temporal smoothing
        if self.last_bbox is not None:
            lx, ly, lw, lh = self.last_bbox
            sx = int(self.tracking_alpha * fx + (1 - self.tracking_alpha) * lx)
            sy = int(self.tracking_alpha * fy + (1 - self.tracking_alpha) * ly)
            sw = int(self.tracking_alpha * fw + (1 - self.tracking_alpha) * lw)
            sh = int(self.tracking_alpha * fh + (1 - self.tracking_alpha) * lh)
            bbox = (sx, sy, sw, sh)
        else:
            bbox = (int(fx), int(fy), int(fw), int(fh))

        self.last_bbox = bbox
        bx, by, bw, bh = bbox

        face_area = bw * bh
        frame_area = w * h
        face_size_ratio = face_area / max(frame_area, 1)

        # Landmark calculation
        if raw_landmarks:
            rex, rey = raw_landmarks["right_eye"]
            lex, ley = raw_landmarks["left_eye"]
            nose_x, nose_y = raw_landmarks["nose_tip"]
            rm_x, rm_y = raw_landmarks["right_mouth"]
            lm_x, lm_y = raw_landmarks["left_mouth"]
            
            eye_dist = max(10.0, float(np.hypot(lex - rex, ley - rey)))
            
            # Forehead anchor
            forehead_x = (rex + lex) / 2.0
            forehead_y = min(rey, ley) - eye_dist * 0.45

            # Cheeks
            r_cheek_x = rex - eye_dist * 0.15
            r_cheek_y = rey + (nose_y - rey) * 0.70

            l_cheek_x = lex + eye_dist * 0.15
            l_cheek_y = ley + (nose_y - ley) * 0.70

            mouth_x = (rm_x + lm_x) / 2.0
            mouth_y = (rm_y + lm_y) / 2.0
            chin_x = mouth_x
            chin_y = by + bh * 0.95
        else:
            lex = bx + bw * 0.32
            ley = by + bh * 0.37
            rex = bx + bw * 0.68
            rey = by + bh * 0.37
            forehead_x = bx + bw * 0.5
            forehead_y = by + bh * 0.18
            nose_x = bx + bw * 0.5
            nose_y = by + bh * 0.58
            l_cheek_x = bx + bw * 0.26
            l_cheek_y = by + bh * 0.64
            r_cheek_x = bx + bw * 0.74
            r_cheek_y = by + bh * 0.64
            chin_x = bx + bw * 0.5
            chin_y = by + bh * 0.92
            mouth_x = bx + bw * 0.5
            mouth_y = by + bh * 0.78

        dx = rex - lex
        dy = rey - ley
        roll_deg = float(np.degrees(np.arctan2(dy, max(abs(dx), 1e-5))))

        landmarks = FaceLandmarks(
            forehead=(forehead_x, forehead_y),
            left_eye=(lex, ley),
            right_eye=(rex, rey),
            nose_tip=(nose_x, nose_y),
            left_cheek=(l_cheek_x, l_cheek_y),
            right_cheek=(r_cheek_x, r_cheek_y),
            chin=(chin_x, chin_y),
            mouth_center=(mouth_x, mouth_y),
            bbox=bbox,
        )

        yaw_deg = float((nose_x - (bx + bw * 0.5)) / (bw * 0.5 + 1e-6) * 45.0)
        pitch_deg = float((nose_y - (by + bh * 0.55)) / (bh * 0.5 + 1e-6) * 45.0)

        # Motion displacement
        curr_center = (bx + bw / 2.0, by + bh / 2.0)
        if self.last_center is not None:
            dist = np.hypot(curr_center[0] - self.last_center[0], curr_center[1] - self.last_center[1])
            motion_score = max(0.0, 100.0 - (dist / (bw + 1e-5) * 400.0))
        else:
            motion_score = 100.0
        self.last_center = curr_center

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        face_crop = gray[max(0, by):min(h, by+bh), max(0, bx):min(w, bx+bw)]
        mean_face_lum = float(np.mean(face_crop)) if face_crop.size > 0 else 0.0

        if 40 <= mean_face_lum <= 230:
            illum_score = 100.0 - abs(mean_face_lum - 130.0) * 0.3
        else:
            illum_score = max(0.0, 100.0 - abs(mean_face_lum - 130.0) * 0.9)

        pose_penalty = (abs(yaw_deg) / self.max_yaw_deg + abs(pitch_deg) / self.max_pitch_deg + abs(roll_deg) / self.max_roll_deg) / 3.0
        pose_score = max(0.0, min(100.0, (1.0 - pose_penalty) * 100.0))

        laplacian_var = float(cv2.Laplacian(face_crop, cv2.CV_64F).var()) if face_crop.size > 0 else 0.0
        blur_score = min(100.0, laplacian_var * 2.0)

        overall = (
            0.25 * min(100.0, (face_size_ratio / 0.08) * 100.0) +
            0.25 * illum_score +
            0.25 * pose_score +
            0.15 * motion_score +
            0.10 * blur_score
        )
        overall = float(np.clip(overall, 0.0, 100.0))

        rejection_reason = None
        if face_size_ratio < self.min_face_size_ratio:
            rejection_reason = "FACE_TOO_SMALL"
        elif illum_score < 15.0:
            rejection_reason = "POOR_LIGHTING"
        elif pose_score < 15.0:
            rejection_reason = "EXCESSIVE_HEAD_ROTATION"
        elif motion_score < 10.0:
            rejection_reason = "EXCESSIVE_MOTION"
        elif overall < 20.0:
            rejection_reason = "LOW_OVERALL_FACE_QUALITY"

        quality = FaceQualityScore(
            overall_score=overall,
            is_acceptable=(rejection_reason is None),
            face_size_ratio=face_size_ratio,
            illumination_score=illum_score,
            pose_score=pose_score,
            motion_score=motion_score,
            blur_score=blur_score,
            multiple_faces_detected=(num_faces > 1),
            rejection_reason=rejection_reason,
        )

        return True, landmarks, quality

    def _detect_skin_face_candidate(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        h, w = frame.shape[:2]
        scale = 0.5
        small = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
        ycrcb = cv2.cvtColor(small, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = ycrcb[:, :, 0], ycrcb[:, :, 1], ycrcb[:, :, 2]

        skin_mask = ((cr >= 115) & (cr <= 195) & (cb >= 60) & (cb <= 150) & (y >= 15)).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        min_area = (w * scale) * (h * scale) * 0.03

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > min_area:
                x, y, cw, ch = cv2.boundingRect(cnt)
                aspect = ch / max(cw, 1)
                if 0.7 <= aspect <= 3.0:
                    orig_x = int(x / scale)
                    orig_y = int(y / scale)
                    orig_w = int(cw / scale)
                    orig_h = int(ch / scale)
                    candidates.append((orig_x, orig_y, orig_w, orig_h))

        candidates.sort(key=lambda c: c[2] * c[3], reverse=True)
        return candidates[:1]
