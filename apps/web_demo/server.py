"""
Real-Time Web Application & Prototype Server for AuraPulse.
Serves a responsive on-device web UI with live camera streaming,
real-time dynamic ROI overlay, live BVP waveform chart, and vital signs cards.
"""

from __future__ import annotations
import base64
import json
import sys
from pathlib import Path
import time
import cv2
import numpy as np
from starlette.applications import Starlette
from starlette.responses import HTMLResponse, JSONResponse
from starlette.routing import Route

root_dir = Path(__file__).parent.parent.parent.resolve()
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from core.pipeline import AuraPulseEngine

# Initialize single-session engine
engine = AuraPulseEngine(min_measurement_duration_s=6.0, window_duration_s=25.0, target_fs=30.0)
engine.start_session()

html_path = Path(__file__).parent / "index.html"


def convert_to_serializable(obj):
    """Recursively converts NumPy types to native Python JSON-serializable types."""
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [convert_to_serializable(x) for x in obj]
    return obj


async def homepage(request):
    with open(html_path, "r", encoding="utf-8") as f:
        content = f.read()
    return HTMLResponse(content)


async def reset_session(request):
    engine.start_session()
    return JSONResponse({"status": "SESSION_RESET", "timestamp": time.time()})


async def process_frame_api(request):
    try:
        data = await request.json()
        image_b64 = data.get("image", "")
        timestamp_ms = data.get("timestamp_ms", None)

        if not image_b64:
            return JSONResponse({"error": "No image data provided"}, status_code=400)

        # Decode base64 image (data:image/jpeg;base64,...)
        if "," in image_b64:
            image_b64 = image_b64.split(",", 1)[1]

        img_bytes = base64.b64decode(image_b64)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return JSONResponse({"error": "Failed to decode image"}, status_code=400)

        t_s = (float(timestamp_ms) / 1000.0) if timestamp_ms is not None else None
        has_face, landmarks, rois, face_q = engine.process_frame(frame, timestamp_s=t_s)

        vitals_result = engine.compute_vitals()

        # Prepare response payload
        roi_polygons = {}
        if rois:
            for r_name, r_data in rois.items():
                roi_polygons[r_name] = {
                    "polygon": r_data.polygon.tolist() if isinstance(r_data.polygon, np.ndarray) else list(r_data.polygon),
                    "isValid": bool(r_data.is_valid),
                    "skinPct": float(round(r_data.skin_pixel_pct, 1))
                }

        face_bbox = [int(v) for v in landmarks.bbox] if (landmarks and has_face) else None

        response_payload = {
            "hasFace": bool(has_face),
            "faceBBox": face_bbox,
            "faceQuality": {
                "overall": float(round(face_q.overall_score, 1)),
                "isAcceptable": bool(face_q.is_acceptable),
                "rejectionReason": face_q.rejection_reason,
                "illumination": float(round(face_q.illumination_score, 1)),
                "pose": float(round(face_q.pose_score, 1)),
                "motion": float(round(face_q.motion_score, 1))
            },
            "rois": roi_polygons,
            "vitals": convert_to_serializable(vitals_result.to_dict())
        }

        return JSONResponse(convert_to_serializable(response_payload))

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)


routes = [
    Route("/", homepage),
    Route("/api/reset", reset_session, methods=["POST"]),
    Route("/api/process_frame", process_frame_api, methods=["POST"]),
]

app = Starlette(debug=True, routes=routes)

if __name__ == "__main__":
    import uvicorn
    print("[*] Launching AuraPulse Web Demo on http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)
