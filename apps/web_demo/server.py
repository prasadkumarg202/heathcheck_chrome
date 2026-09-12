"""
Real-Time Ultra-High FPS Web Server for AuraPulse.
Features:
1. WebSocket streaming endpoint (/ws/signals) for locked 30+ FPS telemetry with microsecond latency.
2. Keyframe Face Alignment (/api/detect_rois) executed once every ~1.5s.
3. Batched & single HTTP fallback endpoints (/api/push_signals_batch, /api/push_signals).
"""

from __future__ import annotations
import base64
import json
import sys
from pathlib import Path
import time
import asyncio
import cv2
import numpy as np
from starlette.applications import Starlette
from starlette.responses import HTMLResponse, JSONResponse
from starlette.routing import Route, WebSocketRoute
from starlette.websockets import WebSocket, WebSocketDisconnect

root_dir = Path(__file__).parent.parent.parent.resolve()
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from core.pipeline import AuraPulseEngine
from core.face.detector import FaceDetector
from core.roi.extractor import ROIExtractor, ROIData

# Initialize single-session engine
engine = AuraPulseEngine(min_measurement_duration_s=6.0, window_duration_s=25.0, target_fs=30.0)
engine.start_session()

face_detector = FaceDetector(detection_interval=1)
roi_extractor = ROIExtractor()

# User profile state
user_profile = {
    "age": 35.0,
    "is_male": True,
    "height_cm": 175.0,
    "weight_kg": 72.0,
    "waist_cm": 84.0,
    "is_smoker": False,
    "is_diabetic": False,
    "bmi": 23.5,
}

cached_vitals = engine.compute_vitals(user_profile)
last_vitals_compute_time = 0.0

html_path = Path(__file__).parent / "index.html"
thresholds_path = Path(__file__).parent / "thresholds.js"


def convert_to_serializable(obj):
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (float, np.floating, np.float64, np.float32)):
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


async def serve_thresholds(request):
    from starlette.responses import Response
    with open(thresholds_path, "r", encoding="utf-8") as f:
        content = f.read()
    return Response(content, media_type="application/javascript")


async def reset_session(request):
    global cached_vitals, last_vitals_compute_time
    engine.start_session()
    cached_vitals = engine.compute_vitals(user_profile)
    last_vitals_compute_time = 0.0
    return JSONResponse({ "status": "SESSION_RESET", "timestamp": time.time() })


async def update_profile_api(request):
    global user_profile, cached_vitals, last_vitals_compute_time
    try:
        data = await request.json()
        for k in ["age", "is_male", "height_cm", "weight_kg", "waist_cm", "is_smoker", "is_diabetic"]:
            if k in data:
                user_profile[k] = data[k]
        
        # Recalculate BMI
        h_m = max(0.5, float(user_profile["height_cm"]) / 100.0)
        w_kg = max(20.0, float(user_profile["weight_kg"]))
        user_profile["bmi"] = round(w_kg / (h_m * h_m), 1)

        cached_vitals = engine.compute_vitals(user_profile)
        return JSONResponse(convert_to_serializable({
            "status": "PROFILE_UPDATED",
            "profile": user_profile,
            "vitals": cached_vitals.to_dict()
        }))
    except Exception as e:
        return JSONResponse({ "error": str(e) }, status_code=400)


async def detect_rois_keyframe(request):
    """
    Keyframe Endpoint: Runs neural face detection once every 1-2 seconds
    and returns exact relative ROI bounding coordinates to the client.
    """
    try:
        data = await request.json()
        image_b64 = data.get("image", "")

        if not image_b64:
            return JSONResponse({ "error": "No image data" }, status_code=400)

        if "," in image_b64:
            image_b64 = image_b64.split(",", 1)[1]

        img_bytes = base64.b64decode(image_b64)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return JSONResponse({ "error": "Failed to decode frame" }, status_code=400)

        has_face, landmarks, face_q = face_detector.detect(frame)

        if not has_face or landmarks is None:
            return JSONResponse({
                "hasFace": False,
                "faceQuality": {
                    "overall": 0.0,
                    "isAcceptable": False,
                    "rejectionReason": face_q.rejection_reason
                },
                "rois": {}
            })

        rois = roi_extractor.extract_rois(frame, landmarks)
        roi_polygons = {}
        for r_name, r_data in rois.items():
            roi_polygons[r_name] = {
                "polygon": r_data.polygon.tolist(),
                "isValid": bool(r_data.is_valid),
                "skinPct": float(round(r_data.skin_pixel_pct, 1))
            }

        face_bbox = [int(v) for v in landmarks.bbox]

        return JSONResponse(convert_to_serializable({
            "hasFace": True,
            "frameWidth": int(frame.shape[1]),
            "frameHeight": int(frame.shape[0]),
            "faceBBox": face_bbox,
            "faceQuality": {
                "overall": float(round(face_q.overall_score, 1)),
                "isAcceptable": bool(face_q.is_acceptable),
                "illumination": float(round(face_q.illumination_score, 1)),
                "pose": float(round(face_q.pose_score, 1)),
                "motion": float(round(face_q.motion_score, 1))
            },
            "rois": roi_polygons
        }))

    except Exception as e:
        return JSONResponse({ "error": str(e) }, status_code=500)


def _ingest_roi_signals(timestamp_s: float, roi_signals: dict):
    dummy_poly = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    roi_dict = {}
    for r_name in ["forehead", "left_cheek", "right_cheek"]:
        if r_name in roi_signals:
            rgb = tuple(roi_signals[r_name])
            roi_dict[r_name] = ROIData(
                name=r_name,
                polygon=dummy_poly,
                mean_rgb=rgb,
                median_rgb=rgb,
                std_rgb=(1.0, 1.0, 1.0),
                skin_pixel_pct=95.0,
                num_valid_pixels=200,
                is_valid=True
            )
    engine.signal_buffer.append(timestamp_s, roi_dict)


async def push_signals_api(request):
    """
    Ultra-Fast Signal Ingestion Endpoint:
    Receives extracted RGB vectors from client, appends to buffer, and returns live vitals.
    """
    global cached_vitals, last_vitals_compute_time
    try:
        data = await request.json()
        timestamp_s = float(data.get("timestamp_s", time.time()))
        roi_signals = data.get("signals", {})
        meta = data.get("profile", user_profile)

        _ingest_roi_signals(timestamp_s, roi_signals)

        now = time.perf_counter()
        if (now - last_vitals_compute_time) >= 0.25:
            cached_vitals = engine.compute_vitals(meta)
            last_vitals_compute_time = now

        return JSONResponse(convert_to_serializable({
            "status": "OK",
            "vitals": cached_vitals.to_dict()
        }))

    except Exception as e:
        return JSONResponse({ "error": str(e) }, status_code=500)


async def push_signals_batch_api(request):
    """
    Batched Signal Ingestion Endpoint:
    Receives an array of signal frames in one HTTP call.
    """
    global cached_vitals, last_vitals_compute_time
    try:
        data = await request.json()
        frames = data.get("frames", [])
        meta = data.get("profile", user_profile)
        for f in frames:
            ts = float(f.get("timestamp_s", time.time()))
            sigs = f.get("signals", {})
            _ingest_roi_signals(ts, sigs)

        now = time.perf_counter()
        if (now - last_vitals_compute_time) >= 0.20:
            cached_vitals = engine.compute_vitals(meta)
            last_vitals_compute_time = now

        return JSONResponse(convert_to_serializable({
            "status": "OK",
            "vitals": cached_vitals.to_dict()
        }))
    except Exception as e:
        return JSONResponse({ "error": str(e) }, status_code=500)


async def websocket_signals_endpoint(websocket: WebSocket):
    """
    WebSocket Telemetry Stream:
    Microsecond bidirectional vector pipeline running at 30-60 Hz without HTTP latency.
    """
    global cached_vitals, last_vitals_compute_time, user_profile
    await websocket.accept()
    try:
        while True:
            text = await websocket.receive_text()
            data = json.loads(text)
            
            if data.get("type") == "reset":
                engine.start_session()
                cached_vitals = engine.compute_vitals(user_profile)
                last_vitals_compute_time = 0.0
                await websocket.send_text(json.dumps({ "status": "SESSION_RESET" }))
                continue

            if data.get("type") == "update_profile":
                prof = data.get("profile", {})
                for k in ["age", "is_male", "height_cm", "weight_kg", "waist_cm", "is_smoker", "is_diabetic"]:
                    if k in prof:
                        user_profile[k] = prof[k]
                h_m = max(0.5, float(user_profile["height_cm"]) / 100.0)
                w_kg = max(20.0, float(user_profile["weight_kg"]))
                user_profile["bmi"] = round(w_kg / (h_m * h_m), 1)
                cached_vitals = engine.compute_vitals(user_profile)
                await websocket.send_text(json.dumps(convert_to_serializable({
                    "status": "PROFILE_UPDATED",
                    "profile": user_profile,
                    "vitals": cached_vitals.to_dict()
                })))
                continue

            ts = float(data.get("timestamp_s", time.time()))
            sigs = data.get("signals", {})
            _ingest_roi_signals(ts, sigs)

            now = time.perf_counter()
            if (now - last_vitals_compute_time) >= 0.15:  # Update vitals ~6 times per second
                cached_vitals = engine.compute_vitals(user_profile)
                last_vitals_compute_time = now

                payload = convert_to_serializable({
                    "status": "OK",
                    "vitals": cached_vitals.to_dict()
                })
                await websocket.send_text(json.dumps(payload))

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.close()
        except Exception:
            pass


routes = [
    Route("/", homepage),
    Route("/thresholds.js", serve_thresholds),
    Route("/api/reset", reset_session, methods=["POST"]),
    Route("/api/update_profile", update_profile_api, methods=["POST"]),
    Route("/api/detect_rois", detect_rois_keyframe, methods=["POST"]),
    Route("/api/push_signals", push_signals_api, methods=["POST"]),
    Route("/api/push_signals_batch", push_signals_batch_api, methods=["POST"]),
    WebSocketRoute("/ws/signals", websocket_signals_endpoint),
]

app = Starlette(debug=True, routes=routes)

if __name__ == "__main__":
    import uvicorn
    print("[*] Launching 30+ FPS AuraPulse Server on http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)
