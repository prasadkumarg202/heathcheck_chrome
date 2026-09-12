"""
AuraPulse Command Line Testbed & Real-Time Video Runner.
Supports processing from live webcam, pre-recorded video files, or synthetic simulation streams.
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import time

# Ensure project root is in sys.path
root_dir = Path(__file__).parent.parent.resolve()
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import cv2
import numpy as np

from core.pipeline import AuraPulseEngine
from core.vision.camera import VideoInputEngine
from core.roi.extractor import ROIData
from research.synthetic.generator import SyntheticSignalGenerator


def run_camera_stream(source: int | str = 0, duration_s: float = 30.0):
    print(f"[*] Initializing AuraPulse Core Engine...")
    engine = AuraPulseEngine(min_measurement_duration_s=6.0, window_duration_s=25.0)
    engine.start_session()

    video_input = VideoInputEngine(source=source)
    if not video_input.open():
        print(f"[!] Error: Unable to open video source: {source}")
        print(f"[!] Tip: If your web browser is open on http://127.0.0.1:8000, close the browser tab to release the webcam hardware.")
        return

    print(f"[*] Camera opened successfully. Running scan for {duration_s} seconds...")
    print(f"[*] Tip: If another app (like Chrome) is using the webcam, close that tab to avoid camera hardware lock.")
    print(f"[*] Position your face within the frame and keep steady.\n")

    start_time = time.time()
    frame_count = 0

    try:
        while True:
            elapsed = time.time() - start_time
            if elapsed >= duration_s:
                break

            ret, frame, meta = video_input.read_frame()
            if not ret or frame is None or meta is None:
                time.sleep(0.03)
                continue

            frame_count += 1
            has_face, landmarks, rois, face_q = engine.process_frame(frame, timestamp_s=meta.timestamp_s)

            # Terminal feedback every 1 second
            if int(elapsed * 10) % 10 == 0:
                result = engine.compute_vitals()
                status_str = result.status
                if status_str == "VALID":
                    hr = result.heart_rate['value']
                    conf = result.heart_rate['confidence']
                    sqi = result.signal_quality
                    rr = result.respiration_rate['value'] if result.respiration_rate else '--'
                    print(f"[{elapsed:4.1f}s] HR: {hr:5.1f} BPM (Conf: {conf*100:3.0f}%) | RR: {rr} | SQI: {sqi:4.1f}/100 ({result.quality_category})")
                else:
                    face_status = "Face Locked" if has_face else f"Searching Face ({face_q.rejection_reason or 'No Face'})"
                    print(f"[{elapsed:4.1f}s] {face_status} | Duration: {result.measurement_duration_s:.1f}s | {result.reason}")

            # Draw visual debug overlays on frame
            if has_face and landmarks is not None:
                bx, by, bw, bh = landmarks.bbox
                cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
                cv2.putText(frame, f"Face Quality: {face_q.overall_score:.0f}%", (bx, max(20, by - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

                if rois:
                    for r_name, r_data in rois.items():
                        color = (255, 255, 0) if r_data.is_valid else (0, 0, 255)
                        cv2.polylines(frame, [r_data.polygon], isClosed=True, color=color, thickness=2)
                        # Put ROI label
                        centroid_x = int(np.mean(r_data.polygon[:, 0]))
                        centroid_y = int(np.mean(r_data.polygon[:, 1]))
                        cv2.putText(frame, r_name[:4].upper(), (centroid_x - 15, centroid_y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            cv2.imshow("AuraPulse - Contactless Vitals Monitor (Press Q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        video_input.close()
        cv2.destroyAllWindows()

    print("\n" + "=" * 60)
    print("FINAL MEASUREMENT REPORT")
    print("=" * 60)
    final_result = engine.compute_vitals()
    print(json.dumps(final_result.to_dict(), indent=2))


def run_synthetic_benchmark(hr_bpm: float = 75.0, rr_rpm: float = 16.0, duration_s: float = 30.0):
    print(f"[*] Running Synthetic Benchmark: Target HR={hr_bpm} BPM, Target RR={rr_rpm} RPM, Duration={duration_s}s")
    gen = SyntheticSignalGenerator(fs=30.0)
    gt = gen.generate_signal(duration_s=duration_s, hr_bpm=hr_bpm, rr_rpm=rr_rpm, snr_db=18.0)

    engine = AuraPulseEngine(min_measurement_duration_s=6.0, window_duration_s=duration_s)
    engine.start_session()

    dummy_poly = np.array([[10, 10], [50, 10], [50, 50], [10, 50]])

    for idx in range(len(gt.timestamps)):
        t_s = gt.timestamps[idx]
        roi_dict = {}
        for roi_name in ["forehead", "left_cheek", "right_cheek"]:
            rgb = tuple(gt.simulated_rgb[roi_name][idx])
            roi_dict[roi_name] = ROIData(
                name=roi_name,
                polygon=dummy_poly,
                mean_rgb=rgb,
                median_rgb=rgb,
                std_rgb=(1.0, 1.0, 1.0),
                skin_pixel_pct=95.0,
                num_valid_pixels=300,
                is_valid=True
            )
        engine.signal_buffer.append(t_s, roi_dict)

    result = engine.compute_vitals()
    print("\n" + json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AuraPulse Contactless Vitals Engine CLI")
    parser.add_argument("--mode", choices=["camera", "synthetic"], default="camera", help="Mode of operation")
    parser.add_argument("--source", default=0, help="Camera index or video file path")
    parser.add_argument("--hr", type=float, default=74.0, help="Synthetic target HR (BPM)")
    parser.add_argument("--rr", type=float, default=16.0, help="Synthetic target RR (RPM)")
    parser.add_argument("--duration", type=float, default=30.0, help="Measurement duration in seconds")

    args = parser.parse_args()

    if args.mode == "camera":
        src = int(args.source) if str(args.source).isdigit() else args.source
        run_camera_stream(source=src, duration_s=args.duration)
    else:
        run_synthetic_benchmark(hr_bpm=args.hr, rr_rpm=args.rr, duration_s=args.duration)
