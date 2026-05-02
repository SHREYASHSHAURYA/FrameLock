import os
import cv2
import json
import time
import threading
import numpy as np
from flask import Flask, Response, jsonify, request
from flask_cors import CORS
from waitress import serve

import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from feature_detection import convert_to_grayscale, detect_features
from optical_flow import track_features
from motion_estimation import estimate_motion
from smoothing import Trajectory
from evaluation import MotionEvaluator
from roi import get_roi
from transformations import get_translation, get_rotation, get_scaling, apply_affine
import roi as roi_module

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "..", "data", "input")
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "data", "output")

# ── shared state ──────────────────────────────────────────────────────────────
processing_state = {
    "running": False,
    "source": None,
    "mode": "final",
    "events": [],
    "lock": threading.Lock(),
    "stop_flag": False,
    "thread": None,
    "frame_raw": None,  # latest JPEG bytes — raw frame with ROI rect
    "frame_stabilized": None,  # latest JPEG bytes — output frame with ROI rect
    "frame_lock": threading.Lock(),
}


# ── SSE event push ────────────────────────────────────────────────────────────
def push_event(data: dict):
    with processing_state["lock"]:
        processing_state["events"].append(json.dumps(data))
        if len(processing_state["events"]) > 500:
            processing_state["events"] = processing_state["events"][-200:]


# ── frame store (called every frame, JPEG-encodes both) ───────────────────────
def store_frames(raw_frame, stabilized_frame):
    _, raw_jpg = cv2.imencode(".jpg", raw_frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
    _, stab_jpg = cv2.imencode(".jpg", stabilized_frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
    with processing_state["frame_lock"]:
        processing_state["frame_raw"] = raw_jpg.tobytes()
        processing_state["frame_stabilized"] = stab_jpg.tobytes()


def _mjpeg_generator(key):
    last_jpg = None
    while True:
        with processing_state["frame_lock"]:
            jpg = processing_state.get(key)
        if jpg is not None and jpg is not last_jpg:
            last_jpg = jpg
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
        else:
            time.sleep(0.001)


@app.route("/video_feed/raw")
def video_feed_raw():
    return Response(
        _mjpeg_generator("frame_raw"),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/video_feed/stabilized")
def video_feed_stabilized():
    return Response(
        _mjpeg_generator("frame_stabilized"),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── SSE stream ────────────────────────────────────────────────────────────────
@app.route("/stream")
def stream():
    def generate():
        sent = 0
        while True:
            with processing_state["lock"]:
                batch = processing_state["events"][sent:]
                sent = len(processing_state["events"])
            for ev in batch:
                yield f"data: {ev}\n\n"
            if not batch:
                time.sleep(0.03)

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── list videos ───────────────────────────────────────────────────────────────
@app.route("/videos")
def list_videos():
    os.makedirs(INPUT_DIR, exist_ok=True)
    files = []
    for f in sorted(os.listdir(INPUT_DIR)):
        if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
            path = os.path.join(INPUT_DIR, f)
            cap = cv2.VideoCapture(path)
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            fps_v = cap.get(cv2.CAP_PROP_FPS) or 30
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            cap.release()
            size_mb = round(os.path.getsize(path) / 1024 / 1024, 1)
            dur = round(frames / fps_v, 1) if fps_v else 0
            files.append(
                {
                    "name": f,
                    "frames": frames,
                    "fps": round(fps_v, 2),
                    "width": w,
                    "height": h,
                    "duration": dur,
                    "size_mb": size_mb,
                }
            )
    return jsonify(files)


@app.route("/status")
def status():
    return jsonify(
        {
            "running": processing_state["running"],
            "source": processing_state["source"],
            "mode": processing_state["mode"],
        }
    )


@app.route("/start", methods=["POST"])
def start():
    data = request.json or {}
    source = data.get("source")
    mode = data.get("mode", "final")
    if processing_state["running"]:
        return jsonify({"error": "already running"}), 400
    if not source:
        return jsonify({"error": "source required"}), 400
    processing_state.update(
        {"stop_flag": False, "source": source, "mode": mode, "running": True}
    )
    with processing_state["frame_lock"]:
        processing_state["frame_raw"] = processing_state["frame_stabilized"] = None
    with processing_state["lock"]:
        processing_state["events"] = []
    t = threading.Thread(target=run_pipeline, args=(source, mode), daemon=True)
    processing_state["thread"] = t
    t.start()
    return jsonify({"ok": True})


@app.route("/stop", methods=["POST"])
def stop():
    processing_state["stop_flag"] = True
    return jsonify({"ok": True})


@app.route("/mode", methods=["POST"])
def set_mode():
    processing_state["mode"] = (request.json or {}).get("mode", "final")
    return jsonify({"ok": True})


# ── pipeline ──────────────────────────────────────────────────────────────────
def run_pipeline(source, initial_mode):
    try:
        _pipeline(source, initial_mode)
    except Exception as exc:
        push_event({"type": "error", "message": str(exc)})
    finally:
        processing_state["running"] = False
        push_event({"type": "done"})


def _pipeline(source, initial_mode):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if source == "camera":
        cap = cv2.VideoCapture(0)
        total_frames, video_fps, is_camera = 0, 30.0, True
        processing_state["video_fps"] = 30.0
    else:
        path = os.path.join(INPUT_DIR, source)
        if not os.path.exists(path):
            push_event({"type": "error", "message": f"File not found: {source}"})
            return
        cap = cv2.VideoCapture(path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        processing_state["video_fps"] = video_fps
        is_camera = False

    if not cap.isOpened():
        push_event({"type": "error", "message": "Cannot open video source"})
        return

    ret, first_frame = cap.read()
    if not ret:
        cap.release()
        push_event({"type": "error", "message": "Cannot read first frame"})
        return

    fh, fw = first_frame.shape[:2]

    out = None
    if not is_camera:
        name = os.path.splitext(source)[0]
        out_path = os.path.join(OUTPUT_DIR, f"{name}_stabilized.mp4")
        out = cv2.VideoWriter(
            out_path, cv2.VideoWriter_fourcc(*"mp4v"), video_fps, (fw * 2, fh)
        )

    roi_module._prev_cx = roi_module._prev_cy = None
    roi_module._frame_count = 0

    trajectory = Trajectory()
    evaluator = MotionEvaluator()
    roi_evaluator = MotionEvaluator()

    prev_gray = convert_to_grayscale(first_frame)
    x1, y1, x2, y2 = get_roi(first_frame)
    roi_pts = detect_features(prev_gray[y1:y2, x1:x2])
    if roi_pts is not None:
        roi_pts[:, :, 0] += x1
        roi_pts[:, :, 1] += y1
    prev_points = roi_pts

    frame_count = 0
    start_time = time.time()

    push_event(
        {
            "type": "start",
            "total": total_frames,
            "fps": video_fps,
            "width": fw,
            "height": fh,
            "source": source,
        }
    )

    # SSE throttle: emit metrics every N frames only
    METRICS_EVERY = 5

    while True:
        if processing_state["stop_flag"]:
            break

        ret, frame = cap.read()
        if not ret:
            break

        curr_gray = convert_to_grayscale(frame)

        if prev_points is None or len(prev_points) < 50:
            prev_points = detect_features(prev_gray)

        dx = dy = da = diff_x = diff_y = diff_a = roi_dx = roi_dy = 0.0
        tracked = 0

        if prev_points is not None and len(prev_points) >= 10:
            prev_pts, curr_pts = track_features(prev_gray, curr_gray, prev_points)
            tracked = len(curr_pts)

            roi_prev, roi_curr = [], []
            for p, c in zip(prev_pts, curr_pts):
                px, py = p.ravel()
                if x1 <= px <= x2 and y1 <= py <= y2:
                    roi_prev.append(p)
                    roi_curr.append(c)

            if len(roi_prev) >= 10:
                roi_dx, roi_dy, _ = estimate_motion(
                    np.array(roi_prev), np.array(roi_curr)
                )

            if len(prev_pts) >= 10:
                dx, dy, da = estimate_motion(prev_pts, curr_pts)
                evaluator.add_raw(dx, dy)
                roi_evaluator.add_raw(roi_dx, roi_dy)

                trajectory.update(dx, dy, da)
                smoothed = trajectory.smooth_kalman()
                sx, sy, sa = smoothed[-1]

                if len(trajectory.trajectory) > 1:
                    prev_sx, prev_sy, prev_sa = smoothed[-2]
                    diff_x = (sx - prev_sx) - dx
                    diff_y = (sy - prev_sy) - dy
                    diff_a = (sa - prev_sa) - da

                evaluator.add_stabilized(diff_x, diff_y)
                roi_evaluator.add_stabilized(diff_x, diff_y)

            prev_points = curr_pts.reshape(-1, 1, 2) if len(curr_pts) > 0 else None
        else:
            prev_points = detect_features(prev_gray)

        raw_disp, stab_disp = evaluator.compute_score()
        roi_raw, roi_stab = roi_evaluator.compute_score()
        improvement = ((roi_raw - roi_stab) / roi_raw * 100) if roi_raw > 0 else 0.0
        fps_current = frame_count / max(time.time() - start_time, 0.001)

        # ── build stabilized frame ────────────────────────────────────────────
        combined_m = get_rotation(diff_a).copy()
        combined_m[:, 2] += get_translation(diff_x, diff_y)[:, 2]
        stabilized = apply_affine(frame, combined_m)
        border = int(min(fh, fw) * 0.05)
        stabilized = stabilized[border : fh - border, border : fw - border]
        stabilized = cv2.resize(stabilized, (fw, fh))

        # ── apply demo mode ───────────────────────────────────────────────────
        mode = processing_state["mode"]
        cx, cy = fw / 2, fh / 2

        def center_m(M):
            T1 = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]], np.float32)
            T2 = np.array([[1, 0, cx], [0, 1, cy], [0, 0, 1]], np.float32)
            return (T2 @ np.vstack([M, [0, 0, 1]]) @ T1)[:2, :]

        if mode == "translation":
            output = apply_affine(frame, get_translation(60, 0))
        elif mode == "rotation":
            output = apply_affine(frame, center_m(get_rotation(0.3)))
        elif mode == "scaling":
            output = apply_affine(frame, center_m(get_scaling(1.3)))
        elif mode == "affine":
            A = get_rotation(0.3)
            A[:, 2] += [40, 30]
            output = apply_affine(frame, center_m(A))
        elif mode == "perspective":
            src = np.float32([[0, 0], [fw, 0], [0, fh], [fw, fh]])
            dst = np.float32([[60, 40], [fw - 60, 0], [0, fh - 60], [fw, fh]])
            output = cv2.warpPerspective(
                frame, cv2.getPerspectiveTransform(src, dst), (fw, fh)
            )
        elif mode == "reflection":
            output = cv2.flip(frame, 1)
        else:
            zoom = 1.15
            zh, zw = int(fh / zoom), int(fw / zoom)
            zy, zx = (fh - zh) // 2, (fw - zw) // 2
            output = cv2.resize(stabilized[zy : zy + zh, zx : zx + zw], (fw, fh))

        # ── draw ROI rectangle on stabilized output only (not raw) ──────────
        raw_display = frame.copy()
        out_display = output.copy()
        roi_color = (0, 229, 160)  # #00e5a0 in BGR
        cv2.rectangle(out_display, (int(x1), int(y1)), (int(x2), int(y2)), roi_color, 2)

        # ── store for MJPEG ───────────────────────────────────────────────────
        store_frames(raw_display, out_display)

        # ── write output file ─────────────────────────────────────────────────
        if out is not None:
            out.write(np.hstack((frame, output)))

        # ── ROI + gray update ─────────────────────────────────────────────────
        x1, y1, x2, y2 = get_roi(frame)
        prev_gray = curr_gray.copy()
        frame_count += 1

        # ── throttled SSE metrics push ────────────────────────────────────────
        if frame_count % METRICS_EVERY == 0:
            push_event(
                {
                    "type": "metrics",
                    "frame": frame_count,
                    "total": total_frames,
                    "dx": round(float(dx), 3),
                    "dy": round(float(dy), 3),
                    "raw_disp": round(float(raw_disp), 4),
                    "stab_disp": round(float(stab_disp), 4),
                    "roi_raw": round(float(roi_raw), 4),
                    "roi_stab": round(float(roi_stab), 4),
                    "improvement": round(float(improvement), 2),
                    "fps": round(float(fps_current), 1),
                    "features": int(tracked),
                    "mode": mode,
                }
            )

    # ── summary ───────────────────────────────────────────────────────────────
    raw_disp, stab_disp = evaluator.compute_score()
    roi_raw, roi_stab = roi_evaluator.compute_score()
    detailed = evaluator.compute_detailed_stats() or {}
    roi_det = roi_evaluator.compute_detailed_stats() or {}

    cap.release()
    if out:
        out.release()

    push_event(
        {
            "type": "summary",
            "source": source,
            "frames": frame_count,
            "raw_disp": round(float(raw_disp), 4),
            "stab_disp": round(float(stab_disp), 4),
            "roi_raw": round(float(roi_raw), 4),
            "roi_stab": round(float(roi_stab), 4),
            "improvement": round(
                (roi_raw - roi_stab) / roi_raw * 100 if roi_raw > 0 else 0, 2
            ),
            "detailed": {
                k: round(float(v), 4) if isinstance(v, (int, float, np.floating)) else v
                for k, v in detailed.items()
            },
            "roi_detailed": {
                k: round(float(v), 4) if isinstance(v, (int, float, np.floating)) else v
                for k, v in roi_det.items()
            },
            "output_file": (
                os.path.join(
                    OUTPUT_DIR, f"{os.path.splitext(source)[0]}_stabilized.mp4"
                )
                if not is_camera
                else None
            ),
        }
    )


if __name__ == "__main__":
    print("Server running on http://0.0.0.0:5000")
    serve(app, host="0.0.0.0", port=5000, threads=8)
