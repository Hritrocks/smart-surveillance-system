from flask import Flask, jsonify, Response
from flask_cors import CORS
from ultralytics import YOLO
import cv2
import threading
import time
from datetime import datetime

app = Flask(__name__)
CORS(app)

# -----------------------
# Config
# -----------------------
DANGEROUS_OBJECTS = {"knife", "scissors", "gun", "cell phone"}
CONF_THRES = 0.40            # lower = more detections
TARGET_FPS = 12              # limit FPS to reduce CPU load
FRAME_SLEEP = 1.0 / TARGET_FPS

# -----------------------
# Model
# -----------------------
model = YOLO("yolov8n.pt")
# Optional speed tweaks (safe defaults):
# - smaller image size -> faster, less accurate
INFER_IMGSZ = 640  # try 480 if slow

# -----------------------
# Shared state
# -----------------------
state_lock = threading.Lock()
latest_jpeg = None            # latest annotated frame as JPEG bytes
detections = []               # list of dicts
alerts = []                   # list of dicts

# Single camera instance
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

def camera_loop():
    """Single loop:
    - read frame
    - run YOLO once
    - update latest_jpeg + detections/alerts
    """
    global latest_jpeg, detections, alerts

    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.2)
            continue

        # YOLO inference ONCE per frame
        results = model.predict(frame, imgsz=INFER_IMGSZ, verbose=False)
        r0 = results[0]

        # Annotated frame for video stream
        annotated = r0.plot()

        # Encode to JPEG once
        ok2, buffer = cv2.imencode(".jpg", annotated)
        if not ok2:
            time.sleep(FRAME_SLEEP)
            continue
        jpeg_bytes = buffer.tobytes()

        # Build detection list for this frame
        ts = datetime.now().strftime("%H:%M:%S")
        frame_dets = []

        # r0.boxes may be None/empty
        if r0.boxes is not None and len(r0.boxes) > 0:
            for b in r0.boxes:
                cls_id = int(b.cls[0])
                conf = float(b.conf[0])
                if conf < CONF_THRES:
                    continue
                name = model.names.get(cls_id, str(cls_id))

                item = {
                    "object": name,
                    "confidence": round(conf, 2),
                    "time": ts
                }
                frame_dets.append(item)

        # Update shared state (atomic)
        with state_lock:
            latest_jpeg = jpeg_bytes

            # Keep last 200 detections max (avoid memory growth)
            detections.extend(frame_dets)
            if len(detections) > 200:
                detections = detections[-200:]

            # Add alerts subset
            frame_alerts = [d for d in frame_dets if d["object"] in DANGEROUS_OBJECTS]
            alerts.extend(frame_alerts)
            if len(alerts) > 200:
                alerts = alerts[-200:]

        time.sleep(FRAME_SLEEP)

# Start the single producer thread
threading.Thread(target=camera_loop, daemon=True).start()

# -----------------------
# API Routes
# -----------------------
@app.route("/detections")
def get_detections():
    with state_lock:
        return jsonify(detections[-20:])

@app.route("/alerts")
def get_alerts():
    with state_lock:
        return jsonify(alerts[-20:])

@app.route("/video_feed")
def video_feed():
    def gen():
        while True:
            with state_lock:
                frame = latest_jpeg
            if frame is None:
                time.sleep(0.05)
                continue

            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
            time.sleep(0.01)

    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    # IMPORTANT: keep reloader OFF to avoid double-start
    app.run(debug=False, use_reloader=False)