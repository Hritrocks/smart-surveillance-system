from flask import Flask, jsonify, Response
from flask_cors import CORS
from ultralytics import YOLO
import cv2
import threading
import time
from datetime import datetime

app = Flask(__name__, static_folder='frontend', static_url_path='')
CORS(app)

@app.route('/')
def index():
    from flask import send_from_directory
    return send_from_directory('frontend', 'index.html')

# -----------------------
# Config
# -----------------------
DANGEROUS_OBJECTS = {"knife", "scissors", "gun", "cell phone"}
CONF_THRES = 0.40
TARGET_FPS = 12
FRAME_SLEEP = 1.0 / TARGET_FPS

# -----------------------
# Threat Score Config
# -----------------------
# Base threat scores per object
OBJECT_THREAT_SCORES = {
    "gun":        90,
    "knife":      75,
    "scissors":   50,
    "cell phone": 30,
}

# How long (seconds) an object must be visible to increase score
PERSISTENCE_WINDOW = 5  # seconds

# -----------------------
# Model
# -----------------------
model = YOLO("yolov8n.pt")
INFER_IMGSZ = 640

# -----------------------
# Video Source
# -----------------------
# Option A: Webcam (local only)
# cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Option B: Pre-recorded video (works on cloud/Render)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# Auto-loop video file when it ends
def get_frame(cap):
    ok, frame = cap.read()
    if not ok:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, frame = cap.read()
    return ok, frame

# -----------------------
# Shared state
# -----------------------
state_lock = threading.Lock()
latest_jpeg = None
detections = []
alerts = []
threat_state = {
    "score": 0,
    "level": "LOW",
    "reason": "No threats detected",
    "objects": [],
    "person_count": 0,
}

# Track first-seen time of dangerous objects for persistence bonus
object_first_seen = {}  # { object_name: first_seen_timestamp }


def calculate_threat_score(frame_dets):
    """
    Calculate a threat score (0-100) based on:
    1. What dangerous objects are detected
    2. How many persons are nearby
    3. How long the dangerous object has been visible (persistence)
    4. Proximity: how close the object bounding box is to a person bounding box
    """
    now = time.time()
    score = 0
    detected_dangers = []
    person_count = 0
    reasons = []

    for det in frame_dets:
        obj = det["object"]
        conf = det["confidence"]

        if obj == "person":
            person_count += 1

        if obj in OBJECT_THREAT_SCORES:
            base = OBJECT_THREAT_SCORES[obj]

            # Persistence bonus: object visible for >PERSISTENCE_WINDOW secs
            if obj not in object_first_seen:
                object_first_seen[obj] = now
            visible_for = now - object_first_seen[obj]
            persistence_bonus = min(15, int(visible_for / PERSISTENCE_WINDOW) * 5)

            obj_score = base + persistence_bonus
            score += obj_score
            detected_dangers.append(obj)
            reasons.append(f"{obj} detected ({visible_for:.0f}s visible)")

    # Clean up objects no longer seen
    current_objects = {d["object"] for d in frame_dets}
    for obj in list(object_first_seen.keys()):
        if obj not in current_objects:
            del object_first_seen[obj]

    # Person proximity bonus
    if person_count > 0 and detected_dangers:
        proximity_bonus = min(20, person_count * 8)
        score += proximity_bonus
        reasons.append(f"{person_count} person(s) nearby (+{proximity_bonus})")

    # Cap at 100
    score = min(100, score)

    # Determine level
    if score >= 70:
        level = "CRITICAL"
    elif score >= 40:
        level = "WARNING"
    elif score > 0:
        level = "ELEVATED"
    else:
        level = "LOW"

    reason = " | ".join(reasons) if reasons else "No threats detected"

    return {
        "score": score,
        "level": level,
        "reason": reason,
        "objects": detected_dangers,
        "person_count": person_count,
    }


def camera_loop():
    global latest_jpeg, detections, alerts, threat_state

    while True:
        ok, frame = get_frame(cap)
        if not ok:
            time.sleep(0.2)
            continue

        results = model.predict(frame, imgsz=INFER_IMGSZ, verbose=False)
        r0 = results[0]

        annotated = r0.plot()

        ok2, buffer = cv2.imencode(".jpg", annotated)
        if not ok2:
            time.sleep(FRAME_SLEEP)
            continue
        jpeg_bytes = buffer.tobytes()

        ts = datetime.now().strftime("%H:%M:%S")
        frame_dets = []

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

        # Calculate threat score for this frame
        new_threat = calculate_threat_score(frame_dets)

        with state_lock:
            latest_jpeg = jpeg_bytes

            detections.extend(frame_dets)
            if len(detections) > 200:
                detections = detections[-200:]

            frame_alerts = [d for d in frame_dets if d["object"] in DANGEROUS_OBJECTS]
            # Attach threat score to each alert
            for a in frame_alerts:
                a["threat_score"] = new_threat["score"]
                a["threat_level"] = new_threat["level"]
            alerts.extend(frame_alerts)
            if len(alerts) > 200:
                alerts = alerts[-200:]

            threat_state = new_threat

        time.sleep(FRAME_SLEEP)


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


@app.route("/threat")
def get_threat():
    """Returns current live threat score and assessment."""
    with state_lock:
        return jsonify(threat_state)


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
    app.run(debug=False, use_reloader=False, host="0.0.0.0", port=5000)