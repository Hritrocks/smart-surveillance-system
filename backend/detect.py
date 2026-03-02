from ultralytics import YOLO
import cv2
import time
from datetime import datetime

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Dangerous objects list
dangerous_objects = ["knife", "scissors", "gun", "cell phone"]

# Alert cooldown
last_alert_time = 0
alert_cooldown = 3  # seconds

# Open webcam (stable Windows mode)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    current_time = time.time()

    for result in results:
        boxes = result.boxes
        for box in boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = model.names[class_id]

            # Ignore low confidence
            if confidence < 0.50:
                continue

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            if class_name in dangerous_objects:
                if current_time - last_alert_time > alert_cooldown:
                    alert_message = f"{timestamp} | {class_name} | {confidence:.2f} | ALERT"
                    print("🚨", alert_message)

                    with open("detections.log", "a") as f:
                        f.write(alert_message + "\n")

                    last_alert_time = current_time
            else:
                safe_message = f"{timestamp} | {class_name} | {confidence:.2f} | SAFE"
                print(safe_message)

                with open("detections.log", "a") as f:
                    f.write(safe_message + "\n")

    annotated_frame = results[0].plot()
    cv2.imshow("Smart Surveillance System", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()