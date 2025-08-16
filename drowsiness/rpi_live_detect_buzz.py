from ultralytics import YOLO
import cv2
import RPi.GPIO as GPIO
import time

# === GPIO Setup ===
BUZZER_PIN = 17  # BCM GPIO17 (Pin 11)
GPIO.setmode(GPIO.BCM)
GPIO.setup(BUZZER_PIN, GPIO.OUT)
GPIO.output(BUZZER_PIN, GPIO.LOW)

# === Load YOLO Model ===
model = YOLO("best.pt")

cap = cv2.VideoCapture(0)

eye_closed_start_time = None
buzzer_active = False
buzzer_end_time = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_time = time.time()
        results = model(frame, imgsz=352, stream=True)

        eyes_closed = False  # Reset per frame

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                label = model.names[cls]

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                # DEBUG DETECTIONS
                print(f"[YOLO] Detected: {label} | Confidence: {conf:.2f}")

                # Detection logic
                if label == "eye_closed":
                    eyes_closed = True

        # === Timer Logic ===
        if eyes_closed and not buzzer_active:
            if eye_closed_start_time is None:
                eye_closed_start_time = current_time
                print("[INFO] Eyes closed detected - starting timer...")
            elif current_time - eye_closed_start_time >= 3:
                print("[ALERT] Eyes closed > 3 sec – BUZZ ON!")
                GPIO.output(BUZZER_PIN, GPIO.HIGH)
                buzzer_active = True
                buzzer_end_time = current_time + 10
        else:
            if not eyes_closed:
                if eye_closed_start_time:
                    print("[INFO] Eyes reopened before 3 seconds.")
                eye_closed_start_time = None

        # === Buzzer duration control ===
        if buzzer_active and current_time >= buzzer_end_time:
            print("[INFO] Buzzer period ended – turning OFF.")
            GPIO.output(BUZZER_PIN, GPIO.LOW)
            buzzer_active = False
            eye_closed_start_time = None

        cv2.imshow("YOLO Eye Monitor", frame)
        if cv2.waitKey(1) == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    GPIO.output(BUZZER_PIN, GPIO.LOW)
    GPIO.cleanup()
