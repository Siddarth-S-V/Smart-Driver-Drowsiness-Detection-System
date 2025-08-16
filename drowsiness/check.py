from ultralytics import YOLO
import cv2
import RPi.GPIO as GPIO
import time

# === GPIO Setup ===
VIBRATION_PIN = 17  # PWM-compatible pin
GPIO.setmode(GPIO.BCM)
GPIO.setup(VIBRATION_PIN, GPIO.OUT)
vibration_pwm = GPIO.PWM(VIBRATION_PIN, 1000)  # 1kHz frequency
vibration_pwm.start(0)

# === Load YOLO Model ===
model = YOLO("best.pt")

# === Open ESP32-CAM Stream ===
# Replace this with your ESP32-CAM IP
ESP32_CAM_URL = "http://192.168.1.46:81/stream"
cap = cv2.VideoCapture(ESP32_CAM_URL)

# Sanity check
if not cap.isOpened():
    print("[ERROR] Could not open ESP32-CAM stream. Check IP or network!")
    exit(1)

cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 352)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 288)

# === Logic Vars ===
eye_closed_start_time = None
vibration_active = False
vibration_end_time = 0
drowsy_count = 0
last_drowsy_time = time.time()

# === Constants ===
RESET_AFTER_SECONDS = 600  # 10 mins

def set_vibration_level(count):
    if count == 1:
        return 25
    elif count == 2:
        return 50
    elif count == 3:
        return 75
    else:
        return 100

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Failed to grab frame from ESP32-CAM. Retrying...")
            continue

        current_time = time.time()
        results = model(frame, imgsz=352, stream=False)

        eyes_closed = False

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                label = model.names[cls]

                if "eye_closed" in label.lower():
                    print(f"[YOLO] Detected: {label} | Confidence: {conf:.2f}")
                    eyes_closed = True

        # === Drowsiness Detection Logic ===
        if eyes_closed and not vibration_active:
            if eye_closed_start_time is None:
                eye_closed_start_time = current_time
                print("[INFO] Eyes closed detected - timer started")
            elif current_time - eye_closed_start_time >= 1:
                drowsy_count += 1
                last_drowsy_time = current_time
                intensity = set_vibration_level(drowsy_count)
                print(f"[ALERT] Drowsiness #{drowsy_count} – Vibrating at {intensity}% duty")
                vibration_pwm.ChangeDutyCycle(intensity)
                vibration_active = True
                vibration_end_time = current_time + 3
        else:
            if not eyes_closed:
                eye_closed_start_time = None

        # === Turn off vibration after 3 seconds ===
        if vibration_active and current_time >= vibration_end_time:
            print("[INFO] Vibration OFF.")
            vibration_pwm.ChangeDutyCycle(0)
            vibration_active = False
            eye_closed_start_time = None

        # === Reset drowsy count if awake long enough ===
        if drowsy_count > 0 and (current_time - last_drowsy_time) >= RESET_AFTER_SECONDS:
            print("[RESET] Driver has been attentive for 10 minutes. Resetting drowsy count.")
            drowsy_count = 0

        time.sleep(0.01)
        

except KeyboardInterrupt:
    print("\n[SHUTDOWN] Ctrl+C received. Cleaning up...")

finally:
    cap.release()
    vibration_pwm.ChangeDutyCycle(0)
    vibration_pwm.stop()
    GPIO.cleanup()
