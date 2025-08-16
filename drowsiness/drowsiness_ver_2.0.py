from ultralytics import YOLO
import cv2
import RPi.GPIO as GPIO
import time
import subprocess
import os

# === GPIO Setup ===
VIBRATION_PIN = 17  # PWM-compatible pin
GPIO.setmode(GPIO.BCM)
GPIO.setup(VIBRATION_PIN, GPIO.OUT)
vibration_pwm = GPIO.PWM(VIBRATION_PIN, 1000)  # 1kHz frequency
vibration_pwm.start(0)

# === Load YOLOv8 Model ===
model = YOLO("best.pt")

# === Logic Vars ===
eye_closed_start_time = None
vibration_active = False
vibration_end_time = 0
drowsy_count = 0
last_drowsy_time = time.time()

# === Constants ===
IMG_PATH = "frame.jpg"
RESET_AFTER_SECONDS = 600  # 10 minutes

def set_vibration_level(count):
    """Set PWM duty cycle based on number of drowsy events."""
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
        # === Capture image from CSI Camera using libcamera-jpeg ===
        subprocess.run([
            "libcamera-jpeg", "-n", "-o", IMG_PATH,
            "--width", "352", "--height", "288",
            "--quality", "85", "--timeout", "1"
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        if not os.path.exists(IMG_PATH):
            print("[ERROR] Frame not captured!")
            continue

        frame = cv2.imread(IMG_PATH)
        if frame is None:
            print("[ERROR] Failed to read captured image")
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
                print("[INFO] Eyes closed detected – timer started")
            elif current_time - eye_closed_start_time >= 1:
                drowsy_count += 1
                last_drowsy_time = current_time
                intensity = set_vibration_level(drowsy_count)
                print(f"[ALERT] Drowsiness #{drowsy_count} – Vibrating at {intensity}% duty cycle")
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

        # === Reset drowsy count if alert for 10 min ===
        if drowsy_count > 0 and (current_time - last_drowsy_time) >= RESET_AFTER_SECONDS:
            print("[RESET] Driver stayed alert for 10 mins. Resetting drowsy count.")
            drowsy_count = 0

        time.sleep(0.01)

except KeyboardInterrupt:
    print("\n[SHUTDOWN] Ctrl+C received. Exiting gracefully...")

finally:
    vibration_pwm.ChangeDutyCycle(0)
    vibration_pwm.stop()
    GPIO.cleanup()
    if os.path.exists(IMG_PATH):
        os.remove(IMG_PATH)
