import subprocess
import cv2
import numpy as np
import time
from ultralytics import YOLO
import RPi.GPIO as GPIO

# === GPIO Setup ===
BUZZER_PIN = 17
GPIO.setmode(GPIO.BCM)
GPIO.setup(BUZZER_PIN, GPIO.OUT)
GPIO.output(BUZZER_PIN, GPIO.LOW)

# === YOLO Model ===
model = YOLO("best.pt")  # Your trained model

# === libcamera-vid stream settings ===
WIDTH = 352
HEIGHT = 288
FPS = 10

# === Start libcamera-vid subprocess piped into ffmpeg ===
cmd = [
    "libcamera-vid",
    "-t", "0",  # infinite duration
    "--inline",
    "--width", str(WIDTH),
    "--height", str(HEIGHT),
    "--framerate", str(FPS),
    "--codec", "yuv420",
    "-o", "-"  # Output to stdout
]

ffmpeg_cmd = [
    "ffmpeg",
    "-f", "rawvideo",
    "-pix_fmt", "yuv420p",
    "-s", f"{WIDTH}x{HEIGHT}",
    "-r", str(FPS),
    "-i", "-",  # Input from stdin
    "-f", "rawvideo",
    "-pix_fmt", "bgr24",
    "-"
]

# Pipe libcamera-vid into ffmpeg and get frame stream
libcamera = subprocess.Popen(cmd, stdout=subprocess.PIPE)
ffmpeg = subprocess.Popen(ffmpeg_cmd, stdin=libcamera.stdout, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

# === Logic Vars ===
eye_closed_start_time = None
buzzer_active = False
buzzer_end_time = 0

try:
    while True:
        # Read frame from ffmpeg stdout
        raw_frame = ffmpeg.stdout.read(WIDTH * HEIGHT * 3)
        if not raw_frame:
            print("[WARN] No frame data")
            continue

        frame = np.frombuffer(raw_frame, np.uint8).reshape((HEIGHT, WIDTH, 3))

        # YOLO detection
        current_time = time.time()
        results = model(frame, imgsz=352, stream=False)

        eyes_closed = False

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                label = model.names[cls]
                conf = float(box.conf[0])
                if "eye_closed" in label.lower():
                    print(f"[YOLO] Detected: {label} | Confidence: {conf:.2f}")
                    eyes_closed = True

        # === Eye Closed Logic ===
        if eyes_closed and not buzzer_active:
            if eye_closed_start_time is None:
                eye_closed_start_time = current_time
                print("[INFO] Eyes closed detected - timer started")
            elif current_time - eye_closed_start_time >= 1:
                print("[ALERT] Eyes closed > 3 sec – Buzzer ON!")
                GPIO.output(BUZZER_PIN, GPIO.HIGH)
                buzzer_active = True
                buzzer_end_time = current_time + 3
        else:
            if not eyes_closed:
                eye_closed_start_time = None

        if buzzer_active and current_time >= buzzer_end_time:
            print("[INFO] Buzzer period ended – turning OFF.")
            GPIO.output(BUZZER_PIN, GPIO.LOW)
            buzzer_active = False
            eye_closed_start_time = None

        # Optional: Show live video
        # cv2.imshow("Live CSI Feed", frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

except KeyboardInterrupt:
    print("\n[SHUTDOWN] Ctrl+C received. Cleaning up...")

finally:
    GPIO.output(BUZZER_PIN, GPIO.LOW)
    GPIO.cleanup()
    libcamera.terminate()
    ffmpeg.terminate()
    # cv2.destroyAllWindows()
