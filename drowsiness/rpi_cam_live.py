import subprocess
import cv2
import numpy as np

# Define resolution
width = 640
height = 480

# Start libcamera-vid as subprocess, outputting raw frames to stdout
cmd = [
    'libcamera-vid',
    '-t', '0',                     # Run forever
    '--inline',                    # Inline header
    '--width', str(width),
    '--height', str(height),
    '--framerate', '30',
    '--codec', 'yuv420',
    '--nopreview',# Raw YUV frames
    '-o', '-'                      # Output to stdout
]

# Launch process
process = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=width * height * 3)

# Calculate frame size
frame_size = int(width * height * 1.5)  # YUV420 = 1.5 bytes per pixel

print("Started libcamera-vid stream... Press 'q' to quit!")

while True:
    # Read one frame
    raw_frame = process.stdout.read(frame_size)
    if len(raw_frame) != frame_size:
        print("Incomplete frame")
        break

    # Convert raw YUV420 frame to BGR
    yuv_frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((int(height * 1.5), width))
    bgr_frame = cv2.cvtColor(yuv_frame, cv2.COLOR_YUV2BGR_I420)
    

    print(f"Frame Resolution: {bgr_frame.shape[1]}x{bgr_frame.shape[0]}")
    # Show frame
    cv2.imshow('Raspberry Pi CSI Camera - Rev 1.3', bgr_frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
process.terminate()
cv2.destroyAllWindows()
