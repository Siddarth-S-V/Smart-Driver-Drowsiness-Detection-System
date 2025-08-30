# 🚗 Real-Time Driver Drowsiness Detection System

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Raspberry Pi](https://img.shields.io/badge/Raspberry%20Pi-A22846?style=for-the-badge&logo=raspberrypi&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)

A real-time driver drowsiness detection system developed using a **Raspberry Pi 4** and a **YOLO-based** deep learning model. This project monitors a driver's eye state to detect signs of fatigue and triggers an alert to prevent accidents.



***

## 🎯 Overview

Driving while drowsy is a leading cause of fatal road accidents. This project provides a cost-effective, standalone solution to enhance driver safety. By leveraging computer vision and deep learning on an embedded device, the system continuously analyzes the video feed from a camera pointed at the driver. It uses a custom-trained YOLO model to detect if the driver's eyes are open or closed. If the eyes remain closed for a prolonged period, the system classifies the driver as drowsy and sounds an alarm.

***

## ✨ Features

* **Real-Time Monitoring:** Captures and processes video frames in real-time with minimal latency.
* **Deep Learning-Powered:** Utilizes a lightweight YOLO (You Only Look Once) model for accurate eye state detection.
* **Drowsiness Algorithm:** Implements a timer-based algorithm to detect prolonged eye closure, a key indicator of microsleep.
* **Audible Alerts:** Triggers a loud alarm to alert the drowsy driver immediately.
* **Embedded & Cost-Effective:** Runs entirely on a Raspberry Pi 4, making it a portable and affordable safety device for any vehicle.

***

## ⚙️ How It Works

The system follows a simple yet effective workflow:

1.  **Video Capture:** A camera connected to the Raspberry Pi captures the driver's face.
2.  **Frame Processing:** The Raspberry Pi processes each frame from the video stream.
3.  **Eye State Detection:** The YOLO model analyzes the frame to detect the driver's eyes and classify their state as **'open'** or **'closed'**.
4.  **Drowsiness Calculation:**
    * If the eyes are detected as 'closed', a timer starts.
    * If the eyes are 'open', the timer resets.
5.  **Alert Trigger:** If the eye closure timer exceeds a predefined threshold (e.g., 2-3 seconds), the system triggers an audible alarm to wake the driver.
6.  **Continuous Loop:** The process repeats, providing constant monitoring.



***

## 🛠️ Hardware & Software Requirements

### Hardware
* **Raspberry Pi 4** (2GB or higher recommended)
* **Pi Camera Module** or a compatible USB Webcam
* **Buzzer** or small speaker for the alarm
* 5V/3A Power Supply
* MicroSD Card (16GB or higher)

### Software
* **Raspberry Pi OS** (formerly Raspbian)
* **Python 3.7+**
* **OpenCV** for image processing
* **PyTorch** or **TensorFlow/Keras** (depending on the YOLO model implementation)
* NumPy
* `requirements.txt` file:
    ```
    opencv-python
    numpy
    torch
    torchvision
    # Add other specific libraries here
    ```

***

## 🚀 Setup & Installation

1.  **Flash Raspberry Pi OS:** Install the latest version of Raspberry Pi OS on your microSD card.

2.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/your-username/driver-drowsiness-detection.git](https://github.com/your-username/driver-drowsiness-detection.git)
    cd driver-drowsiness-detection
    ```

3.  **Install Dependencies:** It's highly recommended to use a virtual environment.
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

4.  **Hardware Connection:**
    * Connect the Pi Camera to the CSI port or the USB webcam to a USB port.
    * Connect the buzzer to the GPIO pins (e.g., GPIO18 and a Ground pin).

5.  **Download Model Weights:** Place your trained YOLO model weights (`best.pt` or `drowsy.weights`) into the `models/` directory.

***

## ▶️ Usage

Run the main detection script from the terminal:

```bash
python detect_drowsiness.py
