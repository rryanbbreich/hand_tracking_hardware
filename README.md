I used AI to make this readme file, just to clarify, for simplicity.

# AI Gesture-Controlled Hardware Engine 🚀🤖

A full-stack computer vision project that tracks real-time hand movements via a webcam and translates 3D coordinate gestures into physical hardware actions using Python and an Arduino.

## ✨ Features
- **3D Hand Tracking:** Utilises the modern Google MediaPipe Tasks API to map 21 distinct hand landmark coordinates.
- **Pinch-to-Dim Slider:** Calculates the pixel distance between the thumb and index finger using the Pythagorean theorem to dynamically dim an external LED (0-100%).
- **Gesture Toggle Switch:** Tracks finger-to-knuckle thresholds to trigger an absolute ON/OFF switch.
- **Serial Communication:** Streams formatted data packets (`Command,Value\n`) across a USB pipeline via `pyserial`.

## 🛠️ Tech Stack & Components
- **Software:** Python 3.12+, OpenCV, MediaPipe, PySerial
- **IDE:** VS Code (Virtual Environment isolated)
- **Hardware:** Arduino Uno, External LED, 220-ohm Resistor, Breadboard

## 🚀 How to Run It Local

### 1. Clone & Setup Environment
```bash
# Clone the repository
git clone <PASTE_YOUR_GITHUB_REPO_URL_HERE>
cd hand_tracking_hardware

# Create and activate virtual environment
py -m venv venv --system-site-packages
.\venv\Scripts\Activate.ps1

# Install dependencies
.\venv\Scripts\python.exe -m pip install opencv-python mediapipe pyserial
```

### 2. Flash the Arduino
1. Open the Arduino IDE.
2. Upload the C++ listener code (found in the architecture layout) to your board.
3. Check your Windows Device Manager to locate your **COM Port**.

### 3. Run the Python Vision Engine
Update the `COM` port string inside `main.py` to match your board, then launch:
```powershell
.\venv\Scripts\python.exe main.py
```
*Press **'q'** in the webcam frame to exit safely.*
