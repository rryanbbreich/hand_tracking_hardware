import cv2
import mediapipe as mp
import time
import math
import serial  # <--- NEW: Import the Serial library to talk to hardware

# 1. SETUP SERIAL PORT (ADJUST COM PORT NUMBER TO MATCH YOUR ARDUINO)
try:
    # Change 'COM3' to whatever port your Arduino uses when plugged in
    arduino = serial.Serial(port='COM3', baudrate=9600, timeout=0.1)
    print("SUCCESS: Connected to Arduino!")
except Exception as e:
    print("WARNING: Arduino not detected. Running in simulation mode.")
    arduino = None

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

latest_result = None
last_send_time = 0  # <--- NEW: Limit data speed so we don't crash the hardware

def print_result(result: mp.tasks.vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_result
    latest_result = result

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_hands=1,
    result_callback=print_result
)

cap = cv2.VideoCapture(0)

with HandLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        h, w, c = frame.shape
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        frame_timestamp_ms = int(time.time() * 1000)
        landmarker.detect_async(mp_image, frame_timestamp_ms)

        if latest_result and latest_result.hand_landmarks:
            for hand_landmarks in latest_result.hand_landmarks:
                
                # Extract landmarks for fingers
                thumb = hand_landmarks[4]
                index = hand_landmarks[8]
                middle_tip = hand_landmarks[12]
                middle_base = hand_landmarks[9]

                tx, ty = int(thumb.x * w), int(thumb.y * h)
                ix, iy = int(index.x * w), int(index.y * h)
                my_tip = int(middle_tip.y * h)
                my_base = int(middle_base.y * h)

                # --- GESTURE 1: TOGGLE SWITCH ---
                if my_tip < my_base:
                    gesture_status = "SYSTEM: ON"
                    text_color = (0, 255, 0)
                    hardware_command = "H" # 'H' for High/On
                else:
                    gesture_status = "SYSTEM: OFF"
                    text_color = (0, 0, 255)
                    hardware_command = "L" # 'L' for Low/Off

                # --- GESTURE 2: PINCH DIMMER ---
                distance = math.hypot(ix - tx, iy - ty)
                brightness_pct = int(((distance - 20) / 130) * 100)
                brightness_pct = max(0, min(100, brightness_pct))

                # --- NEW: SEND DATA TO ARDUINO ---
                # We limit sending to once every 50ms so the Arduino isn't overwhelmed
                current_time = time.time()
                if arduino and (current_time - last_send_time > 0.05):
                    # Package data as string: "H,45\n" (Command,SliderValue)
                    data_packet = f"{hardware_command},{brightness_pct}\n"
                    
                    # Convert text to raw bytes and send down the USB cord
                    arduino.write(data_packet.encode('utf-8'))
                    last_send_time = current_time

                # Display info
                cv2.putText(frame, f"Switch: {gesture_status}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
                cv2.putText(frame, f"Slider Level: {brightness_pct}%", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        cv2.imshow("Gesture Engine Control Panel", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if arduino:
    arduino.close() # Safely unlock the USB port when quitting
cap.release()
cv2.destroyAllWindows()
