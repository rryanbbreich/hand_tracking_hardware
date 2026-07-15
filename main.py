import cv2
import mediapipe as mp
import time
import math
import serial

# Setup Serial Communication
try:
    arduino = serial.Serial(port='COM4', baudrate=9600, timeout=0.1) # Match your COM port
    print("SUCCESS: Connected to Arduino!")
except Exception as e:
    print("WARNING: Running in simulation mode.")
    arduino = None

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

latest_result = None
last_send_time = 0

# SYSTEM STATE MEMORY VARIABLES
stored_brightness = 0       
is_pinching = False         
smoothed_brightness = 0     

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
        
        # DRAW VISUAL ENGAGEMENT ZONES
        midline_x = w // 2
        cv2.line(frame, (midline_x, 0), (midline_x, h), (100, 100, 100), 1)
        cv2.putText(frame, "DEAD ZONE", (30, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)
        cv2.putText(frame, "CONTROL ZONE", (midline_x + 20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        frame_timestamp_ms = int(time.time() * 1000)
        landmarker.detect_async(mp_image, frame_timestamp_ms)

        current_status = "IDLE (Holding Memory)"

        if latest_result and latest_result.hand_landmarks:
            for hand_landmarks in latest_result.hand_landmarks:
                
                # Get target joints
                thumb = hand_landmarks[4]
                index = hand_landmarks[8]
                
                tx, ty = int(thumb.x * w), int(thumb.y * h)
                ix, iy = int(index.x * w), int(index.y * h)
                
                hand_center_x = (tx + ix) // 2

                # ZONE FILTERING
                if hand_center_x > midline_x:
                    distance = math.hypot(ix - tx, iy - ty)
                    
                    if distance < 30:
                        is_pinching = True
                        current_status = "ADJUSTING BRIGHTNESS"
                        cv2.line(frame, (tx, ty), (ix, iy), (0, 255, 255), 3)
                        
                        raw_pct = int(((h - ty) / h) * 100)
                        target_val = max(0, min(100, raw_pct))
                        
                        smoothed_brightness = int(smoothed_brightness * 0.7 + target_val * 0.3)
                        stored_brightness = smoothed_brightness
                    else:
                        is_pinching = False
                        current_status = "READY (Pinch to grab slider)"
                        cv2.circle(frame, (tx, ty), 6, (255, 0, 0), cv2.FILLED)
                        cv2.circle(frame, (ix, iy), 6, (0, 255, 0), cv2.FILLED)

                    # ==========================================================
                    # 🔥 NEW INSERTION: FINGER COUNTING MACROS 🔥
                    # ==========================================================
                    # Track finger tip landmarks vs knuckle base landmarks
                    tips = [8, 12, 16, 20]   # Index, Middle, Ring, Pinky Tips
                    bases = [6, 10, 14, 18]  # Corresponding Knuckle Bases
                    
                    fingers_open = 0
                    for i in range(4):
                        # In image coordinates, smaller Y means closer to the top (Extended finger)
                        if hand_landmarks[tips[i]].y < hand_landmarks[bases[i]].y:
                            fingers_open += 1

                    # Count the thumb separately (check if it extends horizontally past its knuckle)
                    if hand_landmarks[4].x > hand_landmarks[3].x:
                        fingers_open += 1

                    # If we are NOT pinching, apply the fast macro shortcuts based on finger count!
                    if not is_pinching:
                        if fingers_open == 5:
                            stored_brightness = 100
                            current_status = "MACRO: FULL BRIGHT (100%)"
                        elif fingers_open == 0:
                            stored_brightness = 0
                            current_status = "MACRO: BLACKOUT (0%)"
                        elif fingers_open == 3:
                            stored_brightness = 50
                            current_status = "MACRO: READING MODE (50%)"
                    # ==========================================================

                else:
                    current_status = "IGNORED (Inside Dead Zone)"

        # OUTBOUND HARDWARE DATA PIPELINE
        current_time = time.time()
        if arduino and (current_time - last_send_time > 0.05):
            data_packet = f"H,{stored_brightness}\n"
            arduino.write(data_packet.encode('utf-8'))
            last_send_time = current_time

        # UI Overlay Dashboard
        cv2.putText(frame, f"Status: {current_status}", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"Hardware LED Output: {stored_brightness}%", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.imshow("Advanced Interface Engine", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if arduino:
    arduino.close()
cap.release()
cv2.destroyAllWindows()
