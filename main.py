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

# --- NEW: SYSTEM STATE MEMORY VARIABLES ---
stored_brightness = 0       # Holds the last locked-in brightness value
is_pinching = False         # Tracks if the user is actively adjusting right now
smoothed_brightness = 0     # Blended value to eliminate hand jitter

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
        
        # --- NEW: DRAW VISUAL ENGAGEMENT ZONES ---
        # Split the screen vertically down the middle
        midline_x = w // 2
        cv2.line(frame, (midline_x, 0), (midline_x, h), (100, 100, 100), 1)
        cv2.putText(frame, "DEAD ZONE", (30, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)
        cv2.putText(frame, "CONTROL ZONE", (midline_x + 20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        frame_timestamp_ms = int(time.time() * 1000)
        landmarker.detect_async(mp_image, frame_timestamp_ms)

        # Clear active status text per frame unless updated by detection
        current_status = "IDLE (Holding Memory)"

        if latest_result and latest_result.hand_landmarks:
            for hand_landmarks in latest_result.hand_landmarks:
                
                # Get target joints
                thumb = hand_landmarks[4]
                index = hand_landmarks[8]
                
                tx, ty = int(thumb.x * w), int(thumb.y * h)
                ix, iy = int(index.x * w), int(index.y * h)
                
                # Calculate hand center to see which zone it is inside
                hand_center_x = (tx + ix) // 2

                # --- ADVANCED LOGIC: ZONE FILTERING ---
                if hand_center_x > midline_x:
                    # Calculate pinch distance
                    distance = math.hypot(ix - tx, iy - ty)
                    
                    # If fingers are very close (under 30 pixels), register a pinch engagement
                    if distance < 30:
                        is_pinching = True
                        current_status = "ADJUSTING BRIGHTNESS"
                        
                        # Draw an engagement line between fingers
                        cv2.line(frame, (tx, ty), (ix, iy), (0, 255, 255), 3)
                        
                        # Use the vertical Y position of your pinched hand as the slider height!
                        # Moving your pinched hand UP decreases Y pixel coordinate, so we invert it.
                        raw_pct = int(((h - ty) / h) * 100)
                        target_val = max(0, min(100, raw_pct))
                        
                        # Smooth out the jumping numbers (Low-pass filter math)
                        smoothed_brightness = int(smoothed_brightness * 0.7 + target_val * 0.3)
                        stored_brightness = smoothed_brightness
                    else:
                        is_pinching = False
                        current_status = "READY (Pinch to grab slider)"
                        cv2.circle(frame, (tx, ty), 6, (255, 0, 0), cv2.FILLED)
                        cv2.circle(frame, (ix, iy), 6, (0, 255, 0), cv2.FILLED)
                else:
                    current_status = "IGNORED (Inside Dead Zone)"

        # --- OUTBOUND HARDWARE DATA PIPELINE ---
        current_time = time.time()
        if arduino and (current_time - last_send_time > 0.05):
            # We always broadcast the stored_brightness memory value!
            # Using 'H' as a generic active command character placeholder
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
