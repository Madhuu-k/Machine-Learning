import cv2
import numpy as np
import mediapipe as mp
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pynput.mouse import Controller
import pyautogui

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0

mouse = Controller()

MODEL_PATH = r"D:\Machine Learning\Open CV\hand_landmarker.task"

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.7
)

landmarks = vision.HandLandmarker.create_from_options(options)

ALPHA = 0.15
smoothed_landmarks = None
prev_time = 0

last_zoom_time = 0
ZOOM_COOLDOWN = 0.25
ZOOM_ALPHA = 0.35

zoom_state = "STABLE"
zoom_smooth = None
prev_zoom = None

zoom_display_time = 0
ZOOM_DISPLAY_DURATION = 0.25

ZOOM_THRESHOLD = 0.008
ZOOM_SPEED_MULTIPLIER = 40

MIN_CONTROL = 0.05
MAX_CONTROL = 0.35

# AIR MOUSE VARIABLES

screen_width, screen_height = pyautogui.size()

cursor_x = None
cursor_y = None

CURSOR_ALPHA = 0.2

air_mouse_active = False
prev_index_state = True 
index_down_time = 0

last_cursor_update = 0
CURSOR_UPDATE_INTERVAL = 0.01

capture = cv2.VideoCapture(0)

while capture.isOpened():

    ret, frame = capture.read()
    if not ret:
        print("Failed to capture video.")
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    )

    timestamp = int(time.time() * 1000)
    result = landmarks.detect_for_video(mp_image, timestamp)

    if result.hand_landmarks:

        hand = result.hand_landmarks[0]

        if smoothed_landmarks is None:
            smoothed_landmarks = []
            for lm in hand:
                smoothed_landmarks.append([lm.x, lm.y, lm.z])
        else:
            for i in range(21):
                smoothed_landmarks[i][0] = ALPHA * hand[i].x + (1 - ALPHA) * smoothed_landmarks[i][0]
                smoothed_landmarks[i][1] = ALPHA * hand[i].y + (1 - ALPHA) * smoothed_landmarks[i][1]
                smoothed_landmarks[i][2] = ALPHA * hand[i].z + (1 - ALPHA) * smoothed_landmarks[i][2]

        for lm in smoothed_landmarks:
            x, y = int(lm[0] * w), int(lm[1] * h)
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
            
        wrist_landmark = smoothed_landmarks[0]

        thumb_tip = smoothed_landmarks[4]
        thumb_pip = smoothed_landmarks[3]
        thumb_mcp = smoothed_landmarks[2]
        thumb_up = ( np.linalg.norm(np.array(thumb_tip[:2]) - np.array(wrist_landmark[:2])) >
                    np.linalg.norm(np.array(thumb_pip[:2]) - np.array(wrist_landmark[:2])) * 1.15 )

        index_tip = smoothed_landmarks[8]
        index_pip = smoothed_landmarks[6]
        index_mcp = smoothed_landmarks[5]
        index_up = ( np.linalg.norm(np.array(index_tip[:2]) - np.array(wrist_landmark[:2])) >
                    np.linalg.norm(np.array(index_pip[:2]) - np.array(wrist_landmark[:2])) * 1.15 )

        middle_tip = smoothed_landmarks[12]
        middle_pip = smoothed_landmarks[10]
        middle_mcp = smoothed_landmarks[9]
        middle_up = ( np.linalg.norm(np.array(middle_tip[:2]) - np.array(wrist_landmark[:2])) >
                      np.linalg.norm(np.array(middle_pip[:2]) - np.array(wrist_landmark[:2])) * 1.15 )

        ring_tip = smoothed_landmarks[16]
        ring_pip = smoothed_landmarks[14]
        ring_mcp = smoothed_landmarks[13]
        ring_up  = ( np.linalg.norm(np.array(ring_tip[:2]) - np.array(wrist_landmark[:2])) >
                    np.linalg.norm(np.array(ring_pip[:2]) - np.array(wrist_landmark[:2])) * 1.15 )

        pinky_tip = smoothed_landmarks[20]
        pinky_pip = smoothed_landmarks[18]
        pinky_mcp = smoothed_landmarks[17]
        pinky_up = ( np.linalg.norm(np.array(pinky_tip[:2]) - np.array(wrist_landmark[:2])) >
                    np.linalg.norm(np.array(pinky_pip[:2]) - np.array(wrist_landmark[:2])) * 1.15 )

        finger_state_array = [thumb_up, index_up, middle_up, ring_up, pinky_up]
        air_mouse_active = (not thumb_up and index_up and not middle_up and not ring_up and not pinky_up)   
    
        dx = thumb_tip[0] - index_tip[0]
        dy = thumb_tip[1] - index_tip[1]
        distance = np.sqrt(dx**2 + dy**2)

        zoom_control = (distance - MIN_CONTROL) / (MAX_CONTROL - MIN_CONTROL)
        zoom_control = np.clip(zoom_control, 0, 1)

        if zoom_smooth is None:
            zoom_smooth = zoom_control
            prev_zoom = zoom_smooth
        else:
            zoom_smooth = ZOOM_ALPHA * zoom_control + (1 - ZOOM_ALPHA) * zoom_smooth

        zoom_delta = zoom_smooth - prev_zoom
        prev_zoom = zoom_smooth

        if abs(zoom_delta) < 0.003:
            zoom_delta = 0

        current_time = time.time()

        if (not air_mouse_active and 
            MIN_CONTROL < distance < MAX_CONTROL and 
            finger_state_array[0] and finger_state_array[1] and 
            not any(finger_state_array[2:])):
            
            if abs(zoom_delta) > ZOOM_THRESHOLD and (current_time - last_zoom_time) > ZOOM_COOLDOWN:

                zoom_steps = int(abs(zoom_delta) * ZOOM_SPEED_MULTIPLIER)
                zoom_steps = max(1, min(zoom_steps, 5))

                if zoom_delta > 0:
                    zoom_state = "ZOOM OUT"

                    for _ in range(zoom_steps):
                        pyautogui.hotkey('ctrl', '-')

                else:
                    zoom_state = "ZOOM IN"

                    for _ in range(zoom_steps):
                        pyautogui.hotkey('ctrl', '=')

                last_zoom_time = current_time
                zoom_display_time = current_time

        if current_time - zoom_display_time > ZOOM_DISPLAY_DURATION:
            zoom_state = "STABLE"
        
        # AIR MOUSE LOGIC
        if air_mouse_active:
            raw_x = index_tip[0] * screen_width
            raw_y = index_tip[1] * screen_height
            
            if cursor_x is None:
                cursor_x, cursor_y = raw_x, raw_y
            else: 
                # Apply exponential smoothing to cursor position
                cursor_x = CURSOR_ALPHA * raw_x + (1 - CURSOR_ALPHA) * cursor_x 
                cursor_y = CURSOR_ALPHA * raw_y + (1 - CURSOR_ALPHA) * cursor_y
                
            if time.time() - last_cursor_update > CURSOR_UPDATE_INTERVAL:
                mouse.position = (int(cursor_x), int(cursor_y))
                last_cursor_update = time.time()
            
            #  Check is index finger just went up to trigger a click
            current_index_state = index_up
            if prev_index_state and not current_index_state:
                index_down_time = time.time()
                
            if not prev_index_state and current_index_state:
                if time.time() - index_down_time < 0.5: 
                    pyautogui.click()
                    
            prev_index_state = current_index_state   
        
        cv2.putText(frame, zoom_state, (10,110),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)

        if finger_state_array[0] and finger_state_array[1] and not any(finger_state_array[2:]):
            cv2.putText(frame, f"Distance: {distance:.3f}", (10,70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            
        if air_mouse_active:
            cv2.putText(frame, "AIR MOUSE ACTIVE", (10,150),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

    curr_time = time.time()
    if prev_time != 0:
        fps = 1 / (curr_time - prev_time)
    else:
        fps = 0
    prev_time = curr_time

    cv2.putText(frame, f'FPS: {int(fps)}', (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Air Zoom", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()