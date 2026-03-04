import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time

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
prev_distance_smooth = None
DIST_ALPHA = 0.3

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
            
        thumb_tip = smoothed_landmarks[4]
        thumb_pip = smoothed_landmarks[3]
        thumb_mcp = smoothed_landmarks[2]
        thumb_up = thumb_tip[1] < thumb_pip[1] < thumb_mcp[1]
                  
        index_tip = smoothed_landmarks[8]
        index_pip = smoothed_landmarks[6]
        index_mcp = smoothed_landmarks[5]
        index_up = index_tip[1] < index_pip[1] < index_mcp[1]
        
        middle_tip = smoothed_landmarks[12]
        middle_pip = smoothed_landmarks[10]
        middle_mcp = smoothed_landmarks[9]
        middle_up = middle_tip[1] < middle_pip[1] < middle_mcp[1]
        
        ringfinger_tip = smoothed_landmarks[16]
        ringfinger_pip = smoothed_landmarks[14]
        ringfinger_mcp = smoothed_landmarks[13]
        ringfinger_up = ringfinger_tip[1] < ringfinger_pip[1]
        
        pinky_tip = smoothed_landmarks[20]
        pinky_pip = smoothed_landmarks[18]
        pinky_mcp = smoothed_landmarks[17]
        pinky_up = pinky_tip[1] < pinky_pip[1] < pinky_mcp[1]
        
        # COMPUTE EUCLIDEAN DISTANCE FOR ZOOM GESTURE
        dx = thumb_tip[0] - index_tip[0]
        dy = thumb_tip[1] - index_tip[1]
        distance = np.sqrt(dx**2 + dy**2)            
        
        finger_state_array = [thumb_up, index_up, middle_up, ringfinger_up, pinky_up]
        
        MIN_CONTROL = 0.05
        MAX_CONTROL = 0.35
        
        if prev_distance_smooth is None:
            prev_distance_smooth = distance
            delta = 0
        else:
            distance_smooth = DIST_ALPHA * distance + (1 - DIST_ALPHA) * prev_distance_smooth
            delta = distance_smooth - prev_distance_smooth
            prev_distance_smooth = distance_smooth
        
        if finger_state_array[0] and finger_state_array[1] and not any(finger_state_array[2:]):
            cv2.putText(frame, f"Distance: {distance:.2f}, Delta: {delta:.2f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            
        
    # FPS calculation
    curr_time = time.time()
    if prev_time != 0:
        fps = 1 / (curr_time - prev_time)
    else:
        fps = 0
    prev_time = curr_time
    
    cv2.putText(frame, f'FPS: {int(fps)}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Always show the frame
    cv2.imshow("Air Zoom", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
capture.release()
cv2.destroyAllWindows()