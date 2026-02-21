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