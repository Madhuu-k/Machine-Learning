import numpy as np
import cv2
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
    min_hand_detection_confidence=0.6,
    min_hand_presence_confidence=0.6,
    min_tracking_confidence=0.5
)

landmarker = vision.HandLandmarker.create_from_options(options)

HAND_CONNECTIONS = [
    [0, 1], [1, 2], [2, 3], [3, 4],        # Thumb
    [0, 5], [5, 6], [6, 7], [7, 8],        # Index
    [0, 9], [9, 10], [10, 11], [11, 12],   # Middle
    [0, 13], [13, 14], [14, 15], [15, 16], # Ring
    [0, 17], [17, 18], [18, 19], [19, 20]  # Pinky
]

capture = cv2.VideoCapture(0)
if not capture:
    print("Camera not detected")
    exit
    
while True:
    ret, frame = capture.read()
    if not ret:
        print("Cannot Capture Video")
        break
    
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=frame
    )
    
    timestamp = int(time.time() * 1000)
    result = landmarker.detect_for_video(mp_image, timestamp)
    
    if result.hand_landmarks:
        hand = result.hand_landmarks[0]
        
        for start, end in HAND_CONNECTIONS:
            x1 = int(hand[start].x * w)
            y1 = int(hand[start].y * h)
            x2 = int(hand[end].x * w)
            y2 = int(hand[end].y * h)
            
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            for i, lm in enumerate(hand):
                x = int(lm.x * w)
                y = int(lm.y * h)
                
                cv2.circle(frame, (x, y), 6, (0, 0, 255), -1)
                cv2.putText(
                    frame,
                    str(i),
                    (x + 5, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 255, 255),
                    1
                )
                
    cv2.imshow("Hand Skeleton", frame)
            
    if cv2.waitKey(1) & 0xFF == ord('q'):            
        break
    
capture.release()
cv2.destroyAllWindows()
    