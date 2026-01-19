import cv2
import numpy as np
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


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

landmarker = vision.HandLandmarker.create_from_options(options)
OFFSET_Y = -40
smooth_x, smooth_y = None, None
ALPHA = 0.2  # 20% Learning
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
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
        anchor_lm = hand[5]

        px = int(anchor_lm.x * w)
        py = int(anchor_lm.y * h) + OFFSET_Y
        
        if smooth_x is None:
            smooth_x, smooth_y = px, py
        
        else:
            smooth_x = int(ALPHA * px + (1 - ALPHA) * smooth_x)
            smooth_y = int(ALPHA * py + (1 - ALPHA) * smooth_y)
            
    if not result.hand_landmarks:
        cv2.putText(
        frame,
        "NO HAND DETECTED",
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2
    )

    
    if smooth_x is not None:
        cv2.circle(frame, (smooth_x, smooth_y), 15, (0, 0, 255), -1)

    cv2.imshow("Sudarshan", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
