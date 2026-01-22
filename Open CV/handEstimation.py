import cv2
import numpy as np
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Path to your hand landmark model
MODEL_PATH = r"D:\Machine Learning\Open CV\hand_landmarker.task"

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (5, 9), (9, 10), (10, 11), (11, 12),   # Middle
    (9, 13), (13, 14), (14, 15), (15, 16), # Ring
    (13, 17), (17, 18), (18, 19), (19, 20),# Pinky
    (0, 17)                                # Palm base to pinky base
]


# Base options and hand landmarker setup
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

# Parameters
OFFSET_Y = -40
ALPHA = 0.25  # smoothing factor
GRACE_PERIOD = 0.4

cap = cv2.VideoCapture(0)

smoothed_landmarks = None
last_seen_time = 0

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

    current_time = time.time()

    if result.hand_landmarks:
        hand = result.hand_landmarks[0]

        # Initialize smoothed landmarks
        if smoothed_landmarks is None:
            smoothed_landmarks = []
            for lm in hand:
                new_lm = type(lm)()
                new_lm.x, new_lm.y, new_lm.z = lm.x, lm.y, lm.z
                smoothed_landmarks.append(new_lm)
        else:
            # Apply exponential smoothing
            for i in range(21):
                smoothed_landmarks[i].x = ALPHA * hand[i].x + (1 - ALPHA) * smoothed_landmarks[i].x
                smoothed_landmarks[i].y = ALPHA * hand[i].y + (1 - ALPHA) * smoothed_landmarks[i].y
                smoothed_landmarks[i].z = ALPHA * hand[i].z + (1 - ALPHA) * smoothed_landmarks[i].z

        last_seen_time = current_time

    if smoothed_landmarks and (current_time - last_seen_time) < GRACE_PERIOD:
        points = []
        for i, lm in enumerate(smoothed_landmarks):
            x = int(lm.x * w)
            y = int(lm.y * h)
            points.append((x, y))
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
            
        for start, end in HAND_CONNECTIONS:
            cv2.line(frame, points[start], points[end], (0, 255, 0), 2)

    cv2.imshow("Sudarshan", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()