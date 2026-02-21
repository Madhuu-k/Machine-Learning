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

gesture_frames = 0
REQUIRED_FRAMES = 4

cap = cv2.VideoCapture(0)

smoothed_landmarks = None
last_seen_time = 0

anchor_smooth_x = None
anchor_smooth_y = None
ANCHOR_ALPHA = 0.5

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

    # if smoothed_landmarks and (current_time - last_seen_time) < GRACE_PERIOD:
    # if smoothed_landmarks:
    #     points = []
    #     for i, lm in enumerate(smoothed_landmarks):
    #         x = int(lm.x * w)
    #         y = int(lm.y * h)
    #         points.append((x, y))
    #         cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
            
    #     for start, end in HAND_CONNECTIONS:
    #         cv2.line(frame, points[start], points[end], (0, 255, 0), 2)
            
    gesture_valid = False
    
    if result.hand_landmarks and smoothed_landmarks:
        tip = smoothed_landmarks[8]
        pip = smoothed_landmarks[6]
        mcp = smoothed_landmarks[5]
        
        index_up = tip.y < pip.y < mcp.y
        
        middle_down = smoothed_landmarks[12].y > smoothed_landmarks[9].y
        ring_down = smoothed_landmarks[16].y > smoothed_landmarks[13].y
        pinky_down = smoothed_landmarks[20].y > smoothed_landmarks[17].y
        
        if index_up and middle_down and ring_down and pinky_down:
            gesture_valid = True
    
    if gesture_valid:
        gesture_frames += 1
    else:
        gesture_frames = 0
        
    if not result.hand_landmarks:
        gesture_frames = 0
        
    if result.hand_landmarks and  gesture_frames >= REQUIRED_FRAMES:
        base = smoothed_landmarks[5]
        tip = smoothed_landmarks[8]
        
        dx = tip.x - base.x
        dy = tip.y - base.y
        
        length = np.sqrt(dx*dx + dy*dy)
        if length > 0:
            dx /= length
            dy /= length
            
        depth_factor = np.clip(-tip.z, 0.02, 0.1)
        ANCHOR_DISTANCE = 0.03 + depth_factor
        
        anchor_x = tip.x + dx * ANCHOR_DISTANCE
        anchor_y = tip.y + dy * ANCHOR_DISTANCE 
        
        if anchor_smooth_x is None:
            anchor_smooth_x = anchor_x
            anchor_smooth_y = anchor_y
        
        anchor_smooth_x = ANCHOR_ALPHA * anchor_x + (1 - ANCHOR_ALPHA) * anchor_smooth_x
        anchor_smooth_y = ANCHOR_ALPHA * anchor_y + (1 - ANCHOR_ALPHA) * anchor_smooth_y
        
        x = int(anchor_x * w)
        y = int(anchor_y * h)
        
        size = 40 # Square Size
        x = max(size // 2, min(w - size // 2, x))
        y = max(size // 2, min(h - size // 2, y))
        cv2.rectangle(frame,
            (x - size // 2, y - size // 2),
            (x + size // 2, y + size // 2),
            (0, 225, 0),
            2
        )
        cv2.putText(frame, "Gesture Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow("Sudarshan", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()