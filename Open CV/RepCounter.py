import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import math

RIGHT_SHOULDER = 12
RIGHT_ELBOW = 14
RIGHT_WRIST = 16
LEFT_SHOULDER = 11
LEFT_ELBOW = 13
LEFT_WRIST = 15

rep = 0
stage = None 


def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    ba = a - b
    bc = c - b
    
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    
    return np.degrees(angle)

# Path to model
MODEL_PATH = r"D:\Machine Learning\Open CV\pose_landmarker_lite.task"

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)

options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=False
)

pose_landmarker = vision.PoseLandmarker.create_from_options(options)

# Video Capture
capture = cv2.VideoCapture(0)

if not capture.isOpened():
    print("Camera is not opened")
    exit()
    
while True:
    ret, frame = capture.read()
    
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    
    h, w, _ = frame.shape
    # Change color from BGR -> RBG
    rbg = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Wrap the image in a MediaPipe Image object
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rbg
    )
    # Load the data to Pose estimator
    results = pose_landmarker.detect(mp_image)
    
    if results.pose_landmarks:
        right_shoulder = results.pose_landmarks[0][RIGHT_SHOULDER]
        right_elbow = results.pose_landmarks[0][RIGHT_ELBOW]
        right_wrist = results.pose_landmarks[0][RIGHT_WRIST]
        
        left_shoulder = results.pose_landmarks[0][LEFT_SHOULDER]
        left_elbow = results.pose_landmarks[0][LEFT_ELBOW]
        left_wrist = results.pose_landmarks[0][LEFT_WRIST]
        
        right_shoulder_point = (int(right_shoulder.x * w), int(right_shoulder.y * h))
        right_elbow_point = (int(right_elbow.x * w), int(right_elbow.y * h))
        right_wrist_point = (int(right_wrist.x * w), int(right_wrist.y * h))
        
        left_shoulder_point = (int(left_shoulder.x * w), int(left_shoulder.y * h))
        left_elbow_point = (int(left_elbow.x * w), int(left_elbow.y * h))
        left_wrist_point = (int(left_wrist.x * w), int(left_wrist.y * h))
        
        right_arm_angle = calculate_angle(
            right_shoulder_point,
            right_elbow_point,
            right_wrist_point
        )
        
        left_arm_angle = calculate_angle(
            left_shoulder_point,
            left_elbow_point,
            left_wrist_point
        )
        
        # Rep Counter Logic for both hands
        if right_arm_angle > 160 or left_arm_angle > 160:
            stage = "down"
            
        if right_arm_angle < 30 and stage == "down" and left_arm_angle < 30 and stage == "down":
            stage = "up"
            rep += 1
            
        cv2.line(frame, right_shoulder_point, right_elbow_point, (0, 255, 0), 3)
        cv2.line(frame, right_elbow_point, right_wrist_point, (0, 255, 0), 3)
        
        cv2.line(frame, left_shoulder_point, left_elbow_point, (0, 255, 0), 3)
        cv2.line(frame, left_elbow_point, left_wrist_point, (0, 255, 0), 3)
        
        # Display Rep Count
        cv2.putText(
            frame,
            f"Reps: {rep}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.7,
            (255, 0, 0),
            2
        )
        
        # Display Angles
        cv2.putText(
            frame,
            f"Right Arm Angle: {int(right_arm_angle)}",
            (right_elbow_point[0] + 10, right_elbow_point[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
        
        cv2.putText(
            frame,
            f"Left Arm Angle: {int(left_arm_angle)}",
            (left_elbow_point[0] + 10, left_elbow_point[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
        
        cv2.circle(frame, right_shoulder_point, 5, (0, 0, 255), -1)
        cv2.circle(frame, right_elbow_point, 5, (0, 0, 255), -1)
        cv2.circle(frame, right_wrist_point, 5, (0, 0, 255), -1)
        
        cv2.circle(frame, left_shoulder_point, 5, (0, 0, 255), -1)
        cv2.circle(frame, left_elbow_point, 5, (0, 0, 255), -1)
        cv2.circle(frame, left_wrist_point, 5, (0, 0, 255), -1)
        
    cv2.imshow("Pose Estimation with Rep Counter", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
capture.release()
cv2.destroyAllWindows()
            
        
    
    