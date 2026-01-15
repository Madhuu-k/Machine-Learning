import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import math

MODEL_PATH = r"D:\Machine Learning\Open CV\pose_landmarker_lite.task"

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (
        np.linalg.norm(ba) * np.linalg.norm(bc)
    )

    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)

    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

# Pose model
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=False
)
pose_landmarker = vision.PoseLandmarker.create_from_options(options)

capture = cv2.VideoCapture(0)
if not capture.isOpened():
    print("Cannot open camera")
    exit()

RIGHT_SHOULDER = 12
RIGHT_ELBOW = 14
RIGHT_WRIST = 16

LEFT_SHOULDER = 11
LEFT_ELBOW = 13
LEFT_WRIST = 15

rep = 0
stage = None

while True:
    ret, frame = capture.read()
    if not ret:
        break

    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    results = pose_landmarker.detect(mp_image)

    if results.pose_landmarks:
        shoulder = results.pose_landmarks[0][RIGHT_SHOULDER]
        elbow = results.pose_landmarks[0][RIGHT_ELBOW]
        wrist = results.pose_landmarks[0][RIGHT_WRIST]
        
        l_shoulder = results.pose_landmarks[0][LEFT_SHOULDER]
        l_elbow = results.pose_landmarks[0][LEFT_ELBOW]
        l_wrist = results.pose_landmarks[0][LEFT_WRIST]

        shoulder_point = (int(shoulder.x * w), int(shoulder.y * h))
        l_shoulder_point = (int(l_shoulder.x * w), int(l_shoulder.y * h))
        
        elbow_point = (int(elbow.x * w), int(elbow.y * h))
        l_elbow_point = (int(l_elbow.x * w), int(l_elbow.y * h))
        
        wrist_point = (int(wrist.x * w), int(wrist.y * h))
        l_wrist_point = (int(l_wrist.x * w), int(l_wrist.y * h))
        
        angle = calculate_angle(shoulder_point, elbow_point, wrist_point)
        l_angle = calculate_angle(
            l_shoulder_point, l_elbow_point, l_wrist_point
        )
        
        if angle > 160:
            stage = "up"
        
        if angle < 70 and stage == "up":
            stage = "down"
            rep += 1
        
        cv2.putText(
            frame,
            f"Reps: {rep}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (255, 0, 0),
            2
        )

        cv2.putText(
            frame,
            f" Right Elbow Angle: {int(angle)}",
            (elbow_point[0] + 10, elbow_point[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
        
        cv2.putText(
            frame,
            f"Left Elbow Angle: {int(l_angle)}",
            (l_elbow_point[0] + 10, l_elbow_point[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.line(frame, shoulder_point, elbow_point, (255, 0, 0), 2)
        cv2.line(frame, elbow_point, wrist_point, (255, 0, 0), 2)
        cv2.line(frame, l_shoulder_point, l_elbow_point, (255, 0, 0), 2)
        cv2.line(frame, l_elbow_point, l_wrist_point, (255, 0, 0), 2)

        cv2.circle(frame, shoulder_point, 5, (0, 0, 255), -1)
        cv2.circle(frame, elbow_point, 5, (0, 0, 255), -1)
        cv2.circle(frame, wrist_point, 5, (0, 0, 255), -1)
        
        cv2.circle(frame, l_shoulder_point, 5, (0, 0, 255), -1)
        cv2.circle(frame, l_elbow_point, 5, (0, 0, 255), -1)
        cv2.circle(frame, l_wrist_point, 5, (0, 0, 255), -1)

    cv2.imshow("Pose Estimation with Angles", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()


# import cv2
# import numpy as np
# import mediapipe as mp
# from mediapipe.tasks import python
# from mediapipe.tasks.python import vision
# import math

# # Path to model
# MODEL_PATH = r"D:\Machine Learning\Open CV\pose_landmarker_lite.task"

# def calculate_angle(a, b, c):
#     a = np.array(a)
#     b = np.array(b)
#     c = np.array(c)
    
#     ba = a - b
#     bc = c - b
    
#     cosine_angle = np.dot(ba, bc) / ( np.linalg.norm(ba) * np.linalg.norm(bc) )
#     angle = np.arccos(cosine_angle)
#     return np.degrees(angle)

# # Create PoseLandmarker
# base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
# options = vision.PoseLandmarkerOptions(
#     base_options=base_options,
#     output_segmentation_masks=False
# )
# pose_landmarker = vision.PoseLandmarker.create_from_options(options)

# capture = cv2.VideoCapture(0)
# if not capture.isOpened():
#     print("Cannot Open Camera")
#     exit()

# RIGHT_SHOULDER = 12
# RIGHT_ELBOW = 14
# RIGHT_WRIST = 16

# while True:
#     ret, frame = capture.read()
#     if not ret:
#         break;
    
#     rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
#     mp_image = mp.Image(
#         image_format=mp.ImageFormat.SRGB,
#         data=rgb
#     )
    
#     results = pose_landmarker.detect(mp_image)
    
#     if results.pose_landmarks:
#         for landmark in results.pose_landmarks[0]:
#             h, w, _ = frame.shape
            
#             shoulder = results.pose_landmarks[0][RIGHT_SHOULDER]
#             elbow = results.pose_landmarks[0][RIGHT_ELBOW]
#             wrist = results.pose_landmarks[0][RIGHT_WRIST]
            
#             shoulder_point = (int(shoulder.x * w), int(shoulder.y * h))
#             elbow_point = (int(elbow.x * w), int(elbow.y * h))
#             wrist_point = (int(wrist.x * w), int(wrist.y * h))
            
#             angle = calculate_angle(shoulder_point, elbow_point, wrist_point)
            
#             cv2.putText(
#                 frame,
#                 f"Elbow Angle: {int(angle)}",
#                 (elbow_point[0] + 10, elbow_point[1] - 10),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 0.7,
#                 (0, 255, 0), 
#                 2
#             )
            
#             cv2.line(frame, shoulder_point, elbow_point, (255, 0, 0), 2)
#             cv2.line(frame, elbow_point, wrist_point, (255, 0 , 0), 2)
            
#             cv2.circle(frame, shoulder_point, 5, (0, 0, 255), -1)
#             cv2.circle(frame, elbow_point, 5, (0, 0, 255), -1)
#             cv2.circle(frame, wrist_point, 5, (0, 0, 255), -1)
            
#     cv2.imshow("Pose Estimation with Angles", frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
    
# capture.release()
# cv2.destroyAllWindows()