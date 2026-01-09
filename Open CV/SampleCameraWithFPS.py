import cv2
import time

prev_time = 0
capture = cv2.VideoCapture(0)

if not capture.isOpened():
    print("camera is not opened")
    exit(0)
    
while True:
    ret, frame = capture.read()
    
    if not ret:
        print("Failed to grab frame")
        break
    
    curr_time = time.time()
    if prev_time != 0:
        fps = 1 / (curr_time - prev_time)
    else:
        fps = 0
    prev_time = curr_time
    
    cv2.putText(
        frame,
        f"FPS: {int(fps)}",
        (30, 90),
        cv2.FONT_HERSHEY_SIMPLEX
        , 1,
        (0, 255, 0), 2
    )

    # Draw Circle
    cv2.circle(frame, (50, 50), 30, (255, 0, 0), 2)

    # Show Frame with FPS
    cv2.imshow("Camera with FPS", frame)
    
    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
capture.release()
cv2.destroyAllWindows()
    
    