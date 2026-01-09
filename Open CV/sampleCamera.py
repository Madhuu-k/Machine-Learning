import cv2

capture = cv2.VideoCapture(0)

if not capture.isOpened():
    print("Error: Could not open camera.")
    exit()
    
while True:
    ret, frame = capture.read()
    
    if not ret:
        print("Error: Could not read frame.")
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Camera Feed", gray)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
capture.release()
capture.destrroyAllWindows()