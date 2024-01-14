import imutils
import cv2
import time

cam = cv2.VideoCapture(0)

if not cam.isOpened():
    print("Error: Could not open camera.")
    exit()

time.sleep(1)

firstFrame = None
area = 720

while True:
    ret, frame = cam.read()

    if not ret:
        print("Error: Failed to grab frame")
        break

    text = "Normal"
    frame = imutils.resize(frame, width=500)

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gaussian_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0)

    if firstFrame is None:
        firstFrame = gaussian_frame
        continue

    frame_diff = cv2.absdiff(firstFrame, gaussian_frame)

    thresh_frame = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)[1]

    thresh_frame = cv2.dilate(thresh_frame, None, iterations=2)

    cnts = cv2.findContours(thresh_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cnts = imutils.grab_contours(cnts)
    for c in cnts:
        if cv2.contourArea(c) < area:
            continue
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        text = "Moving object detected"

    print(text)
    cv2.putText(frame, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.imshow("Camera Feed", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()
