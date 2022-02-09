# Made by @yyoavv

import cv2 as cv
import time
import mediapipe as mp
import hand_tracking_module as htm

    # Create video object
cap = cv.VideoCapture(0)
# Initialize framerate
previous_time = 0
current_time = 0

detector = handDetector()

while True:
    # Always do to run webcam.
    success, img = cap.read() # Give us frame.
    hands_img = detector.find_hands(img)
    lm_list = detector.get_position(img, draw = False)
    if len(lm_list) != 0:
        print(lm_list[4]) # index of finger
    # Show FPS
    current_time = time.time()
    fps = 1/(current_time - previous_time)
    previous_time = current_time
    cv.putText(img, "FPS: " + str(int(fps)),(20,60), cv.FONT_HERSHEY_SIMPLEX, 1, (225,0,0), 3)

    cv.imshow("Image",hands_img)
    cv.waitKey(1)
