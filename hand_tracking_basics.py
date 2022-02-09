# Made by @yyoavv
"""
 ---INTRO---
 Hand tracking uses two main modules at the backend.
1. Palm Detection - basically works on complete image and and provides a crooped image of the hand.
2. Hand Landmarks - finds 21 different landmarks on this cropped image of the hand.
"""

import cv2 as cv
import mediapipe as mp
import time # To check the frame rate.

# Create video object
cap = cv.VideoCapture(0)

mp_hands = mp.solutions.hands # Create object from class Hands, formality to do before start using Hands model.
hands = mp_hands.Hands() # static_image_mode - False to track and detect, True to detection only (slow), max number of hands = 2.
mp_draw = mp.solutions.drawing_utils

# Initialize framerate
previous_time = 0
current_time = 0

while True:
# Always do to run webcam.
    success, img = cap.read() # Give us frame.
    img_rgb = cv.cvtColor(img,cv.COLOR_BGR2RGB) # This class (Hands) uses only uses rgb images.
    results = hands.process(img_rgb)
    #print(results.multi_hand_landmarks) # Check if somethings is detected or not

    if results.multi_hand_landmarks != None:
        for hand_lms in results.multi_hand_landmarks: # Extract information for each hand.
            for id,lm in enumerate(hand_lms.landmark): # id relate to the exact index number of finger.
                #print(id,lm) # Prints coordinates , ratio of the image
                height, width, channel = img.shape
                cx, cy = int(lm.x * width), int(lm.y * height)# Find the position in pixels
                print("id: " + str(id) + ", cx: " + str(cx)+ ", cy: " + str(cy))

                if id == 4:
                    cv.circle(img, (cx,cy), 10, (255,0,255), cv.FILLED)

            mp_draw.draw_landmarks(img, hand_lms,mp_hands.HAND_CONNECTIONS) # Don't draw on rgb image because we don't display RGB image.

    current_time = time.time()
    fps = 1/(current_time - previous_time)
    previous_time = current_time

    cv.putText(img, "FTS: " + str(int(fps)),(20,60), cv.FONT_HERSHEY_SIMPLEX, 1, (225,0,0), 3)

    cv.imshow("Image",img)
    cv.waitKey(1) # The function waitKey waits for a key event infinitely and the delay is in milliseconds. waitKey(0) means forever.
