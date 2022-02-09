# Made by @yyoavv
"""
 ---INTRO---
A module with helpers funtion to use Hands by MediaPipe easier.
"""

import cv2 as cv
import mediapipe as mp
import time # To check the frame rate.
import math

class handDetector():

    def __init__(self, mode = False, max_hands = 2, detection_con = 0.5, track_con = 0.5):
    # Functionallity: initizalize MediaPipe Hands object.
    # Input: mode - switch between detection and tracking, max_hands - max number of hands, detection_con/track_con - precentage to switch.
    # Output: None
        self.mode = mode
        self.max_hands = max_hands
        self.detection_con = detection_con
        self.track_con = track_con

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(self.mode, self.max_hands, self.detection_con, self.track_con)
        self.mp_draw = mp.solutions.drawing_utils
        self.tips_ids = [4, 8, 12, 16, 20]

    def find_hands(self, img, draw = True):
    # Functionallity: Draw on img lines and points between hands indices.
    # Input: img - image to find hands on, draw - flag.
    # Output: img with drawing

        img_rgb = cv.cvtColor(img,cv.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)

        if self.results.multi_hand_landmarks != None:
            for hand_lms in self.results.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(img, hand_lms,self.mp_hands.HAND_CONNECTIONS)

        return img

    def get_position(self, img, hand_num = 0, draw = True):
    # Functionallity: Give the user the position of each finger index.
    # Input: img - image to get size, hand_num - number of hand to get informationabout, draw - flag.
    # Output: Prints coordinates

        x_list = []
        y_list = []
        bounding_box = []

        self.landmarks_list = []
        if self.results.multi_hand_landmarks != None:
            hand = self.results.multi_hand_landmarks[hand_num] # Hand to get information about.

            for id,lm in enumerate(hand.landmark):
                # print(id,lm) # Prints coordinates , ratio of the image
                height, width, channel = img.shape
                # Convert to id's coordinates by pixels
                cx, cy = int(lm.x * width), int(lm.y * height)
                x_list.append(cx)
                y_list.append(cy)
                #print("id: " + str(id) + ", cx: " + str(cx)+ ", cy: " + str(cy))
                self.landmarks_list.append([id, cx, cy])
                if draw:
                    cv.circle(img, (cx,cy), 3, (255,0,0), cv.FILLED)

                x_min, x_max = min(x_list), max(x_list)
                y_min, y_max = min(y_list), max(y_list)

            bounding_box = x_min, y_min, x_max, y_max

            # +- 20 to move retangle away from pinky.
            if draw:
                cv.rectangle(img, (bounding_box[0]-20, bounding_box[1]-20), (bounding_box[2]+20, bounding_box[3]+20), (206,189,31), 1)

        return self.landmarks_list, bounding_box

    def find_distance(self, img , id1, id2, draw = True):
    # Functionallity: Mark two id's and the line between,
    # Input: img - image to find coordinates, id's - the part of fingers, draw - flag.
    # Output: return length, img , and line information

        # Create circles around thumb tip and index tip
        id1_x, id1_y = self.landmarks_list[id1][1], self.landmarks_list[id1][2]
        id2_x, id2_y = self.landmarks_list[id2][1], self.landmarks_list[id2][2]
        # Calculate center of line
        center_x, center_y = (id1_x + id2_x)// 2, (id1_y + id2_y)// 2
        # Calculate length of line
        length = math.hypot(id1_x - id2_x , id1_y - id2_y)

        if draw:
            cv.circle(img, (id1_x,id1_y), 5, (255,128,0), cv.FILLED)
            cv.circle(img, (id2_x,id2_y), 5, (255,128,0), cv.FILLED)
            cv.line(img, (id1_x,id1_y), (id2_x,id2_y), (255,128,0), 2)
            cv.circle(img, (center_x,center_y), 2, (153,76,0), cv.FILLED)

        return length, img, [id1_x, id1_y, id2_x, id2_y, center_x, center_y]

    def fingers_up(self):
    # Functionallity: Claculate how if fingers is up or down.
    # Input: None.
    # Output: return sized 5 list, 1 if finger is up, else 0.
        fingers = []

        # Thumb
        if self.landmarks_list[self.tips_ids[0]][1] > self.landmarks_list[self.tips_ids[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # Other fingers
        for id in range(1, 5):
            if self.landmarks_list[self.tips_ids[id]][2] < self.landmarks_list[self.tips_ids[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers
