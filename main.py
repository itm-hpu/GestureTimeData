import cvzone
import time
import numpy as np
import mediapipe as mp
import cv2
import math
from cvzone.HandTrackingModule import HandDetector

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

cap = cv2.VideoCapture(1)
detector = HandDetector(detectionCon=0.7, maxHands=2)

pTime = 0

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img,flipType=True) #With Draw
    #hands = detector.findHands(img, draw=False) #No draw
    #print(len(hands)) # number of hands

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    if hands:
        # Hand 1
        hand1 = hands[0]
        lmList1 = hand1["lmList"] # List of 21 Landmarks points
        bbox1 = hand1["bbox"] # Bouding Box info x,y,w,h
        centerPoint1 = hand1["center"] # Center of the hand cx, cy
        handType1 = hand1["type"] # hand type Left of Right

        #print(lmList1[4], lmList1[8]) #thumb, index
        x1,y1=lmList1[4][0], lmList1[4][1]
        x2,y2=lmList1[8][0], lmList1[8][1]
        cx,cy = (x1+x2)//2, (y1+y2)//2

        cv2.circle(img, (x1,y1), 15, (255,255,255), cv2.FILLED)
        cv2.circle(img, (x2,y2), 15, (255,255,255), cv2.FILLED)
        cv2.line(img, (x1,y1),(x2,y2),(255,255,255),3)
        cv2.circle(img, (cx,cy), 15, (255,255,255), cv2.FILLED)

        length = math.hypot(x2-x1,y2-y1)
        print(length)
        cv2.putText(img, f'{int(length)}', (cx, cy), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)

        if length <50:
            cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED)

        #print(bbox1)
        #print(centerPoint1) #cv to Circle?
        #print(handType1)
        fingers1=detector.fingersUp(hand1)
        #length, info, img = detector.findDistance(lmList1[8], lmList1[12], img) #with draw - this is not working
        #length, info = detector.findDistance(lmList1[8], lmList1[12]) #no draw - this is not working

        f00, f01, f02, f03, f04 = fingers1[0], fingers1[1], fingers1[2], fingers1[3], fingers1[4]
        count0 = f00 + f01 + f02 + f03 + f04
        if f00 == 1:
            cv2.putText(img, '1', (lmList1[4][0], lmList1[4][1]), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)
        if f01 == 1:
            cv2.putText(img, '1', (lmList1[8][0], lmList1[8][1]), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)
        if f02 == 1:
            cv2.putText(img, '1', (lmList1[12][0], lmList1[12][1]), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)
        if f03 == 1:
            cv2.putText(img, '1', (lmList1[16][0], lmList1[16][1]), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)
        if f04 == 1:
            cv2.putText(img, '1', (lmList1[20][0], lmList1[20][1]), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)
        cv2.putText(img, f'{int(count0)}', (40,100), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)

        if len(hands)==2:
            hand2 = hands[1]
            lmList2 = hand2["lmList"]  # List of 21 Landmarks points
            bbox2 = hand2["bbox"]  # Bouding Box info x,y,w,h
            centerPoint2 = hand2["center"]  # Center of the hand cx, cy
            handType2 = hand2["type"]  # hand type Left of Right

            fingers2 = detector.fingersUp(hand2)
            #print(fingers1, fingers2)
            #length, info, img = detector.findDistance(centerPoint1, centerPoint2, img) #with draw - this is not working

            f10, f11, f12, f13, f14 = fingers2[0], fingers2[1], fingers2[2], fingers2[3], fingers2[4]
            count1 = f10 + f11 + f12 + f13 + f14
            if f10 == 1:
                cv2.putText(img, '1', (lmList2[4][0], lmList2[4][1]), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)
            if f11 == 1:
                cv2.putText(img, '1', (lmList2[8][0], lmList2[8][1]), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)
            if f12 == 1:
                cv2.putText(img, '1', (lmList2[12][0], lmList2[12][1]), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)
            if f13 == 1:
                cv2.putText(img, '1', (lmList2[16][0], lmList2[16][1]), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)
            if f14 == 1:
                cv2.putText(img, '1', (lmList2[20][0], lmList2[20][1]), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)
            cv2.putText(img, f'{int(count1)}', (40, 150), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)
            cv2.putText(img, f'{int(count0+count1)}', (40, 200), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)

            #print(centerPoint1, centerPoint2)
            #print(lmList1[8], lmList2[8])
            x1, y1 = lmList2[4][0], lmList2[4][1]
            x2, y2 = lmList2[8][0], lmList2[8][1]
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            cv2.circle(img, (x1, y1), 15, (255, 255, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (255, 255, 255), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
            cv2.circle(img, (cx, cy), 15, (255, 255, 255), cv2.FILLED)

            length = math.hypot(x2 - x1, y2 - y1)
            print(length)
            cv2.putText(img, f'{int(length)}', (cx, cy), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)

            if length < 50:
                cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED)

    cv2.putText(img, f'FPS:{int(fps)}', (40,50), cv2.FONT_HERSHEY_COMPLEX, 1,(255,0,0),3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)


#https://www.youtube.com/watch?v=3xfOa4yeOb0&ab_channel=Murtaza%27sWorkshop-RoboticsandAI
#https://google.github.io/mediapipe/solutions/hands#python-solution-api
