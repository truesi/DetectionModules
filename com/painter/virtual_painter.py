import cv2
import numpy as np
import os
from com.trackers.hands import HandDetector as hd

folderPath = "background"
myList = os.listdir(folderPath)
overlayList = []

for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)

background = overlayList[0]  # header
drawColor = (255, 0, 255)

camera = cv2.VideoCapture(0)
camera.set(3, 1920)
camera.set(4, 1080)

detector = hd.HandDetector(detecionConf=0.9, trackConf=0.9)

brushThickness = 15
eraserThickness = 75

xp, yp = 0, 0

imgCanvas = np.zeros((720, 1280, 3), np.uint8)

while True:
    success, img = camera.read()

    ######################
    img = cv2.flip(img, 1)
    ######################

    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        # find position of whole pointy finger
        x1, y1 = lmList[8][1:]
        # find position of whole fuck finger
        x2, y2 = lmList[12][1:]
        fingers = detector.fingersUp()

        ## selection mode with two fingers up
        if fingers[1] and fingers[2]:
            xp, yp = 0, 0
            # checking for the click
            if y1 < 125:
                if 250 < x1 < 450:
                    background = overlayList[0]
                    drawColor = (255, 0, 255)
                elif 550 < x1 < 750:
                    background = overlayList[1]
                    drawColor = (255, 0, 0)
                elif 800 < x1 < 950:
                    background = overlayList[2]
                    drawColor = (0, 255, 0)
                elif 1050 < x1 < 1200:
                    background = overlayList[3]
                    drawColor = (0, 0, 0)
            cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv2.FILLED)

        ## paint mode
        if fingers[1] and fingers[2] == False:
            cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)
            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            if drawColor == (0, 0, 0):
                cv2.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)
            else:
                cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)

            xp, yp = x1, y1

    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInverse = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInverse = cv2.cvtColor(imgInverse, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInverse)
    img = cv2.bitwise_or(img, imgCanvas)

    # setting background
    img[0:125, 0:1280] = background
    # img = cv2.addWeighted(img, 0.5, imgCanvas, 0.5, 0)
    cv2.imshow("Image", img)
    cv2.waitKey(1)
