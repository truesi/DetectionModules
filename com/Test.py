import cv2
import time
from com.trackers.hands import HandDetector as hd
from com.trackers.body import BodyDetector as bd

previousTime = 0
currentTime = 0
# to capture video
camera1 = cv2.VideoCapture(0)
handDetector = hd.HandDetector(detecionConf=0.7)
bodyDetector = bd.BodyDetector()

while True:
    success, frame = camera1.read()
    success, frame2 = camera1.read()
    frame = handDetector.findHands(frame)
    frame = bodyDetector.findBodyMovement(frame, frame2)

    # landMarkList = handDetector.findPosition(frame)
    # if len(landMarkList) != 0:
    #   print(landMarkList[4])

    # to show fps
    currentTime = time.time()
    fps = 1 / (currentTime - previousTime)
    previousTime = currentTime

    # to print video
    cv2.putText(frame, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
    cv2.imshow("Image", frame)
    cv2.waitKey(1)
