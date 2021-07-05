import cv2
import time
from com.trackers.hands import HandDetector as hd

previousTime = 0
currentTime = 0
# to capture video
cap = cv2.VideoCapture(0)
detector = hd.HandDetector()

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    landMarkList = detector.findPosition(img)
    if len(landMarkList) != 0:
        print(landMarkList[4])

    # to show fps
    currentTime = time.time()
    fps = 1 / (currentTime - previousTime)
    previousTime = currentTime

    # to print video
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
    cv2.imshow("Image", img)
    cv2.waitKey(1)