import cv2
import time
from com.trackers.hands import HandDetector as hd
from com.trackers.body import BodyDetector as bd
from com.recognizer.face import FaceRecognizerModule as fr

# for fps
previousTime = 0
currentTime = 0
# to capture video
widthCam, heightCam = 640, 640
camera1 = cv2.VideoCapture(0)
camera1.set(3, widthCam)
camera1.set(4, heightCam)

handDetector = hd.HandDetector(detecionConf=0.5)
bodyDetector = bd.BodyDetector()
faceRecognizer = fr.FaceRecognizerModule()

while True:
    success, frame = camera1.read()
    success, frame2 = camera1.read()
    frame = handDetector.findHands(frame)
    frame = bodyDetector.findBodyMovement(frame, frame2)
    frame = faceRecognizer.recognizeFace(frame)

    # landMarkList = handDetector.findPosition(frame)
    # if len(landMarkList) != 0:
    #   print(landMarkList[4])

    # to show fps
    currentTime = time.time()
    fps = 1 / (currentTime - previousTime)
    previousTime = currentTime

    # to print video
    cv2.putText(frame, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 3)
    cv2.imshow("Image", frame)
    cv2.waitKey(1)
