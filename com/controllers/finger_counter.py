import winsound
import cv2
from com.trackers.hands import HandDetector as hd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
folderPath = "/audio"
myList = os.listdir(BASE_DIR + folderPath)
audioList = []
for wavPath in myList:
    audios = cv2.imread(f'{folderPath}/{wavPath}')
    audioList.append(wavPath)
# winsound.PlaySound('1.wav',winsound.SND_ASYNC)

camera = cv2.VideoCapture(0)
detector = hd.HandDetector(detecionConf=0.75)

tipIds = [4, 8, 12, 16, 20]

finger_pos = {
    'index': [0, 1, 0, 0, 0],
    'fuck': [0, 0, 1, 0, 0],
    'ring': [0, 0, 0, 1, 0],
    'pinky': [0, 0, 0, 0, 1],
    'thumb': [1, 0, 0, 0, 0]
}
values = finger_pos.values()
#print(values)

while True:
    success, frame = camera.read()
    frame = detector.findHands(frame)

    lmList = detector.findPosition(frame, draw=False)

    if len(lmList) != 0:
        fingers = []

        # thumb check
        if lmList[tipIds[0]][1] < lmList[tipIds[0] - 1][1]:
            fingers.append(1)
        # winsound.PlaySound(BASE_DIR + folderPath + '/' + audioList[0], winsound.SND_ASYNC)
        else:
            fingers.append(0)

        # 4 other fingers
        for id in range(1, 5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        totalFingers = fingers.count(1)
        winsound.PlaySound(BASE_DIR + folderPath + '/' + audioList[totalFingers - 1], winsound.SND_APPLICATION)

            # if lmList[fing] == finger_pos[fing]:
            #     print(finger_pos)

        # totalFingersCount = 0
        # if totalFingers == totalFingersCount:
        #     print(totalFingers, totalFingersCount)
        #     winsound.PlaySound(BASE_DIR + folderPath + '/' + audioList[totalFingers - 1], winsound.SND_ASYNC)
        #     totalFingersCount = totalFingers

    cv2.imshow('frame', frame)
    cv2.waitKey(10) == ord('q')
