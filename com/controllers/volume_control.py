import cv2
import numpy as np
from com.trackers.hands import HandDetector as hdm
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

widthCam, heightCam = 1280, 720

camera = cv2.VideoCapture(0)
camera.set(3, widthCam)
camera.set(4, heightCam)

handDetector = hdm.HandDetector(detecionConf=0.8, trackConf=0.8 )

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
volumeRange = volume.GetVolumeRange()
minVolume = volumeRange[0]
maxVolume = volumeRange[1]


while True:
    success, frame = camera.read()

    frame = handDetector.findHands(frame)
    lmList = handDetector.findPosition(frame, draw=False)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if len(lmList) != 0:
        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
        cv2.circle(frame, (cx, cy), 3, (255, 0, 255), 3)

        length = math.hypot(x2 - x1, y2 - y1)

        vol= np.interp(length, [50,300],[minVolume, maxVolume])
        volume.SetMasterVolumeLevel(vol, None)
    cv2.imshow("Frame", frame)
    cv2.waitKey(10) == ord('q')
