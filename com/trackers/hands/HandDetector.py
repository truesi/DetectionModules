import mediapipe as mp
import cv2


class HandDetector:
    def __init__(self, mode=False, maxHands=2, detecionConf=0.5, trackConf=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionConf = detecionConf
        self.trackConf = trackConf

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.detectionConf, self.trackConf)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.result = self.hands.process(imgRGB)
        # print(result.multi_hand_landmarks)

        if self.result.multi_hand_landmarks:
            for handLandmarks in self.result.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLandmarks, self.mpHands.HAND_CONNECTIONS)

        return img

    def findPosition(self, img, handNumber=0, draw=True):

        landMarkList = []
        if self.result.multi_hand_landmarks:
            myHand = self.result.multi_hand_landmarks[handNumber]

            for id, lm in enumerate(myHand.landmark):
                # print(id, lm)
                height, weight, channels = img.shape
                # because the position is not a pixel and we want pixel position not a decimal value we multiply
                cx, cy = int(lm.x * weight), int(lm.y * height)
                landMarkList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
                # print(cx, cy)
                # to track every node position of landmark
                # if id == 4:
                #     cv2.circle(img, (cx,cy), 5 , (255,0,255), cv2.FILLED)

        return landMarkList
