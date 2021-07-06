import cv2
import pickle

face_cascade = cv2.CascadeClassifier(
    'C:/Users/askar/PycharmProjects/HandTracking/com/recognizer/trainings/haarcascade_frontalface_alt2.xml')
# face_cascade = cv2.CascadeClassifier('C:/Users/askar/PycharmProjects/HandTracking/com/recognizer/trainings
# /haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("recognizer/face/trainer.yml")
labels = {}

with open("recognizer/face/labels.pickle", 'rb') as file:
    labels = pickle.load(file)
    labels = {value: key for key, value in labels.items()}



class FaceRecognizerModule:
    def __init__(self):
        self

    def recognizeFace(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.32, minNeighbors=5)
        for (x, y, w, h) in faces:
            # print(x, y, w, h)
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = frame[y:y + h, x:x + w]

            id_, conf = recognizer.predict(roi_gray)
            if 45 <= conf <= 85:
                cv2.putText(frame, labels[id_], (x, y), cv2.FONT_HERSHEY_PLAIN, 5, (255, 0, 255), 2)

            # image_for_new_training = "6.png"
            # cv2.imwrite(image_for_new_training, roi_color)

            #cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 3)

        return frame
