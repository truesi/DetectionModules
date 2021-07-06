import cv2
import numpy as np
import os


def faceRecognizer(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(
        'C:/Users/askar/PycharmProjects/HandTracking/com/recognizer/trainings/haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.32, minNeighbors=5)

    return faces, gray_image


def labels_for_training_data(directory):
    faces = []
    faceID = []

    for path, subdirnames, filenames in os.walk(directory):
        for filename in filenames:
            if filename.startswith("."):
                print('skipping system file')
                continue

            image_path = os.path.join(path, filename)
            imageID = os.path.basename(path)
            print('image_path', image_path)
            print('id', imageID)
            test_image = cv2.imread(image_path)
            if test_image is None:
                print('image not loaded properly')
                continue
            faces_rect, gray_image = faceRecognizer(test_image)
            if len(faces_rect) != 1:
                continue  # since we are assuming only single person imgage are being fed to classifier
            (x, y, w, h) = faces_rect[0]
            roi_gray = gray_image[y:y + w, x:x + h]
            faces.append(roi_gray)
            faceID.append(int(imageID))
    return faces, faceID


def train_classifier(faces, facesID):
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.train(faces, np.array(facesID))
    return face_recognizer


def draw_rect(test_image, face):
    (x, y, w, h) = face
    cv2.rectangle(test_image, (x, y), (x + w, y + h), (255, 0, 255), 5)


def put_text(test_image, text, x, y):
    cv2.putText(test_image, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 5)
