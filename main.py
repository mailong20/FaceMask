""" Created by MrBBS """
# 2/10/2022
# -*-encoding:utf-8-*-

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from face_detection.face_detection import face_process
from tensorflow.keras.models import load_model
import cv2
import numpy as np


def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
    """Return a sharpened version of the image, using an unsharp mask."""
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened


model = load_model('model.h5')

cap = cv2.VideoCapture(2)
while cap.isOpened():
    ret, frame = cap.read()
    if ret is None:
        break
    for (x1, y1, x2, y2), face in face_process(frame):
        face = unsharp_mask(face)
        print(face.shape)
        face = cv2.resize(cv2.cvtColor(face, cv2.COLOR_BGR2GRAY), (32, 32))
        face = np.expand_dims(np.expand_dims(face, axis=2), axis=0)
        pred = model.predict(face)[0].astype("int8")

        print(pred)
        text = "without_mask"
        if pred[0] == 0:
            text = "with_mask"
            color = (255, 0, 0)
        else:
            color = (0, 255, 0)
        cv2.putText(frame, text, (x1+5, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, color)
        try:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        except:
            pass
    cv2.imshow('cam', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
