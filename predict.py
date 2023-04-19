""" Created by MrBBS """
# 2/24/2022
# -*-encoding:utf-8-*-

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# from model import build_model
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import time

model = load_model('D:/model.10-0.16.h5')

path = r'D:/cat_dog_dataset/test/dog/'

cout = 0
arrname = os.listdir(path)
#0.844 0.99
print("cats", len(arrname))
img = []
for i in range(len(arrname)):

    image = cv2.imread(path+arrname[i])#
    img = cv2.resize(image.copy(), (224, 224))
    img = np.expand_dims(img.copy(), axis=0)

    label = model.predict(img)[0].astype("int8")

    if label[1] == 1:
        cout += 1
    #     cv2.putText(image, str('dogs'), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 255)
    # cv2.imshow('vv', image)
    # cv2.waitKey(0)
print("accuracy: ", cout/len(arrname))

# print(img.shape)
#
# print()
# print(time.time() - start)
