""" Created by MrBBS """
# 2/10/2022
# -*-encoding:utf-8-*-

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Input
from tensorflow.keras.models import Model
import tensorflow as tf
import cv2
import numpy as np


def build_model():
    input_shape = Input((128, 128, 1), name='Input')
    x = Conv2D(32, (3, 3),strides=1, padding='same', activation='relu', kernel_initializer='he_normal', name='con1')(input_shape)
    x = MaxPool2D(name='max1')(x)
    x = Conv2D(32, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal', name='con2')(x)
    x = MaxPool2D(name='max2')(x)
    x = Flatten()(x)
    x = Dense(64, activation='relu', kernel_initializer='he_normal', name='dense1')(x)
    x = Dense(2, activation='softmax', kernel_initializer='he_normal', name='softmax')(x)
    model = Model(input_shape, x)
    return model

model = build_model()

x = cv2.imread(r'D:\BTS\train\traindata\images - 2022-02-16T164108.638.jfif', cv2.IMREAD_GRAYSCALE)
x = cv2.resize(x, (128, 128))
x = np.expand_dims(np.expand_dims(x, axis=2), axis=0)

successive_outputs = [layer.output for layer in model.layers[1:]]
visualization_model = tf.keras.models.Model(inputs=model.input, outputs=successive_outputs)
successive_feature_maps = visualization_model.predict(x)
layer_names = [layer.name for layer in model.layers]
for layer_name, feature_map in zip(layer_names, successive_feature_maps):
    print(feature_map.shape)
    if len(feature_map.shape) == 4:
        imgs = []
        n_features = feature_map.shape[-1]
        for i in range(n_features):
            x = feature_map[0, :, :, i]
            x -= x.mean()
            x /= x.std()
            x *= 63
            x += 128
            x = np.clip(x, 0, 255).astype('uint8')
            imgs.append(x)
        cv2.imshow(layer_name, np.hstack(imgs))
        cv2.waitKey()