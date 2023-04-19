""" Created by MrBBS """
# 10/29/2021
# -*-encoding:utf-8-*-

import numpy as np
import cv2


class FaceAligner:
    def __init__(self, desiredLeftEye=(0.35, 0.35)):
        self.desiredLeftEye = desiredLeftEye

    def align(self, image, right_eye_point, left_eye_point, face_width, face_height):
        dY = right_eye_point[1] - left_eye_point[1]
        dX = right_eye_point[0] - left_eye_point[0]
        angle = np.degrees(np.arctan2(dY, dX)) - 180

        # desiredRightEyeX = 1.0 - self.desiredLeftEye[0]

        # dist = np.sqrt((dX ** 2) + (dY ** 2))
        # desiredDist = (desiredRightEyeX - self.desiredLeftEye[0])
        # desiredDist *= face_width
        scale = 0.75  # desiredDist / dist

        eyesCenter = ((left_eye_point[0] + right_eye_point[0]) // 2,
                      (left_eye_point[1] + right_eye_point[1]) // 2)

        M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)

        tX = face_width * 0.5
        tY = face_height * self.desiredLeftEye[1]
        M[0, 2] += (tX - eyesCenter[0])
        M[1, 2] += (tY - eyesCenter[1])

        # (w, h) = (face_width - 20, face_height - 30)
        output = cv2.warpAffine(image, M, (face_width, face_height),
                                flags=cv2.INTER_CUBIC)
        return output
