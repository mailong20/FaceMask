""" Created by MrBBS """
# 1/12/2022
# -*-encoding:utf-8-*-


import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import onnxruntime
from .face_aligner import FaceAligner

model_face_detect = load_model('face_detection/face_detection.h5')
options = onnxruntime.SessionOptions()
options.inter_op_num_threads = 1
options.intra_op_num_threads = 1  # min(1,4)
options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
options.log_severity_level = 3

session = onnxruntime.InferenceSession('face_detection/landmark.onnx', sess_options=options)
input_name = session.get_inputs()[0].name
fa = FaceAligner()
left_eye_id = [37, 40]
right_eye_id = [44, 47]
logit_factor = 16.

mean = np.float32(np.array([0.485, 0.456, 0.406]))
std = np.float32(np.array([0.229, 0.224, 0.225]))
mean /= std
std *= 255.0

mean = - mean
std = 1.0 / std
mean_32 = np.tile(mean, [32, 32, 1])
std_32 = np.tile(std, [32, 32, 1])
mean_224 = np.tile(mean, [224, 224, 1])
std_224 = np.tile(std, [224, 224, 1])
m_res = 224.
mean_res = mean_224
std_res = std_224
res_i = int(m_res)

out_res = 27.
out_res_i = int(out_res) + 1


def create_letterbox_image(frame, dim):
    h, w = frame.shape[0:2]
    scale = min(dim / h, dim / w)
    nh, nw = int(scale * h), int(scale * w)
    resized = cv2.resize(frame, (nw, nh))
    new_image = np.zeros((dim, dim, 3), np.uint8)
    new_image.fill(256)
    dx = (dim - nw) // 2
    dy = (dim - nh) // 2
    new_image[dy:dy + nh, dx:dx + nw, :] = resized
    return new_image


def xywh_to_tlbr(boxes, y_first=False):
    final_boxes = boxes.copy()
    if not y_first:
        final_boxes[:, 0:2] = np.clip(boxes[:, 0:2] - (boxes[:, 2:4] / 2), 0,
                                      None)  # clipping at 0 since image dim starts at 0
        final_boxes[:, 2:4] = boxes[:, 0:2] + (boxes[:, 2:4] / 2)
    else:
        final_boxes[:, 0:2] = np.clip(boxes[:, [1, 0]] - (boxes[:, [3, 2]] / 2), 0, None)
        final_boxes[:, 2:4] = boxes[:, [1, 0]] + (boxes[:, [3, 2]] / 2)
    return final_boxes


def convert_to_orig_points(results, orig_dim, letter_dim):
    if results.ndim == 1:
        np.expand_dims(results, 0)
    inter_scale = min(letter_dim / orig_dim[0], letter_dim / orig_dim[1])
    inter_h, inter_w = int(inter_scale * orig_dim[0]), int(inter_scale * orig_dim[1])
    offset_x, offset_y = (letter_dim - inter_w) / 2.0 / letter_dim, (letter_dim - inter_h) / 2.0 / letter_dim
    scale_x, scale_y = letter_dim / inter_w, letter_dim / inter_h
    results[:, 0:2] = (results[:, 0:2] - [offset_x, offset_y]) * [scale_x, scale_y]
    results[:, 2:4] = results[:, 2:4] * [scale_x, scale_y]
    results[:, 4:16:2] = (results[:, 4:16:2] - offset_x) * (scale_x * 2)
    results[:, 5:17:2] = (results[:, 5:17:2] - offset_y) * (scale_y * 2)
    # converting from 0-1 range to (orign_dim) range
    results[:, 0:16:2] *= orig_dim[1]
    results[:, 1:17:2] *= orig_dim[0]

    return results.astype(np.int32)


def process_detections(results, orig_dim, max_boxes=5, score_threshold=0.5, iou_threshold=0.5):
    box_tlbr = xywh_to_tlbr(results[:, 0:4], y_first=True)
    out_boxes = tf.image.non_max_suppression(box_tlbr, results[:, -1], max_boxes, score_threshold=score_threshold,
                                             iou_threshold=iou_threshold)
    filter_boxes = results[out_boxes.numpy(), :-1]
    orig_points = convert_to_orig_points(filter_boxes, orig_dim, 128)
    return xywh_to_tlbr(orig_points).astype(np.int32)


def midpoint(ptA, ptB):
    return int((ptA[0] + ptB[0]) * 0.5), int((ptA[1] + ptB[1]) * 0.5)


def face_landmark_preprocess(im):
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = np.float32(cv2.resize(im, (res_i, res_i), interpolation=cv2.INTER_LINEAR)) * std_res + mean_res
    im = np.expand_dims(im, 0)
    im = np.transpose(im, (0, 3, 1, 2))
    return im


def logit_arr(p, factor=16.0):
    p = np.clip(p, 0.0000001, 0.9999999)
    return np.log(p / (1 - p)) / float(factor)


def landmarks(tensor, crop_info):
    crop_x1, crop_y1, scale_x, scale_y, _ = crop_info
    res = m_res - 1
    c0, c1, c2 = 66, 132, 198
    t_main = tensor[0:c0].reshape((c0, out_res_i * out_res_i))
    t_m = t_main.argmax(1)
    indices = np.expand_dims(t_m, 1)
    t_conf = np.take_along_axis(t_main, indices, 1).reshape((c0,))
    t_off_x = np.take_along_axis(tensor[c0:c1].reshape((c0, out_res_i * out_res_i)), indices, 1).reshape(
        (c0,))
    t_off_y = np.take_along_axis(tensor[c1:c2].reshape((c0, out_res_i * out_res_i)), indices, 1).reshape(
        (c0,))
    t_off_x = res * logit_arr(t_off_x, logit_factor)
    t_off_y = res * logit_arr(t_off_y, logit_factor)
    t_x = crop_y1 + scale_y * (res * np.floor(t_m / out_res_i) / out_res + t_off_x)
    t_y = crop_x1 + scale_x * (res * np.floor(np.mod(t_m, out_res_i)) / out_res + t_off_y)
    avg_conf = np.average(t_conf)
    lms = np.stack([t_x, t_y, t_conf], 1)
    lms[np.isnan(lms).any(axis=1)] = np.array([0., 0., 0.], dtype=np.float32)
    return avg_conf, np.array(lms)


def get_coords(landmark):
    left_eye_coord_1 = landmark[left_eye_id[0]][:2][::-1]
    left_eye_coord_2 = landmark[left_eye_id[1]][:2][::-1]
    right_eye_coord_1 = landmark[right_eye_id[0]][:2][::-1]
    right_eye_coord_2 = landmark[right_eye_id[1]][:2][::-1]
    x = [l[1] for l in landmark]
    y = [l[0] for l in landmark]
    x_min = int(min(x))
    x_max = int(max(x))
    y_min = int(min(y))
    y_max = int(max(y))
    left_eye = midpoint(left_eye_coord_1, left_eye_coord_2)
    right_eye = midpoint(right_eye_coord_1, right_eye_coord_2)
    return (x_min, y_min, x_max, y_max), left_eye, right_eye


def face_detect(image):
    orig_h, orig_w = image.shape[0:2]
    frame = create_letterbox_image(image, 128)
    input_frame = cv2.cvtColor(cv2.resize(frame, (128, 128)), cv2.COLOR_BGR2RGB)
    input_tensor = np.expand_dims(input_frame.astype(np.float32), 0) / 127.5 - 1
    result = model_face_detect.predict(input_tensor)[0]
    final_boxes = process_detections(result, (orig_h, orig_w))
    return final_boxes


def face_process(image, isLoad=False, cropData=False):
    orig_h, orig_w = image.shape[0:2]
    final_boxes = face_detect(image)
    for bx in final_boxes:
        x1, y1, x2, y2 = bx[:4]
        w_face = x2 - x1
        h_face = y2 - y1
        if x1 < 0 or y1 < 0 or x2 > orig_w or y2 > orig_h:
            if not cropData:
                continue
        face = image[y1: y2, x1: x2, :]
        crop = face_landmark_preprocess(face)
        output = session.run([], {input_name: crop})[0]
        crop_x1 = x1 - int(w_face * 0.1)
        crop_y1 = y1 - int(h_face * 0.125)
        crop_x2 = x2 + int(w_face * 0.1)
        crop_y2 = y2 + int(h_face * 0.125)
        scale_x = float(crop_x2 - crop_x1) / m_res
        scale_y = float(crop_y2 - crop_y1) / m_res
        conf, lms = landmarks(output[0], (crop_x1, crop_y1, scale_x, scale_y, 0.1))
        if conf < 0.6:
            continue
        (x_min, y_min, x_max, y_max), left_eye, right_eye = get_coords(lms)
        if x_min < 0 or y_min < 0 or x_max > orig_w or y_max > orig_h:
            if not cropData:
                continue
        face_aligned = fa.align(image, left_eye, right_eye, w_face, h_face)
        face_aligned = cv2.copyMakeBorder(face_aligned, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=0)
        xf1, yf1, xf2, yf2 = face_detect(face_aligned)[0][:4]
        face_aligned = face_aligned[yf1:yf2, xf1:xf2]
        # face_aligned = cv2.copyMakeBorder(face_aligned, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        if isLoad or cropData:
            yield face_aligned
        else:
            yield (x1, y1, x2, y2), face_aligned
