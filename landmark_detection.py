import cv2
import numpy as np
from face_recognition import face_landmarks
from helper import *


def single_face_landmark_detect(source, target):
    landmarks_s = face_landmarks(source)
    landmarks_t = face_landmarks(target)

    if len(landmarks_s) == 0 or len(landmarks_t) == 0:
        return []

    # find same features b/w source and target (we assume only one face in each video so we take [0])
    s_face = landmarks_s[0]
    t_face = landmarks_t[0]
    p_s, p_t = intersection(s_face, t_face)

    return p_s, p_t, s_face, t_face


'''
Reference: https://stackoverflow.com/questions/39308030/how-do-i-increase-the-contrast-of-an-image-in-python-opencv/41075028
'''


def landmark_detect_c(source_image, target_image):

    def increase_contrast(frame):
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=25, tileGridSize=(16, 16))
        cl = clahe.apply(l_channel)
        # merge the CLAHE enhanced L-channel with the a and b channel
        limg = cv2.merge((cl, a, b))
        enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        return enhanced_img

    s = increase_contrast(source_image)
    t = increase_contrast(target_image)
    return single_face_landmark_detect(s, t)

# find same features b/w source and target


def commonFeature(feature_s, feature_t):
    pts_s, pts_t = [], []

    for k, v in feature_s.items():
        if k in feature_t:
            v_ = feature_t[k]
            if len(v) == len(v_):
                pts_s.extend(v)
                pts_t.extend(v_)

    return pts_s, pts_t


def intersection(feature_s, feature_t):
    pts_s, pts_t = commonFeature(feature_s, feature_t)

    p_source = np.array(pts_s).tolist()
    p_target = np.array(pts_t).tolist()

    p_srcs = []
    p_tgts = []
    tupleList(p_srcs, p_source)
    tupleList(p_tgts, p_target)

    return p_srcs, p_tgts
