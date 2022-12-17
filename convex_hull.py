import cv2
import numpy as np


def convex_hull_outer(src_pts, tgt_pts):
    convex_index = cv2.convexHull(
        np.array(tgt_pts).astype(np.int32), returnPoints=False)

    outer_s = [src_pts[int(i)] for i in convex_index]
    outer_t = [tgt_pts[int(i)] for i in convex_index]

    return outer_s, outer_t


def convex_hull(src_pts, tgt_pts, s_face, t_face):
    # return the index of target_pts that is in the edge of convex hull
    outer_s, outer_t = convex_hull_outer(src_pts, tgt_pts)

    # feature points
    features = ['left_eye', 'right_eye', 'nose_tip', 'bottom_lip']
    feature_s = []
    feature_t = []

    for ele in features:
        if ele in s_face and ele in t_face:
            if len(s_face[ele]) > 0 and len(t_face[ele]) > 0:
                feature_s.append(s_face[ele][0])
                feature_t.append(t_face[ele][0])

    # add features into the convex hull
    hull_s = outer_s + feature_s
    hull_t = outer_t + feature_t

    return hull_s, hull_t


def convex_hull_target_emotion(src_pts, tgt_pts, s_face, t_face):
    # return the index of target_pts that is in the edge of convex hull
    convex_index = cv2.convexHull(
        np.array(tgt_pts), returnPoints=False)

    # outer edge
    outer_s = [src_pts[int(i)] for i in convex_index]
    outer_t = [tgt_pts[int(i)] for i in convex_index]

    # feature points
    features = ['left_eye', 'right_eye', 'nose_tip',
                'nose_bridge', 'bottom_lip']
    feature_s = []
    feature_t = []

    for ele in features:
        if ele in s_face and ele in t_face:
            if len(s_face[ele]) > 0 and len(t_face[ele]) > 0:
                if ele == 'top_lip':
                    feature_s.extend(s_face[ele][7:])
                    feature_t.extend(t_face[ele][7:])
                elif ele == 'bottom_lip':
                    feature_s.extend(s_face[ele][6:])
                    feature_t.extend(t_face[ele][6:])
                else:
                    feature_s.extend(s_face[ele][:5])
                    feature_t.extend(t_face[ele][:5])

    # add features into the convex hull
    hull_s = outer_s + feature_s
    hull_t = outer_t + feature_t

    return hull_s, hull_t
