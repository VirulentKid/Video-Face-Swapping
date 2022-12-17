import cv2
import numpy as np
import pdb
import logging
from skimage.transform import SimilarityTransform, matrix_transform
from helper import *
from convex_hull import *
from triangulation import triangulation
from warping import warping
from landmark_detection import *

# Parameters for lucas kanade optical flow
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


def performOpticalFlow(prev_output, pts_t, frame_t, prev_frame_t):
    p0 = np.asarray(pts_t).astype(np.float32)[:, :, None]
    p0 = np.transpose(p0, (0, 2, 1))
   
    frame_t_gray = cv2.cvtColor(frame_t, cv2.COLOR_BGR2GRAY)
    prev_frame_t_gray = cv2.cvtColor(prev_frame_t, cv2.COLOR_BGR2GRAY)

    p1, st, err = cv2.calcOpticalFlowPyrLK(
        prev_frame_t_gray, frame_t_gray, p0, None, **lk_params)
    good_new = p1[st == 1]
    good_old = p0[st == 1]

    newOutput = np.copy(prev_output)

    tr = SimilarityTransform()
    if tr.estimate(good_old, good_new):
        newOutput = image_transform_helper(
            good_old, good_new, prev_output, frame_t)
    # visualizeBGR(newOutput, None)
    return newOutput, tupleList([], good_new.tolist())


def image_transform_helper(old_pts, new_pts, prev_t, frame_t):
    prev_tWarped = np.copy(frame_t)
   
    # feature_visualize(frame_t, new_pts)
    hull_s, hull_t = convex_hull_outer(old_pts.tolist(), new_pts.tolist())
    # feature_visualize(frame_t, hull_t)
    hull_s = np.array(hull_s).astype(np.float32)
    hull_t = np.array(hull_t).astype(np.float32)
    if empty_pts(hull_s, hull_t):
        return frame_t

    hull_t = np.asarray(hull_t)
    width = frame_t.shape[1]
    height = frame_t.shape[0]
    hull_t[:, 0] = np.clip(hull_t[:, 0], 0, width - 1)
    hull_t[:, 1] = np.clip(hull_t[:, 1], 0, height - 1)
    hull_t = tupleList([], hull_t.astype(np.float32).tolist())

    dt = triangulation(frame_t, hull_t)
    if len(dt) == 0:
        return frame_t

    warping(dt, hull_s, hull_t, prev_t, prev_tWarped)

    return prev_tWarped
