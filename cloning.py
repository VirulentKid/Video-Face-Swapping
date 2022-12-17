import cv2,numpy as np
from helper import *

def cloning(target_Warped, target_frame, target_hull):
    target_hall_array = np.array(target_hull)
    target_hull = cv2.convexHull(target_hall_array.astype(np.int32), returnPoints=True)
    target_hull = np.squeeze(target_hull)

    target_hull_tuplelist = tupleList([], target_hull)
    mask_shape = np.zeros(target_frame.shape, dtype=target_frame.dtype)
    cv2.fillConvexPoly(mask_shape, np.int32(target_hull_tuplelist), (255, 255, 255))

    rect = cv2.boundingRect(np.float32([target_hull]))
    center = ((rect[0] + int(rect[2] / 2), rect[1] + int(rect[3] / 2)))

    output = cv2.seamlessClone(np.uint8(target_Warped), target_frame, mask_shape, center, cv2.NORMAL_CLONE)
    return output
