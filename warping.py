import cv2
import numpy as np
from helper import *


def warping(tri_result, hull_s, hull_t, frame_s, warpped_t):
    # Transform Delaunay triangles using affine transformation
    for i in range(0, len(tri_result)):

        # Get the triangles' matching points for frame_s and warpped_t.
        tri_points_s = [hull_s[tri_result[i][j]] for j in range(0, 3)]
        tri_points_t = [hull_t[tri_result[i][j]] for j in range(0, 3)]

        # Find bounding rectangle for each triangle
        boundRect_s = cv2.boundingRect(np.float32([tri_points_s]))
        boundRect_t = cv2.boundingRect(np.float32([tri_points_t]))

        # Points that are offset from the corresponding rectangles' top left corner
        offset_boundRect_s = [((tri_points_s[j][0] - boundRect_s[0]), (tri_points_s[j][1] - boundRect_s[1])) for j in range(0, 3)]
        offset_boundRect_t = [((tri_points_t[j][0] - boundRect_t[0]), (tri_points_t[j][1] - boundRect_t[1])) for j in range(0, 3)]

        # an copy for making mask
        offset_boundRect_t_copy = [((tri_points_t[j][0] - boundRect_t[0]), (tri_points_t[j][1] - boundRect_t[1])) for j in range(0, 3)]

        # Fill the triangle to obtain the mask.
        mask = np.zeros((boundRect_t[3], boundRect_t[2], 3), dtype=np.float32)
        cv2.fillConvexPoly(mask, np.int32(offset_boundRect_t_copy), (1.0, 1.0, 1.0), 16, 0)

        # use warpImage on the small rectangular patches
        sourceRect = frame_s[boundRect_s[1]:boundRect_s[1] + boundRect_s[3], boundRect_s[0]:boundRect_s[0] + boundRect_s[2]]

        size = (boundRect_t[2], boundRect_t[3])
        warpMat = cv2.getAffineTransform(np.float32(offset_boundRect_s), np.float32(offset_boundRect_t))
        targetRect = cv2.warpAffine(sourceRect, warpMat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        targetRect = targetRect * mask

        # Copy triangular region of the rectangular patch to the output image
        warpped_t[boundRect_t[1]:boundRect_t[1] + boundRect_t[3],
        boundRect_t[0]:boundRect_t[0] + boundRect_t[2]] = warpped_t[boundRect_t[1]:boundRect_t[1] + boundRect_t[3],
                                                          boundRect_t[0]:boundRect_t[0] + boundRect_t[2]] * ((1.0, 1.0, 1.0) - mask)

        warpped_t[boundRect_t[1]:boundRect_t[1] + boundRect_t[3],
        boundRect_t[0]:boundRect_t[0] + boundRect_t[2]] = warpped_t[boundRect_t[1]:boundRect_t[1] + boundRect_t[3],
                                                          boundRect_t[0]:boundRect_t[0] + boundRect_t[2]] + targetRect



