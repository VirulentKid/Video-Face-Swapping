import cv2
import numpy as np
import random
from helper import *
from scipy import spatial


def triangulation(frame_t, hull_t):
    rectangle = (0, 0,frame_t.shape[1], frame_t.shape[0])
    # trianglation_visualize(rectangle, hull_t, frame_t)
    Tri = spatial.Delaunay(hull_t)
    triangle_res = Tri.simplices
    return tupleList([], triangle_res.tolist())


def trianglation_visualize(rectangle, pts, img):
    subdiv = cv2.Subdiv2D(rectangle)
    for p in pts:
        subdiv.insert(p)

    imgToShow = np.copy(img)
    draw_delaunay(imgToShow, subdiv)
    visualizeBGR(imgToShow, None)





