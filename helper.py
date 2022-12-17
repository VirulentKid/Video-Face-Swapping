import numpy as np
import cv2
import matplotlib.pyplot as plt
import pdb
import logging


def tupleList(res, pts):
    for lst in pts:
        l = [i for i in lst]
        res.append(tuple(l))
    return res


'''
Feature Detection Visualization
'''


def empty_pts(p_s, p_t):
    if len(p_s) == 0 or len(p_t) == 0:
        return True

    return False


def feature_visualize(bgrImg, pts):
    pts = np.array(pts)
    visualizeBGR(bgrImg, (pts[:, 0], pts[:, 1]))


def visualizeRGB(rgbImg, pts):
    plt.imshow(rgbImg)

    if pts:
        x, y = pts
        plt.scatter(x, y, s=5, c='g')

    plt.show()


def visualizeBGR(bgrImg, pts):
    temp = np.copy(bgrImg)
    rgbImg = cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)
    plt.imshow(rgbImg)

    if pts:
        x, y = pts
        plt.scatter(x, y, s=5, c='g')

    plt.show()


def visualizeGrey(img):
    plt.imshow(img, cmap='gray')
    plt.show()

# Check if a point is inside a rectangle


def rectContains(rect, point):
    if point[0] < rect[0]:
        return False
    elif point[1] < rect[1]:
        return False
    elif point[0] > rect[0] + rect[2]:
        return False
    elif point[1] > rect[1] + rect[3]:
        return False
    return True

# Draw delaunay triangles


def draw_delaunay(img, subdiv, delaunay_color=(255, 255, 255)):
    triangleList = subdiv.getTriangleList()
    size = img.shape
    r = (0, 0, size[1], size[0])

    for t in triangleList:
        pt1 = (int(t[0]), int(t[1]))
        pt2 = (int(t[2]), int(t[3]))
        pt3 = (int(t[4]), int(t[5]))

        if rectContains(r, pt1) and rectContains(r, pt2) and rectContains(r, pt3):
            cv2.line(img, pt1, pt2, delaunay_color, 1, cv2.LINE_AA, 0)
            cv2.line(img, pt2, pt3, delaunay_color, 1, cv2.LINE_AA, 0)
            cv2.line(img, pt3, pt1, delaunay_color, 1, cv2.LINE_AA, 0)


def convert_BGR2Gray(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img
