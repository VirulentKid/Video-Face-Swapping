import numpy as np
import cv2

from scipy.interpolate import UnivariateSpline


def cartoon_effect(img_rgb):
    numDownSamples = 2      
    numBilateralFilters = 7  

    img_color = img_rgb
    for _ in range(numDownSamples):
        img_color = cv2.pyrDown(img_color)

    for _ in range(numBilateralFilters):
        img_color = cv2.bilateralFilter(img_color, 9, 9, 7)

    for _ in range(numDownSamples):
        img_color = cv2.pyrUp(img_color)

    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    img_blur = cv2.medianBlur(img_gray, 7)

    img_edge = cv2.adaptiveThreshold(img_blur, 255,
                                        cv2.ADAPTIVE_THRESH_MEAN_C,
                                        cv2.THRESH_BINARY, 9, 2)
    
    
    img_edge = cv2.cvtColor(img_edge, cv2.COLOR_GRAY2RGB)

    output = cv2.bitwise_and(img_color, img_edge)

    return output