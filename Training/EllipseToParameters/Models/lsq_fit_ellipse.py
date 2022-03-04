import cv2 as cv
import pandas as pd
from ellipse import LsqEllipse
import numpy as np



def lsq_fit_ellipse_test(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # threshold to binary and invert
    thresh = cv.threshold(gray, 40, 255, cv.THRESH_BINARY)[1]

    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    img_contoured = img.copy()

    points = np.array([point[0] for point in
                       sorted([(len(contour), i, contour) for i, contour in enumerate(contours)], reverse=True)[0][2]])

    contour = np.array([[[point[0], point[1]]] for point in points])

    cv.drawContours(img_contoured, contour, -1, (0, 0, 255), 1)

    x = np.array([point[0] for point in contour], dtype=float)

    reg = LsqEllipse().fit(x)
    print(reg.as_parameters())
    center, width, height, phi = reg.as_parameters()

    # draw ellipse on input img
    result = img.copy()
    cv.ellipse(result, (int(center[0]), int(center[1])), (int(width), int(height)), np.rad2deg(phi), 0, 360,
               (0, 0, 255), 2)

    return [result, img_contoured, thresh]


def lsq_fit_ellipse(gray):
    from Utils.constants import ELLIPSE_PARAMETER_NAMES, ELLIPSE_X, ELLIPSE_Y, ELLIPSE_B, ELLIPSE_A, ELLIPSE_THETA
    # threshold to binary and invert
    thresh = cv.threshold(gray, 40, 255, cv.THRESH_BINARY)[1]

    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    points = np.array([point[0] for point in
                       sorted([(len(contour), i, contour) for i, contour in enumerate(contours)], reverse=True)[0][2]])

    contour = np.array([[[point[0], point[1]]] for point in points])

    x = np.array([point[0] for point in contour], dtype=float)

    reg = LsqEllipse().fit(x)

    center, width, height, phi = reg.as_parameters()

    ellipse = np.zeros(ELLIPSE_PARAMETER_NAMES, dtype=float)

    ellipse[ELLIPSE_X] = center[0]
    ellipse[ELLIPSE_Y] = center[1]

    ellipse[ELLIPSE_A] = width

    ellipse[ELLIPSE_B] = height

    ellipse[ELLIPSE_THETA] = phi

    return ellipse
