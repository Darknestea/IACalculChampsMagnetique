import cv2 as cv
from ellipse import LsqEllipse
import numpy as np


def lsq_fit_ellipse(img_path):
    # read img
    img = cv.imread(img_path)[:, 304:1744]  # Img 1440*1080 since opencv is reversed

    # convert to grayscale
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
    cv.ellipse(result, (int(center[0]), int(center[1])), (int(width), int(height)), np.rad2deg(phi), 0, 360, (0, 0, 255), 2)

    return [result, img_contoured, thresh]
