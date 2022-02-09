from os import listdir
from os.path import join

import cv2 as cv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy import mean
from numpy.linalg import linalg
from scipy import ndimage

from Data.ellipseParametrisationMethods.AIFitEllipseV0 import ai_fit_ellipse_v0
from Data.ellipseParametrisationMethods.naiveFitEllipseV1 import naive_fit_ellipse_v1
from Data.ellipseParametrisationMethods.naiveFitEllipseV0 import naive_fit_ellipse_v0
from constants import ELLIPSES_PATH, ELLIPSE_PARAMETER_NAMES, THRESHOLD_ELLIPSE_DETECTION, POS_X, POS_Y, POS_A, POS_B, \
    POS_THETA, SAVE_DATA_PATH

ELLIPSES_FOLDER = ELLIPSES_PATH

def code_from_internet(path):
    # read img
    img = cv.imread(path)

    # convert to grayscale and blur
    kernel = np.ones((5, 5), np.float32) / 25
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # cv.filter2D(gray, -1, kernel)

    # Morphological opening: Get rid of the stuff at the top of the ellipse
    gray = cv.morphologyEx(gray, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5)))
    cv.imshow("Gray removed noise", gray)
    cv.imwrite('Gray_removed_noise.png', gray)

    # Resize img to original size
    gray = cv.resize(gray, dsize=(img.shape[1], img.shape[0]))

    # threshold to binary and invert
    thresh = cv.threshold(gray, 40, 255, cv.THRESH_BINARY)[1]
    h, w = gray.shape
    s = int(w / 8)
    gray = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, s+(s+1) % 2, 7.0)

    # Downsize img (by factor 4) to speed up morphological operations
    gray = cv.resize(gray, dsize=(0, 0), fx=0.25, fy=0.25)
    cv.imshow("Gray resized", gray)
    cv.imwrite('Gray_resized.png', gray)

    # Morphological opening: Get rid of the stuff at the top of the ellipse
    gray = cv.morphologyEx(gray, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5)))
    cv.imshow("Gray removed noise", gray)
    cv.imwrite('Gray_removed_noise.png', gray)

    # Resize img to original size
    gray = cv.resize(gray, dsize=(img.shape[1], img.shape[0]))

    # Find contours
    cnts, hier = cv.findContours(gray, cv.RETR_CCOMP, cv.CHAIN_APPROX_NONE)
    #
    # # Draw found contours in input img
    # img = cv.drawContours(img, cnts, -1, (0, 0, 255), 2)
    #
    # for i, cont in enumerate(cnts):
    #     h = hier[0, i, :]
    #     print(h)
    #     if h[3] != -1:
    #         elps = cv.fitEllipse(cnts[i])
    #     elif h[2] == -1:
    #         elps = cv.fitEllipse(cnts[i])
    #     cv.ellipse(img, elps, (0, 255, 0), 2)
    #
    # # Downsize img
    # out_img = cv.resize(img, dsize=(0, 0), fx=0.25, fy=0.25)
    # cv.imshow("Output img", out_img)
    # cv.imwrite('Output_img.png', out_img)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    # return

    # fit ellipse
    # note: transpose needed to convert y,x points from numpy to x,y for opencv
    # points = np.column_stack(np.where(thresh.transpose() > 0))
    # hull = cv.convexHull(points)

    # # Get x-gradient in "sx"
    # sx = ndimg.sobel(gray, axis=0, mode='reflect')
    # # Get y-gradient in "sy"
    # sy = ndimg.sobel(gray, axis=1, mode='reflect')
    # # Get square root of sum of squares
    # sobel = np.hypot(sx, sy)
    # print(sobel.shape)
    # threshold = 250
    # sobel = (np.array([[max(point, threshold) for point in raw] for raw in sobel]).reshape(sobel.shape) - threshold) * (threshold/(255-threshold))
    # sobel = np.array([[(max(point, threshold)-threshold for point in raw] for raw in sobel]).reshape(sobel.shape)
    # plt.imshow(sobel, cmap=plt.cm.gray)
    # plt.show()

    if True:
        contours, hierarchy = cv.findContours(gray, cv.RETR_TREE, cv.CHAIN_APPROX_TC89_KCOS)
        img_contoured = img.copy()
        # cv.drawContours(img_contoured, contours, -1, (0, 0, 255), 1, None, hierarchy)
        points = np.array([point[0] for point in
                           sorted([(len(contour), i, contour) for i, contour in enumerate(contours)], reverse=True)[0][2]])

        points = remove_bad_points(points)
        # points = np.array([list(point[0][0]) for point in contours])
        # bad_contour = np.array([[[point[0], point[1]]] for point in points])
        # cv.drawContours(img, bad_contour, -1, (0, 255, 0), 1)
        #
        # points = np.array(remove_bad_points(points))
        # not_bad_contour = np.array([[[point[0], point[1]]] for point in points])
        # cv.drawContours(img, not_bad_contour, -1, (255, 0, 0), 1)
        # cv.drawContours(img_contoured, not_bad_contour, -1, (0, 0, 255), 1)

    ((cent_x, cent_y), (width, height), angle) = cv.fitEllipse(points)
    print("center x,y:", cent_x, cent_y)
    print("diameters:", width, height)
    print("orientation angle:", angle)

    # draw ellipse on input img
    result = img.copy()
    cv.ellipse(result, (int(cent_x), int(cent_y)), (int(width / 2), int(height / 2)), angle, 0, 360, (0, 0, 255), 2)


def load_test_ellipse(save=False):
    methods_list = [naive_fit_ellipse_v0, naive_fit_ellipse_v1]
    # methods_list = [ai_fit_ellipse_v0]
    for method in methods_list:
        for number in ["09752", "14604", "18607", "25194", "39281", "93704"]:
            list_return = method(join(ELLIPSES_FOLDER, "EXP3-7-12-21_000" + number + ".png"))
            for i, result in enumerate(list_return):
                # show results
                cv.namedWindow(f"{method.__name__} {i}", cv.WINDOW_NORMAL)
                cv.imshow(f"{method.__name__} {i}", result)
                if save:
                    cv.imwrite(SAVE_DATA_PATH + f"EllipseResults\\{method.__name__}_{i}.png", result)
            cv.waitKey(0)
            cv.destroyAllWindows()
    return


def load_ellipses():
    data = []
    for filename in sorted(listdir(ELLIPSES_FOLDER)):
        img = cv.imread(join(ELLIPSES_FOLDER, filename))
        ellipse_parameters = get_ellipse_parameters(img)
        data.append(ellipse_parameters)
    dataset = pd.DataFrame(data, columns=ELLIPSE_PARAMETER_NAMES)
    return dataset


def get_ellipse_parameters(img):
    _, thresh = cv.threshold(img, THRESHOLD_ELLIPSE_DETECTION, 255, 0)
    contours, _ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    img = cv.drawContours(img, contours, -1, 255, 1)
    cv.imshow("img", img)
    cv.waitKey()

    # fit ellipse
    contour = largest_contours[0]
    ellipse = cv.fitEllipseDirect(contour)

    # Get ellipse parameters in the desired format
    parameters = [ellipse[0][0], ellipse[0][1], ellipse[1][0], ellipse[1][1], max(ellipse[2], 180 - ellipse[2])]
    return parameters


if __name__ == '__main__':
    load_test_ellipse()