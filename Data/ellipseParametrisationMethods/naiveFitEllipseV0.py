import cv2 as cv
import numpy as np


def naive_fit_ellipse_v0(path):
    # read img
    img = cv.imread(path)[:, 304:1744] #Img 1440*1080 since opencv is reversed

    # convert to grayscale
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # threshold to binary and invert
    thresh = cv.threshold(gray, 40, 255, cv.THRESH_BINARY)[1]

    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    print([len(contour) for contour in contours])

    img_contoured = img.copy()

    points = np.array([point[0] for point in
                       sorted([(len(contour), i, contour) for i, contour in enumerate(contours)], reverse=True)[0][2]])

    contour = np.array([[[point[0], point[1]]] for point in points])

    cv.drawContours(img_contoured, contour, -1, (0, 0, 255), 1)

    # cv.imshow("contoured", img_contoured)
    # cv.waitKey(0)

    ((cent_x, cent_y), (width, height), angle) = cv.fitEllipse(points)
    print("center x,y:", cent_x, cent_y)
    print("diameters:", width, height)
    print("orientation angle:", angle)

    # draw ellipse on input img
    result = img.copy()
    cv.ellipse(result, (int(cent_x), int(cent_y)), (int(width / 2), int(height / 2)), angle, 0, 360, (0, 0, 255), 2)

    return [result, img_contoured, thresh]
