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
from Data.ellipseParametrisationMethods.lsqFitEllipse import lsq_fit_ellipse
from Data.ellipseParametrisationMethods.naiveFitEllipseV1 import naive_fit_ellipse_v1
from Data.ellipseParametrisationMethods.naiveFitEllipseV0 import naive_fit_ellipse_v0
from constants import ELLIPSES_PATH, ELLIPSE_PARAMETER_NAMES, THRESHOLD_ELLIPSE_DETECTION, POS_X, POS_Y, POS_A, POS_B, \
    POS_THETA, SAVE_DATA_PATH

ELLIPSES_FOLDER = ELLIPSES_PATH

def load_test_ellipse(save=False):
    methods_list = [naive_fit_ellipse_v0, naive_fit_ellipse_v1, ai_fit_ellipse_v0]
    methods_list = [lsq_fit_ellipse]
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
        img_path = join(ELLIPSES_FOLDER, filename)
        ellipse_parameters = get_ellipse_parameters(img_path)
        data.append(ellipse_parameters)
    dataset = pd.DataFrame(data, columns=ELLIPSE_PARAMETER_NAMES)
    return dataset


def get_ellipse_parameters(img_path):
    return naive_fit_ellipse_v0(img_path)


if __name__ == '__main__':
    load_test_ellipse()
