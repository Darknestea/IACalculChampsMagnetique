from os import listdir
from os.path import join

import cv2 as cv
import pandas as pd

from Training.EllipseToParameters.Models.ai_fit_ellipse_v0 import ai_fit_ellipse_v0
from Training.EllipseToParameters.Models.lsq_fit_ellipse import lsq_fit_ellipse
from Training.EllipseToParameters.Models.naive_fit_ellipse_v1 import naive_fit_ellipse_v1
from Training.EllipseToParameters.Models.naive_fit_ellipse_v0 import naive_fit_ellipse_v0
from Utils.constants import ELLIPSES_PATH, ELLIPSE_PARAMETER_NAMES, SAVE_DATA_PATH

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
