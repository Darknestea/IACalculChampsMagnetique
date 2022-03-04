from os import listdir
from os.path import join

import cv2 as cv
import pandas as pd

from Training.EllipseToParameters.Models.ai_fit_ellipse_v0 import ai_fit_ellipse_v0
from Training.EllipseToParameters.Models.lsq_fit_ellipse import lsq_fit_ellipse, lsq_fit_ellipse_test
from Training.EllipseToParameters.Models.naive_fit_ellipse_v0 import naive_fit_ellipse_v0_test
from Training.EllipseToParameters.Models.naive_fit_ellipse_v1 import naive_fit_ellipse_v1
from Utils.constants import ELLIPSES_PATH, ELLIPSE_PARAMETER_NAMES, DATA_PATH, ELLIPSE_PARAMETERS_BY_METHOD, \
    ELLIPSE_PARAMETER_EXTRACTION_METHODS, ELLIPSE_PARAMETER_EXTRACTION_DEFAULT_METHOD
from Utils.utils import load_gray_image

ELLIPSES_FOLDER = ELLIPSES_PATH


def load_test_ellipse(save=False):
    methods_list = [
        naive_fit_ellipse_v0_test,
        naive_fit_ellipse_v1,
        ai_fit_ellipse_v0,
        lsq_fit_ellipse_test
    ]
    methods_list = [lsq_fit_ellipse_test]
    for method in methods_list:
        for number in ["09752", "14604", "18607", "25194", "39281", "93704"]:
            image_path = join(ELLIPSES_FOLDER, "EXP3-7-12-21_000" + number + ".png")
            img = cv.imread(image_path)[:, 304:1744]  # Img 1440*1080 since opencv is reversed
            list_return = method(img)
            for i, result in enumerate(list_return):
                # show results
                cv.namedWindow(f"{method.__name__} {i}", cv.WINDOW_NORMAL)
                cv.imshow(f"{method.__name__} {i}", result)
                if save:
                    cv.imwrite(DATA_PATH() + f"EllipseResults\\{method.__name__}_{i}.png", result)
            cv.waitKey(0)
            cv.destroyAllWindows()
    return


# Create a dataframe of ellipses parameters using the best available method
def load_ellipses(experiment_name, parameter_extraction_method=ELLIPSE_PARAMETER_EXTRACTION_DEFAULT_METHOD):
    data = []
    for filename in sorted(listdir(ELLIPSES_FOLDER)):
        img_path = join(ELLIPSES_FOLDER, filename)
        ellipse_parameters = get_ellipse_parameters(img_path, parameter_extraction_method)
        data.append(ellipse_parameters)
    dataset = pd.DataFrame(data, columns=ELLIPSE_PARAMETER_NAMES)
    dataset.to_csv(ELLIPSE_PARAMETERS_BY_METHOD(experiment_name, parameter_extraction_method), index=False)
    return dataset


# get ellipses parameters from images using the specified method
def get_ellipse_parameters(img_path, method=ELLIPSE_PARAMETER_EXTRACTION_DEFAULT_METHOD):
    img = load_gray_image(img_path)
    return method(img)


if __name__ == '__main__':
    load_test_ellipse()
