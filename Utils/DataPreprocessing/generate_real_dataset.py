from os import listdir
import glob

import cv2 as cv
import numpy as np
import pandas as pd

from Utils.constants import CURRENT_EXPERIMENT, SCREEN_SHAPE, REAL_BEAM_SLICES, RAW_REAL_BEAM_SLICES_PATH, \
    ELLIPSE_PARAMETER_NAMES, ELLIPSE_PARAMETER_EXTRACTION_DEFAULT_METHOD, ELLIPSE_PARAMETERS_BY_METHOD, VERBOSE, \
    VERBOSE_INFO
from Utils.utils import load_image, gray_from_image, load_gray_image


def clean_real_dataset(do_ellipse_parameter_extraction, experiment=CURRENT_EXPERIMENT):

    raw_beam_path = RAW_REAL_BEAM_SLICES_PATH(experiment)

    images = [path for path in glob.iglob(raw_beam_path + '\\*.png', recursive=False)]

    images = sorted(images, key=lambda x: int(x.split('\\')[-1].split('_')[0]))

    if len(images) == 0:
        if VERBOSE & VERBOSE_INFO:
            print(f"There is no valid images in the {raw_beam_path} folder")
        raise FileNotFoundError

    data = np.zeros((len(images),) + SCREEN_SHAPE, dtype=float)

    ellipses_data = None
    if do_ellipse_parameter_extraction:
        ellipses_data = np.zeros((len(images), len(ELLIPSE_PARAMETER_NAMES)), dtype=float)

    for i, image_path in enumerate(images):
        gray = load_gray_image(image_path)
        data[i, :, :] = gray

        if do_ellipse_parameter_extraction:
            ellipses_data[i] = ELLIPSE_PARAMETER_EXTRACTION_DEFAULT_METHOD(gray)

    if do_ellipse_parameter_extraction:
        ellipses = pd.DataFrame(data=ellipses_data, columns=ELLIPSE_PARAMETER_NAMES)
        file_path = ELLIPSE_PARAMETERS_BY_METHOD(experiment, ELLIPSE_PARAMETER_EXTRACTION_DEFAULT_METHOD)
        ellipses.to_csv(file_path, index=False)

    np.save(REAL_BEAM_SLICES(experiment), data)






