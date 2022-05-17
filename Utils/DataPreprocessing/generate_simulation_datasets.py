import numpy as np
import pandas as pd
from PIL import Image as PilIm
from cv2 import bitwise_or, imshow, waitKey
from nytche import Microscope
from nytche.utilities.cuttingmasks import CuttingMasks

from Utils.DataPreprocessing.data_preprocessing import get_cleaned_configuration
from Utils.constants import SCREEN_SHAPE, SIMULATED_BEAM_SLICES, RAW_SIMULATED_BEAM_SLICES_PATH, \
    CURRENT_EXPERIMENT, CBOR_FILE, ELLIPSE_PARAMETER_NAMES, ELLIPSE_PARAMETERS_BY_METHOD, \
    ELLIPSE_PARAMETER_EXTRACTION_DEFAULT_METHOD, SIMULATED_TYPE
from Utils.simulation_wrapper import configurations_to_nytche, set_nytche_configuration, retrieve_nytche_image_output, \
    add_circle_to_data, add_plane_to_data


def clean_simulation_dataset(do_ellipse_parameter_extraction, experiment=CURRENT_EXPERIMENT):
    microscope_sim = Microscope(CBOR_FILE())
    configurations = get_cleaned_configuration(experiment, full=True)
    nytche_configurations = configurations_to_nytche(configurations)
    image_directory_path = RAW_SIMULATED_BEAM_SLICES_PATH(experiment)

    data = np.zeros((len(list(nytche_configurations.iterrows())),) + SCREEN_SHAPE + (3,), dtype=np.uint8)

    ellipses_data = None
    if do_ellipse_parameter_extraction:
        ellipses_data = np.zeros((nytche_configurations.shape[0], len(ELLIPSE_PARAMETER_NAMES)), dtype=float)

    for i, nytche_configuration in enumerate(nytche_configurations.iterrows()):
        set_nytche_configuration(nytche_configuration, microscope_sim)
        # todo retrieve magnetic field
        masks, rotation = retrieve_nytche_image_output(microscope_sim)
        for mask in masks.values():
            sub_beam_data = np.ones(data[i].shape, data.dtype) * 255
            circles = mask[CuttingMasks.CIRCLES]
            half_plans = mask[CuttingMasks.PLANS]
            for circle in circles:
                add_circle_to_data(circle, sub_beam_data, rotation)
            # imshow('sub beam before plane', sub_beam_data)
            # waitKey(0)
            for half_plan in half_plans:
                add_plane_to_data(half_plan, sub_beam_data, rotation)
            # imshow('sub beam', sub_beam_data)
            # waitKey(0)
            data[i] = bitwise_or(data[i], sub_beam_data)

        pil_img = PilIm.fromarray(data[i])
        pil_img.save(image_directory_path + f"{i}.png")
        if do_ellipse_parameter_extraction:
            ellipses_data[i] = ELLIPSE_PARAMETER_EXTRACTION_DEFAULT_METHOD(data[i, :, :, 0])

    data = data[:, :, :, 0]

    np.save(SIMULATED_BEAM_SLICES(experiment), data)

    if do_ellipse_parameter_extraction:
        ellipses = pd.DataFrame(data=ellipses_data, columns=ELLIPSE_PARAMETER_NAMES)
        file_path = ELLIPSE_PARAMETERS_BY_METHOD(
            experiment, ELLIPSE_PARAMETER_EXTRACTION_DEFAULT_METHOD, data_type=SIMULATED_TYPE
        )
        ellipses.to_csv(file_path, index=False)


