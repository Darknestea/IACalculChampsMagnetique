import numpy as np
import pandas as pd
from PIL import Image as PilIm
from nytche import Microscope

from Utils.DataPreprocessing.data_preprocessing import get_cleaned_configuration
from Utils.constants import SCREEN_SHAPE, SIMULATED_BEAM_SLICES, RAW_SIMULATED_BEAM_SLICES_PATH, \
    CURRENT_EXPERIMENT, CBOR_FILE, ELLIPSE_PARAMETER_NAMES, ELLIPSE_PARAMETERS_BY_METHOD, \
    ELLIPSE_PARAMETER_EXTRACTION_DEFAULT_METHOD, SIMULATED_TYPE
from Utils.simulation_wrapper import configurations_to_nytche, set_nytche_configuration, retrieve_nytche_image_output, \
    add_circle_to_data


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
        circles, _ = retrieve_nytche_image_output(microscope_sim)
        for circle in circles:
            # todo modify since the circles are the apertures and like this it fails if apertures are open
            add_circle_to_data(circle, data[i])

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


