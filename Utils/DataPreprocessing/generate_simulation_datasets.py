import cv2
import numpy as np
import pandas as pd
from nytche import Microscope
from PIL import Image as pilim

from Utils.DataPreprocessing.data_preprocessing import get_cleaned_configuration
from Utils.constants import MU_OFFSET_BP_P, MU_OFFSET_BP_V, MU_OFFSET_APP, MU_PARAM_YAML_NAMES, MU_PARAM_LENSES, \
    MU_NYTCHE_PARAMS, MU_PARAM_NAMES, MU_PARAM_BIPRISMS, MU_PARAM_APERTURES, MU_OFFSET_BP_LAST, \
    MU_OFFSET_AP_LAST, APERTURE_OPEN_DIAMETER, MU_PARAM_V0, SCREEN_POSITION, SCREEN_SHAPE, SCREEN_CENTER, \
    SIMULATED_BEAM_SLICES, RAW_SIMULATED_BEAM_SLICES_PATH, \
    CURRENT_EXPERIMENT, CBOR_FILE, ELLIPSE_PARAMETER_NAMES, ELLIPSE_PARAMETERS_BY_METHOD, \
    ELLIPSE_PARAMETER_EXTRACTION_DEFAULT_METHOD, SIMULATED_TYPE
from Utils.utils import gray_from_image, load_image


def set_accelerator(name, values, microscope_sim):
    pass


def set_biprism(name, bp_p, bp_v, microscope_sim):
    microscope_sim.biprisms[name].inserted = bp_p < 0.5  # meaning == 0
    microscope_sim.biprisms[name].voltage = bp_v / 1000.


def set_aperture(name, ap_diam_index, microscope_sim):
    if ap_diam_index == 0:
        diam = APERTURE_OPEN_DIAMETER
    else:
        # Plus grand diamètre d'index 1 est l'élément 0 de la liste
        diam = microscope_sim.apertures[name].diameters[ap_diam_index - 1] * 1e-6
    microscope_sim.apertures[name].diameter = diam


def set_gun(name, values, microscope_sim):
    pass


def set_lens(name, value, microscope_sim):
    microscope_sim.lenses[name].current = value


def set_others(name, values, microscope_sim):
    pass


def configurations_to_nytche(configurations):
    columns_not_in_nytche = [name for i, name in enumerate(MU_PARAM_NAMES) if i not in MU_NYTCHE_PARAMS]
    nytche_configurations = configurations.drop(columns_not_in_nytche, axis=1)
    return nytche_configurations


def set_nytche_configuration(nytche_configuration, microscope_sim):
    # set source
    name = MU_PARAM_NAMES[MU_PARAM_V0]
    microscope_sim.acc_voltage = nytche_configuration[1][name]
    # set lenses
    for lens in MU_PARAM_LENSES:
        if lens not in MU_NYTCHE_PARAMS:
            pass
        nytche_index = MU_NYTCHE_PARAMS.index(lens)
        name = MU_PARAM_NAMES[lens]
        yaml_name = MU_PARAM_YAML_NAMES[lens]
        value = nytche_configuration[1][name]
        set_lens(name=yaml_name, value=value, microscope_sim=microscope_sim)

    # set biprism
    for biprism in MU_PARAM_BIPRISMS:
        if all([b not in MU_NYTCHE_PARAMS for b in range(biprism, biprism + MU_OFFSET_BP_LAST)]):
            pass
        name_v = MU_PARAM_NAMES[biprism + MU_OFFSET_BP_V]
        name_p = MU_PARAM_NAMES[biprism + MU_OFFSET_BP_P]
        yaml_name = MU_PARAM_YAML_NAMES[biprism]
        bp_v, bp_p = nytche_configuration[1][name_v], nytche_configuration[1][name_p]
        set_biprism(name=yaml_name, bp_v=bp_v, bp_p=bp_p, microscope_sim=microscope_sim)

    # set apertures
    for aperture in MU_PARAM_APERTURES:
        if all([a not in MU_NYTCHE_PARAMS for a in range(aperture, aperture + MU_OFFSET_AP_LAST)]):
            pass
        name_diam = MU_PARAM_NAMES[aperture + MU_OFFSET_APP]
        yaml_name = MU_PARAM_YAML_NAMES[aperture]
        ap_diam_index = int(nytche_configuration[1][name_diam])
        set_aperture(name=yaml_name, ap_diam_index=ap_diam_index, microscope_sim=microscope_sim)

    # set deflectors

    # set others
    return


def modify_nytche_secondary_configuration():
    pass


# Get information on screen position
def retrieve_nytche_image_output(microscope_sim):
    beam_tree = microscope_sim.beam()
    beam_slice = beam_tree.find(SCREEN_POSITION)
    circles = beam_slice.beam_nodes[0].cuts(SCREEN_POSITION)
    screen_rotation = microscope_sim.larmor_angle(SCREEN_POSITION)
    return circles, screen_rotation


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
            add_circle_to_data(circle, data, i)

        pilim_img = pilim.fromarray(data[i])
        pilim_img.save(image_directory_path + f"{i}.png")
        if do_ellipse_parameter_extraction:
            ellipses_data[i] = ELLIPSE_PARAMETER_EXTRACTION_DEFAULT_METHOD(data[i, :, :, 0])

    data = data[:, :, :, 0]

    np.save(SIMULATED_BEAM_SLICES(experiment), data)

    if do_ellipse_parameter_extraction:
        ellipses = pd.DataFrame(data=ellipses_data, columns=ELLIPSE_PARAMETER_NAMES)
        file_path = ELLIPSE_PARAMETERS_BY_METHOD(experiment, ELLIPSE_PARAMETER_EXTRACTION_DEFAULT_METHOD, data_type=SIMULATED_TYPE)
        ellipses.to_csv(file_path, index=False)


def add_circle_to_data(circle, data, i):
    center_x = SCREEN_CENTER[0] + circle[0]
    center_y = SCREEN_CENTER[1] + circle[1]
    radius = circle[2]
    data[i, :, :, :] = cv2.circle(img=data[i, :, :, :],
                                  center=(int(center_x), int(center_y)),
                                  radius=int(radius),
                                  color=(255, 0, 0),
                                  thickness=cv2.FILLED)
