import numpy as np
import cv2
from PIL import Image as PilIm

from Tests.test_param_to_image import IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_SHAPE
from Utils.constants import *


def command_to_image(configuration, microscope_sim, shape=IMAGE_SHAPE):
    data = np.zeros(SCREEN_SHAPE + (3,), dtype=np.uint8)
    set_nytche_configuration(configuration, microscope_sim)
    circles, _ = retrieve_nytche_image_output(microscope_sim)
    for circle in circles:
        # todo modify since the circles are the apertures and like this it fails if apertures are open
        add_circle_to_data(circle, data)
    return PilIm.fromarray(data[i])


def partial_command_to_field(configuration, microscope_sim):
    data = np.zeros(SCREEN_SHAPE + (3,), dtype=np.uint8)
    set_nytche_configuration(configuration, microscope_sim)
    magnetic_fields = get_nytche_magnetic_fields(microscope_sim)
    return magnetic_fields


# def partial_field_to_ellipse(magnetic_field_field, microscope_sim):
#     # TODO
#     return np.random.random(DUMMY_ELLIPSE_PARAMETERS_NUMBER)


def partial_field_to_image(magnetic_field_field, microscope_sim):
    # TODO
    return np.random.random(IMAGE_HEIGHT * IMAGE_WIDTH).reshape(IMAGE_SHAPE)


def set_accelerator(name, values, microscope_sim):
    pass


def set_biprism(name, bp_p, bp_v, microscope_sim):
    microscope_sim.biprisms[name].inserted = bp_p < 0.5  # meaning == 0
    microscope_sim.biprisms[name].voltage = bp_v / 1000.


def set_aperture(name, ap_diam_index, microscope_sim):
    if ap_diam_index == 0:
        diam = APERTURE_OPEN_DIAMETER
    else:
        # The greatest diameter is at index 1 and correspond to the element at poisition 0 in the list.
        diam = microscope_sim.apertures[name].diameters[ap_diam_index - 1] * 1e-6
    microscope_sim.apertures[name].diameter = diam


def set_gun(name, values, microscope_sim):
    pass


def set_lens(name, value, microscope_sim):
    microscope_sim.lenses[name].current = value


def set_others(name, values, microscope_sim):
    pass


# Transform a configuration df so that it only contains Nytche's columns
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
        set_nytche_lens(lens, microscope_sim, nytche_configuration)

    # set biprisms
    for biprism in MU_PARAM_BIPRISMS:
        set_nytche_biprism(biprism, microscope_sim, nytche_configuration)

    # set apertures
    for aperture in MU_PARAM_APERTURES:
        set_nytche_aperture(aperture, microscope_sim, nytche_configuration)

    # todo set deflectors

    # set others
    return


def set_nytche_aperture(aperture, microscope_sim, nytche_configuration):
    if all([a not in MU_NYTCHE_PARAMS for a in range(aperture, aperture + MU_OFFSET_AP_LAST)]):
        pass
    name_diam = MU_PARAM_NAMES[aperture + MU_OFFSET_APP]
    yaml_name = MU_PARAM_YAML_NAMES[aperture]
    ap_diam_index = int(nytche_configuration[1][name_diam])
    set_aperture(name=yaml_name, ap_diam_index=ap_diam_index, microscope_sim=microscope_sim)


def set_nytche_biprism(biprism, microscope_sim, nytche_configuration):
    if all([b not in MU_NYTCHE_PARAMS for b in range(biprism, biprism + MU_OFFSET_BP_LAST)]):
        pass
    name_v = MU_PARAM_NAMES[biprism + MU_OFFSET_BP_V]
    name_p = MU_PARAM_NAMES[biprism + MU_OFFSET_BP_P]
    yaml_name = MU_PARAM_YAML_NAMES[biprism]
    bp_v, bp_p = nytche_configuration[1][name_v], nytche_configuration[1][name_p]
    set_biprism(name=yaml_name, bp_v=bp_v, bp_p=bp_p, microscope_sim=microscope_sim)


def set_nytche_lens(lens, microscope_sim, nytche_configuration):
    if lens not in MU_NYTCHE_PARAMS:
        pass
    nytche_index = MU_NYTCHE_PARAMS.index(lens)
    name = MU_PARAM_NAMES[lens]
    yaml_name = MU_PARAM_YAML_NAMES[lens]
    value = nytche_configuration[1][name]
    set_lens(name=yaml_name, value=value, microscope_sim=microscope_sim)


def modify_nytche_magnetic_fields(magnetic_fields):
    # TODO
    pass


def get_nytche_magnetic_fields(microscope_sim):
    # TODO
    magnetic_fields = np.random.random(DUMMY_MAGNETIC_FIELD_SIZE)
    return magnetic_fields


# Get information on screen position
def retrieve_nytche_image_output(microscope_sim):
    beam_tree = microscope_sim.beam()
    beam_slice = beam_tree.find(SCREEN_POSITION)
    circles = beam_slice.beam_nodes[0].cuts(SCREEN_POSITION)
    screen_rotation = microscope_sim.larmor_angle(SCREEN_POSITION)
    return circles, screen_rotation


def add_circle_to_data(circle, data):
    center_x = SCREEN_CENTER[0] + circle[0]
    center_y = SCREEN_CENTER[1] + circle[1]
    radius = circle[2]
    data[:] = cv2.circle(img=data,
                         center=(int(center_x), int(center_y)),
                         radius=int(radius),
                         color=(255, 0, 0),
                         thickness=cv2.FILLED)
