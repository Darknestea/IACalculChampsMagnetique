from math import cos, sin

import numpy as np
import cv2
from PIL import Image as PilIm
from nytche.utilities.cuttingmasks import CuttingMasks

from Tests.test_param_to_image import IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_SHAPE
from Utils.constants import *


def command_to_image(configuration, microscope_sim, shape=IMAGE_SHAPE):
    data = np.zeros(SCREEN_SHAPE + (3,), dtype=np.uint8)
    set_nytche_configuration(configuration, microscope_sim)
    masks, rotation = retrieve_nytche_image_output(microscope_sim)
    for mask in masks.values():
        data_temp = np.zeros(SCREEN_SHAPE + (3,), dtype=np.uint8) + 255
        circles = mask[CuttingMasks.CIRCLES]
        half_plans = mask[CuttingMasks.PLANS]
        for circle in circles:
            add_circle_to_data(circle, data_temp, rotation)
        for half_plan in half_plans:
            add_plane_to_data(half_plan, data_temp, rotation)
        data[:] = cv2.bitwise_or(data, data_temp)
    return PilIm.fromarray(data)


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


def get_attr_from_sim(sim, attr_name):
    return getattr(sim, attr_name, f'The attribute named {attr_name} does not exist')


def set_attr_from_sim(sim, attr_name, value):
    if hasattr(sim, attr_name):
        setattr(sim, attr_name, value)
    else:
        raise f'The attribute named {attr_name} does not exist'


def set_accelerator(name, values, microscope_sim):
    pass


def set_biprism(biprism, bp_pow, bp_pos, bp_v):
    biprism.inserted = (bp_pow > 0.5) * (bp_pos > 0.5)  # meaning inserted and active
    biprism.voltage = bp_v / 1000.


def set_aperture(aperture, ap_diam_index):
    if ap_diam_index < 0.5:  # should be == 0 but like this it allows continual range
        diam = APERTURE_OPEN_DIAMETER
    else:
        # The greatest diameter is at index 1 and correspond to the element at position 0 in the list.
        relative_index = min(int(ap_diam_index) - 1, len(aperture.diameters) - 1)
        diam = aperture.diameters[relative_index] * 1e-6
    aperture.diameter = diam


def set_gun(name, values, microscope_sim):
    pass


def set_lens(lens, value, microscope_sim):
    lens.current = value


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
    # list_of_things = [x[1] for x in  [('MU_PARAM_C1', 1.98774), ('MU_PARAM_C2', 4.0), ('MU_PARAM_C3', 0.852465), ('MU_PARAM_OBJ', 12.2781), ('MU_PARAM_I1', 2.821), ('MU_PARAM_I2', 1.75558), ('MU_PARAM_I3', 0.935873), ('MU_PARAM_P1', 1.85201), ('MU_PARAM_P2', 4.3917), ('MU_PARAM_V0', 300000.0)]]
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


def set_nytche_aperture(aperture_id, microscope_sim, nytche_configuration):
    if all([a not in MU_NYTCHE_PARAMS for a in range(aperture_id, aperture_id + MU_OFFSET_AP_LAST)]):
        pass
    aperture = microscope_sim.apertures[MU_PARAM_NYTCHE_NAME[aperture_id]]
    name_diam = MU_PARAM_NAMES[aperture_id + MU_OFFSET_APP]
    ap_diam_index = int(nytche_configuration[1][name_diam])
    set_aperture(aperture=aperture, ap_diam_index=ap_diam_index)


def set_nytche_biprism(biprism_id, microscope_sim, nytche_configuration):
    if all([b not in MU_NYTCHE_PARAMS for b in range(biprism_id, biprism_id + MU_OFFSET_BP_LAST)]):
        pass
    biprism = microscope_sim.biprisms[MU_PARAM_NYTCHE_NAME[biprism_id]]
    name_pos = MU_PARAM_NAMES[biprism_id + MU_OFFSET_BP]
    name_v = MU_PARAM_NAMES[biprism_id + MU_OFFSET_BP_V]
    name_pow = MU_PARAM_NAMES[biprism_id + MU_OFFSET_BP_P]
    bp_pos = nytche_configuration[1][name_pos]
    bp_v = nytche_configuration[1][name_v]
    bp_pow = nytche_configuration[1][name_pow]
    set_biprism(biprism=biprism, bp_v=bp_v, bp_pow=bp_pow, bp_pos=bp_pos)


def set_nytche_lens(lens_id, microscope_sim, nytche_configuration):
    if lens_id not in MU_NYTCHE_PARAMS:
        pass
    lens = microscope_sim.lenses[MU_PARAM_NYTCHE_NAME[lens_id]]
    name = MU_PARAM_NAMES[lens_id]
    value = nytche_configuration[1][name]
    set_lens(lens=lens, value=value, microscope_sim=microscope_sim)


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
    beam_slice = beam_tree.find(SCREEN_POSITION_CORRECTED)
    masks = {}
    if beam_slice is None:
        return masks, 0

    for node in beam_slice:
        masks[node] = node.cut(SCREEN_POSITION_CORRECTED)
    screen_rotation = microscope_sim.larmor_angle(SCREEN_POSITION_CORRECTED)
    return masks, screen_rotation


def add_circle_to_data(circle, data, rotation):
    center_x = SCREEN_CENTER[0] + convert_to_pixel(circle['x'] * cos(rotation) + circle['y'] * sin(rotation))
    center_y = SCREEN_CENTER[1] + convert_to_pixel(circle['x'] * sin(rotation) + circle['y'] * cos(rotation))
    radius = convert_to_pixel(circle['r'])
    radius = min(radius, SCREEN_SIZE * SCREEN_DPI)


    mask = np.zeros(data.shape, dtype=np.uint8)
    cv2.circle(img=mask,
               center=(
                   int(center_x),
                   int(center_y)
               ),
               radius=int(radius),
               color=(255, 255, 255),
               thickness=cv2.FILLED).astype(np.uint8)

    # cv2.imshow('data before', data)
    # cv2.waitKey(0)
    # cv2.imshow('mask', mask)
    # cv2.waitKey(0)

    data[:] = cv2.bitwise_and(data, mask)

    # cv2.imshow('data after', data)
    # cv2.waitKey(0)


def convert_to_pixel(length):
    return length * SCREEN_SIZE_RATIO * SCREEN_DPI


# Return red if the point is inside the half-plane and black otherwise
def is_in_half_plane(cx, cy, bx, by, py, px):
    return 255 * ((px - cx) * bx + (py - cy) * by > 0)


def add_plane_to_data(plane, data, rotation):
    center_x = SCREEN_CENTER[0] + convert_to_pixel(plane['x'] * cos(rotation) + plane['y'] * sin(rotation))
    center_y = SCREEN_CENTER[1] + convert_to_pixel(plane['y'] * cos(rotation) + plane['x'] * sin(rotation))
    theta = plane['theta']

    border_point_x = cos(theta + np.pi/2 + rotation)
    border_point_y = sin(theta + np.pi/2 + rotation)

    mask = np.fromfunction(
        lambda i, j, k: is_in_half_plane(
            center_x,
            center_y,
            border_point_x,
            border_point_y,
            i,
            j
        ), # i: y, j: x
        data.shape).astype(np.uint8)

    data[:] = cv2.bitwise_and(data, mask)
    # cv2.imshow('mask', mask)
    # cv2.waitKey(0)
    #
    # cv2.imshow('data', data)
    # cv2.waitKey(0)
