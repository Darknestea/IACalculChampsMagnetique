from os import mkdir
from os.path import isdir

import cv2
from nytche import Microscope
import numpy as np
import pandas as pd

from constants import MU_OFFSET_BP_P, MU_OFFSET_BP_V, MU_OFFSET_APP, MU_PARAM_YAML_NAMES, MU_REDUCED_PARAMS, \
    MU_PARAM_LENSES, MU_NYTCHE_PARAMS, MU_PARAM_NAMES, MU_PARAM_BIPRISMS, MU_PARAM_APERTURES, MU_OFFSET_BP_LAST, \
    MU_OFFSET_AP_LAST, APERTURE_OPEN_DIAMETER, MU_PARAM_V0, MU_PARAM_V1, SCREEN_POSITION, SCREEN_SHAPE, SCREEN_CENTER, \
    SAVE_CLEAN_SIMULATION_DATA_PATH
from data_preprocessing import get_cleaned_configuration


def test_microscope_sim():
    microscope_sim = Microscope("i2tem.cbor")
    # print([item[0] for item in microscope_sim.lenses.items()])
    # print([item[0] for item in microscope_sim.biprisms.items()])
    # print([item[0] for item in microscope_sim.apertures.items()])
    # print([item[0] for item in microscope_sim.deflectors.items()])
    # print([item[0] for item in microscope_sim.object_planes.items()])
    # print([item[0] for item in microscope_sim.screens.items()])

    # print(microscope_sim.state)
    # print(microscope_sim.z_source)
    # print(microscope_sim.z_start)
    # print(microscope_sim.z_end)
    #
    # print(microscope_sim._z)
    # print(microscope_sim.acc_voltage)
    #
    # lenses = [item for item in microscope_sim.lenses.items()]
    #
    # one_field = lenses[0][1]._potential.currents
    # print(one_field)
    #
    # current = 4.0
    # print([lenses[0][1]._potential.ahw(current)])
    #
    # potentials = [lens[1]._potential.ahw(current) for lens in lenses[0:2]]
    # potentials += [lens[1]._potential.ahw for lens in lenses[2:]]
    # plot(range(len(potentials[0])), potentials[0])
    # show()

    print([[i for i in item] for item in microscope_sim.biprisms.items()])


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
def retrieve_nytche_secondary_information(microscope_sim):
    beam_tree = microscope_sim.beam()
    beam_slice = beam_tree.find(SCREEN_POSITION)
    circles = beam_slice.beam_nodes[0].cuts(SCREEN_POSITION)
    screen_rotation = microscope_sim.larmor_angle(SCREEN_POSITION)
    return circles, screen_rotation


def generate_simulation_datasets():
    microscope_sim = Microscope("i2tem.cbor")
    configuration_file_name = "Exp3log.txt"
    configurations, default = get_cleaned_configuration(configuration_file_name, full=True)
    nytche_configurations = configurations_to_nytche(configurations)

    data = np.zeros((len(list(nytche_configurations.iterrows())),) + SCREEN_SHAPE + (3,), dtype=float)

    for i, nytche_configuration in enumerate(nytche_configurations.iterrows()):
        set_nytche_configuration(nytche_configuration, microscope_sim)
        circles, _ = retrieve_nytche_secondary_information(microscope_sim)
        for circle in circles:
            center_x = SCREEN_CENTER[0] + circle[0]
            center_y = SCREEN_CENTER[1] + circle[1]
            radius = circle[2]
            data[i, :, :, :] = cv2.circle(img=data[i, :, :, :],
                                          center=(int(center_x), int(center_y)),
                                          radius=int(radius),
                                          color=(255, 0, 0),
                                          thickness=cv2.FILLED)
    data = data[:, :, :, 0]
    np.save(SAVE_CLEAN_SIMULATION_DATA_PATH + configuration_file_name.replace("log.txt", "_simulation_images.npy"), data)
    image_directory_path = SAVE_CLEAN_SIMULATION_DATA_PATH + configuration_file_name.replace("log.txt", f"_simulation_images")
    if not isdir(image_directory_path):
        mkdir(image_directory_path)
    for i in range(data.shape[0]):
        cv2.imwrite(image_directory_path + f"\\{i}.png",
                    data[i],
                    )


if __name__ == "__main__":
    # main_specific_tasks('generate_dataset', dirname(realpath(__file__)) + "\\run_counter.dat")
    generate_simulation_datasets()
    # test_microscope_sim())
