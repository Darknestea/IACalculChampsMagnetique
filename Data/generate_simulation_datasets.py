from nytche import Microscope
import numpy as np
import pandas as pd

from constants import MU_OFFSET_BP_P, MU_OFFSET_BP_V, MU_OFFSET_APP, MU_PARAM_YAML_NAMES, MU_REDUCED_PARAMS, \
    MU_PARAM_LENSES, MU_NYTCHE_PARAMS, MU_PARAM_NAMES
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


def set_biprism(name, values, microscope_sim):
    microscope_sim.biprisms[name].is_on = values[MU_OFFSET_BP_P] < 0.5  # meaning == 0
    microscope_sim.biprisms[name].voltage = values[MU_OFFSET_BP_V]


def set_diaphragm(name, values, microscope_sim):
    microscope_sim.apertures[name].diameter = (values[MU_OFFSET_APP])


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
    # set lenses
    for lens in MU_PARAM_LENSES:
        if lens not in MU_NYTCHE_PARAMS:
            pass
        nytche_index = MU_NYTCHE_PARAMS.index(lens)
        name = MU_PARAM_NAMES[lens]
        yaml_name = MU_PARAM_YAML_NAMES[lens]
        value = nytche_configuration[1][name]
        set_lens(yaml_name, value, microscope_sim)
    # set biprism
    # set diaphragms
    # set others
    pass


def modify_nytche_secondary_configuration():
    pass


def retrieve_nytche_secondary_information():
    pass


def generate_simulation_datasets():
    microscope_sim = Microscope("i2tem.cbor")
    configurations, default = get_cleaned_configuration("Exp3log.txt", full=True)
    nytche_configurations = configurations_to_nytche(configurations)

    for nytche_configuration in nytche_configurations.iterrows():
        set_nytche_configuration(nytche_configuration, microscope_sim)
        retrieve_nytche_secondary_information()


if __name__ == "__main__":
    # main_specific_tasks('generate_dataset', dirname(realpath(__file__)) + "\\run_counter.dat")
    generate_simulation_datasets()
    # test_microscope_sim())

