from os.path import realpath, dirname

from nytche import Microscope

from utils import main_specific_tasks

from matplotlib.pyplot import plot, show


def test_microscope():
    microscope = Microscope("i2tem.cbor")
    # print([item[0] for item in microscope.lenses.items()])
    # print([item[0] for item in microscope.biprisms.items()])
    # print([item[0] for item in microscope.apertures.items()])
    # print([item[0] for item in microscope.deflectors.items()])
    # print([item[0] for item in microscope.object_planes.items()])
    # print([item[0] for item in microscope.screens.items()])

    # print(microscope.state)
    # print(microscope.z_source)
    # print(microscope.z_start)
    # print(microscope.z_end)
    #
    # print(microscope._z)
    # print(microscope.acc_voltage)
    #
    # lenses = [item for item in microscope.lenses.items()]
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


    print([[i for i in item] for item in microscope.biprisms.items()])


def generate_microscope_parameters_dataset():
    pass


def generate_simulation_datasets():
    microscope = Microscope("i2tem.cbor")


if __name__ == "__main__":
    # main_specific_tasks('generate_dataset', dirname(realpath(__file__)) + "\\run_counter.dat")
    # generate_simulation_datasets()
    test_microscope()
