from os.path import realpath, dirname

from nytche import Microscope

from utils import main_specific_tasks


def generate_simulation_datasets():
    microscope = Microscope("i2tem.cbor")
    print([item[0] for item in microscope.lenses.items()])
    print([item[0] for item in microscope.biprisms.items()])
    print([item[0] for item in microscope.apertures.items()])
    print([item[0] for item in microscope.deflectors.items()])
    print([item[0] for item in microscope.object_planes.items()])
    print([item[0] for item in microscope.screens.items()])

    print(microscope.state)
    print(microscope.z_source)
    print(microscope.z_start)
    print(microscope.z_end)

    print(microscope._z)
    print(microscope.acc_voltage)


if __name__ == "__main__":
    # main_specific_tasks('generate_dataset', dirname(realpath(__file__)) + "\\run_counter.dat")
    generate_simulation_datasets()
