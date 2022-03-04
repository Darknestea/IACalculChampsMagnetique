from nytche import Microscope

from Utils.constants import CBOR_FILE


def test_simulation():
    microscope_sim = Microscope(CBOR_FILE())
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
