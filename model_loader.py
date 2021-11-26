from keras import Sequential
from keras.layers import Dense
from constants import *


def new_model():
    # TODO
    model = Sequential()
    model.add(Dense(64, input_shape=(DUMMY_MICROSCOPE_PARAMETERS_NUMBER,)))
    model.add(Dense(32))
    model.add(Dense(DUMMY_ELLIPSE_PARAMETERS_NUMBER))
    return model


def load_model(path, name):
    # TODO
    return new_model()


def save_model(model, path, name):
    model.save(path + f"\\{name}.h5")
