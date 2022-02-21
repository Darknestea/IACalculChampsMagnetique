from keras import Sequential
from keras.layers import Dense
from keras.metrics import RootMeanSquaredError

from Utils.constants import *


def new_model(name=None):
    # TODO

    model_name = f"default_{run_session()}"
    model = Sequential(name=model_name)
    model.add(Dense(64, input_shape=(DUMMY_MICROSCOPE_PARAMETERS_NUMBER,)))
    model.add(Dense(32))
    model.add(Dense(DUMMY_ELLIPSE_PARAMETERS_NUMBER))

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=[RootMeanSquaredError()])
    return model


def load_model(path, name):
    # TODO
    return new_model()


def save_model(model, path):
    print(path)
    print(f"\\{current_extension()}\\{model.name}.h5")
    model.save(path + f"\\{current_extension()}\\{model.name}.h5")
