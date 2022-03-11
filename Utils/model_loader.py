import json

import keras.models

from Utils.DataPreprocessing.create_folders import create_model_folder
from Utils.constants import *
from Utils.utils import get_model_name


# def new_model(name=None):
#     # TODO
#
#     model_name = f"default_{run_session()}"
#     model = Sequential(name=model_name)
#     model.add(Dense(64, input_shape=(DUMMY_MICROSCOPE_PARAMETERS_NUMBER,)))
#     model.add(Dense(32))
#     model.add(Dense(DUMMY_ELLIPSE_PARAMETERS_NUMBER))
#
#     model.compile(optimizer='adam', loss='mean_squared_error', metrics=[RootMeanSquaredError()])
#     return model


def load_model(base_name, user_id, session):
    model_name = get_model_name(base_name, user_id, session)
    model = keras.models.load_model(MODEL(model_name))
    model_information = json.load(MODEL_INFORMATION(model_name))
    return model, model_information


def save_model(model, parameters):
    model_path, parameters_path = create_model_folder(model)
    model.save(model_path)
    with open(parameters_path, 'w') as fp:
        json.dump(parameters, fp)
