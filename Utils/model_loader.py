import json

import keras.models

from Utils.DataPreprocessing.create_folders import create_model_folder
from Utils.constants import *
from Utils.utils import get_model_name


def load_model(base_name, user_id, session):
    model_name = get_model_name(base_name, user_id, session)
    model = keras.models.load_model(MODEL(model_name))
    with open(MODEL_INFORMATION(model_name)) as fp:
        model_information = json.load(fp)
    return model, model_information


def save_model(model, model_information):
    model_path, information_path = create_model_folder(model)
    model.save(model_path)
    with open(information_path, 'w') as fp:
        json.dump(model_information, fp)
