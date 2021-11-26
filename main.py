from constants import *
from data_preprocessing import get_data, get_pretraining_data, get_training_data
from model_loader import new_model, load_model, save_model

MODE = MODE_TRAINING_FULL
SAVE_PATH = './saves/'
PRETRAINING_DATASET = None
TRAINING_DATASET = None
PRETRAINED_MODEL_NAME = None
TRAINED_MODEL_NAME = None



if __name__ == '__main__':

    if MODE | MODE_PRETRAINING:
        x_train, y_train, x_test, y_test = get_pretraining_data(PRETRAINING_DATASET)
        model = new_model()
        pretrained_model = pretrain(model, x_train, y_train)
        record_pretrained_model(pretrained_model, x_test, y_test)
        save_model(pretrained_model, SAVE_PATH, PRETRAINED_MODEL_NAME)

    else:
        pretrained_model = load_model(SAVE_PATH, PRETRAINED_MODEL_NAME)


    if MODE | MODE_REAL_TRAINING:
        x_train, y_train, x_test, y_test = get_training_data(TRAINING_DATASET)
        trained_model = train(pretrained_model, x_train, y_train)
        save_model(trained_model, SAVE_PATH, TRAINED_MODEL_NAME)
        record_trained_model(trained_model, x_test, y_test)

    else:
        trained_model = load_model(SAVE_PATH, TRAINED_MODEL_NAME)
