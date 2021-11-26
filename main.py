from constants import *
from constants import (
    MODE,
    SAVE_PATH,
    PRETRAINING_DATASET,
    TRAINING_DATASET,
    PRETRAINED_MODEL_NAME,
    TRAINED_MODEL_NAME,
)
from data_preprocessing import get_pretraining_data, get_training_data
from model_loader import new_model, load_model, save_model
from record_model_performances import record_pretrained_model, record_trained_model
from training_model import train
from utils import main_specific_tasks

if __name__ == "__main__":
    main_specific_tasks('real')

    if MODE | MODE_PRETRAINING:
        # Do pretraining from Command to field
        pass
    else:
        pretrained_model = load_model(SAVE_PATH, PRETRAINED_MODEL_NAME)

    if MODE | MODE_REAL_TRAINING:
        x_train, y_train, x_test, y_test = get_training_data(TRAINING_DATASET)
        trained_model = train(pretrained_model, x_train, y_train)
        save_model(trained_model, SAVE_PATH, TRAINED_MODEL_NAME)
        record_trained_model(trained_model, x_test, y_test)

    else:
        trained_model = load_model(SAVE_PATH, TRAINED_MODEL_NAME)
