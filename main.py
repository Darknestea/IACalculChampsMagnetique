from Utils.constants import *
from Utils.constants import (
    MODE,
    SAVE_PATH,
    TRAINING_DATASET,
    PRETRAINED_MODEL_NAME,
    TRAINED_MODEL_NAME,
)
from Utils.DataPreprocessing.data_preprocessing import get_training_data
from Utils.model_loader import load_model, save_model
from Utils.record_model_performances import record_trained_model
from Training.training_model import train
from Utils.utils import main_specific_tasks

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
        save_model(trained_model, SAVE_PATH)
        record_trained_model(trained_model, x_test, y_test)

    else:
        trained_model = load_model(SAVE_PATH, TRAINED_MODEL_NAME)
