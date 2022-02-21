from Utils.constants import (
    PRETRAINING_DATASET,
    EXTENSION, RUN_SESSION, SAVE_MODEL_PATH
)
from Utils.DataPreprocessing.data_preprocessing import get_pretraining_data
from Utils.model_loader import new_model, save_model
from Utils.record_model_performances import record_pretrained_model
from Training.training_model import pretrain
from Utils.utils import main_specific_tasks


def pretrain_command_to_field():
    x_train, y_train, x_test, y_test = get_pretraining_data(PRETRAINING_DATASET)
    pretrained_model = new_model()
    pretrain(pretrained_model, x_train, y_train)
    record_pretrained_model(pretrained_model, x_test, y_test)
    print(SAVE_MODEL_PATH)
    save_model(pretrained_model, SAVE_MODEL_PATH)


if __name__ == "__main__":
    # Generate unique identifier to avoid data loss
    main_specific_tasks('commands_to_field')
    print(EXTENSION, RUN_SESSION)
    pretrain_command_to_field()
