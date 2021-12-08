from constants import (
    PRETRAINING_DATASET,
    SAVE_PATH, EXTENSION, RUN_SESSION
)
from data_preprocessing import get_pretraining_data
from model_loader import new_model, save_model
from record_model_performances import record_pretrained_model
from training_model import pretrain
from utils import main_specific_tasks


def pretrain_command_to_field():
    x_train, y_train, x_test, y_test = get_pretraining_data(PRETRAINING_DATASET)
    pretrained_model = new_model()
    pretrain(pretrained_model, x_train, y_train)
    record_pretrained_model(pretrained_model, x_test, y_test)
    print(SAVE_PATH)
    save_model(pretrained_model, SAVE_PATH)


if __name__ == "__main__":
    # Generate unique identifier to avoid data loss
    main_specific_tasks('commands_to_field')
    print(EXTENSION, RUN_SESSION)
    pretrain_command_to_field()
