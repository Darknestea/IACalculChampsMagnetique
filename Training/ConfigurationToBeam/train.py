import numpy as np
from sklearn.model_selection import train_test_split

from Training.ConfigurationToBeam.Models.simple_deconvnet import simple_deconvolution
from Utils.constants import MU_USEFUL_PARAMS, REAL_BEAM_SLICES, SIMULATED_BEAM_SLICES, EXPERIMENT_TEST, \
    EXPERIMENT_NAME_FIELD, USER_ID_FIELD, SESSION_FIELD, TEST_SIZE_FIELD, RANDOM_STATE_FIELD, SUMMARY_FIELD, \
    MODEL_NAME_FIELD, HISTORY_FIELD, MODEL_EVALUATION_FIELD, EXPERIMENT_TEST2
from Utils.DataPreprocessing.data_preprocessing import get_cleaned_configuration
from Utils.model_loader import save_model
from Utils.record_model_performances import evaluate_model
from Utils.utils import normalize_by_column, reshape_images, give_name, normalize_image
from my_id import MY_ID

IMAGE_WIDTH = 28
IMAGE_HEIGHT = 28
IMAGE_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH)


def load_training_config(experiment_name, user_id, session):
    df = get_cleaned_configuration(experiment_name, user_id, session, parameters=MU_USEFUL_PARAMS)
    data = df.to_numpy(dtype=float)
    data = normalize_by_column(data)
    return data, df.columns


def load_training_images(experiment_name, user_id, session, real_images=True):
    if real_images:
        images = np.load(REAL_BEAM_SLICES(experiment_name, user_id, session))
    else:
        images = np.load(SIMULATED_BEAM_SLICES(experiment_name, user_id, session))
    if images.shape[1:] != IMAGE_SHAPE:
        images = reshape_images(images, IMAGE_SHAPE)
    images = normalize_image(images)
    return images


def train(create_model_function, experiment_name, user_id, session, test_size=0.2, random_state=42):

    model_information = {
        EXPERIMENT_NAME_FIELD: experiment_name,
        USER_ID_FIELD: user_id,
        SESSION_FIELD: session,
        TEST_SIZE_FIELD: test_size,
        RANDOM_STATE_FIELD: random_state,
    }

    x, columns = load_training_config(experiment_name, user_id, session)
    y = load_training_images(experiment_name, user_id, session)

    model = create_model_function(x.shape[1:], y.shape[1:])
    model.name = give_name(create_model_function.__name__)
    model_information[MODEL_NAME_FIELD] = model.name

    model_information[SUMMARY_FIELD] = model.summary()

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)

    history = model.fit(x_train, y_train, epochs=50, batch_size=32, validation_data=(x_test, y_test))
    model_information[HISTORY_FIELD] = history

    evaluation = evaluate_model(model, history, x_test, y_test)
    model_information[MODEL_EVALUATION_FIELD] = evaluation

    save_model(model, model_information)


if __name__ == '__main__':
    train(simple_deconvolution, EXPERIMENT_TEST2, MY_ID, 42)
