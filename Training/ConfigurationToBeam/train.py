import numpy as np
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

from Training.ConfigurationToBeam.Models.simple_deconvnet import simple_deconvolution
from Training.ConfigurationToBeam.Models.simple_sequential import simple_sequential
from Utils.DataPreprocessing.data_preprocessing import get_cleaned_configuration
from Utils.constants import MU_USEFUL_PARAMS, REAL_BEAM_SLICES, SIMULATED_BEAM_SLICES, EXPERIMENT_NAME_FIELD, \
    USER_ID_FIELD, SESSION_FIELD, TEST_SIZE_FIELD, RANDOM_STATE_FIELD, SUMMARY_FIELD, \
    MODEL_NAME_FIELD, HISTORY_FIELD, MODEL_EVALUATION_FIELD, EXPERIMENT_TEST2, IMAGES_SHAPE_FIELD, EXPERIMENT_TEST, \
    EXPERIMENT_4_4, EXPERIMENT_ONLY_LENSES_200
from Utils.model_loader import save_model
from Utils.record_model_performances import evaluate_model
from Utils.utils import normalize_by_column, reshape_images, give_name, normalize_image, main_specific_tasks, \
    get_model_summary
from my_id import MY_ID

IMAGE_WIDTH = 256
IMAGE_HEIGHT = IMAGE_WIDTH
IMAGE_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH)


def load_training_config(experiment_name, user_id, session):
    df = get_cleaned_configuration(experiment_name, user_id, session, parameters=MU_USEFUL_PARAMS)
    data = df.to_numpy(dtype=float)
    data = normalize_by_column(data)
    return data, df.columns


def load_training_images(experiment_name, user_id, session, use_real_images, img_shape=IMAGE_SHAPE):
    if use_real_images:
        images = np.load(REAL_BEAM_SLICES(experiment_name, user_id, session))
    else:
        images = np.load(SIMULATED_BEAM_SLICES(experiment_name, user_id, session))
    if images.shape[1:] != img_shape:
        images = reshape_images(images, img_shape)
    images = normalize_image(images)
    return images.astype(dtype=float)


def train(create_model_function, experiment_name, user_id, session, use_real_images, test_size=0.2, random_state=42,
          epochs=100, batch_size=32, patience=10):

    model_information = {
        EXPERIMENT_NAME_FIELD: experiment_name,
        USER_ID_FIELD: user_id,
        SESSION_FIELD: session,
        TEST_SIZE_FIELD: test_size,
        RANDOM_STATE_FIELD: random_state,
        IMAGES_SHAPE_FIELD: IMAGE_SHAPE
    }

    x, columns = load_training_config(experiment_name, user_id, session)
    y = load_training_images(experiment_name, user_id, session, use_real_images)

    # todo remove
    x = x[:, :9]

    model = create_model_function(x.shape[1:], y.shape[1:], name=give_name(create_model_function.__name__))

    model_information[MODEL_NAME_FIELD] = model.name
    model_information[SUMMARY_FIELD] = get_model_summary(model)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)

    # np.random.shuffle(x_train)

    # Setup training to stop when no improvements shown
    early_callback = EarlyStopping(monitor='val_loss', patience=patience)
    history = model.fit(
        x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, y_test), callbacks=[early_callback]
    )
    model_information[HISTORY_FIELD] = history.history

    save_model(model, model_information)

    model.experiment_name = experiment_name
    evaluate_model(model, model_information[HISTORY_FIELD], x_test, y_test, number_samples=min(y_test.shape[0], 50), record_images=True)
    evaluation = evaluate_model(model, model_information[HISTORY_FIELD], x_test, y_test)
    model_information[MODEL_EVALUATION_FIELD] = evaluation

    print(model_information)

    save_model(model, model_information)


if __name__ == '__main__':
    main_specific_tasks()
    train(simple_sequential, EXPERIMENT_4_4, MY_ID, 156, use_real_images=True,
          epochs=10000, batch_size=32, patience=10)
