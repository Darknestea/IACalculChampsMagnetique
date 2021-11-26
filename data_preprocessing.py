import numpy as np

from constants import *


def get_dummy_pretraining_data():
    # TODO replace dummy
    return (
        np.random.random(DUMMY_MICROSCOPE_PARAMETERS_NUMBER * DUMMY_TRAIN_SIZE).reshape(
            DUMMY_TRAIN_SIZE, DUMMY_MICROSCOPE_PARAMETERS_NUMBER
        ),
        np.random.random(DUMMY_ELLIPSE_PARAMETERS_NUMBER * DUMMY_TRAIN_SIZE).reshape(
            DUMMY_TRAIN_SIZE, DUMMY_ELLIPSE_PARAMETERS_NUMBER
        ),
        np.random.random(DUMMY_MICROSCOPE_PARAMETERS_NUMBER * DUMMY_TEST_SIZE).reshape(
            DUMMY_TEST_SIZE, DUMMY_MICROSCOPE_PARAMETERS_NUMBER
        ),
        np.random.random(DUMMY_ELLIPSE_PARAMETERS_NUMBER * DUMMY_TEST_SIZE).reshape(
            DUMMY_TEST_SIZE, DUMMY_ELLIPSE_PARAMETERS_NUMBER
        ),
    )


def get_dummy_training_data():
    # TODO replace dummy
    return (
        np.random.random(DUMMY_MICROSCOPE_PARAMETERS_NUMBER * DUMMY_TRAIN_SIZE).reshape(
            DUMMY_TRAIN_SIZE, DUMMY_MICROSCOPE_PARAMETERS_NUMBER
        ),
        np.random.random(DUMMY_ELLIPSE_PARAMETERS_NUMBER * DUMMY_TRAIN_SIZE).reshape(
            DUMMY_TRAIN_SIZE, DUMMY_ELLIPSE_PARAMETERS_NUMBER
        ),
        np.random.random(DUMMY_MICROSCOPE_PARAMETERS_NUMBER * DUMMY_TEST_SIZE).reshape(
            DUMMY_TEST_SIZE, DUMMY_MICROSCOPE_PARAMETERS_NUMBER
        ),
        np.random.random(DUMMY_ELLIPSE_PARAMETERS_NUMBER * DUMMY_TEST_SIZE).reshape(
            DUMMY_TEST_SIZE, DUMMY_ELLIPSE_PARAMETERS_NUMBER
        ),
    )


def get_pretraining_data():
    return get_dummy_pretraining_data()


def get_training_data():
    return get_dummy_training_data()
