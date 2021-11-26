from enum import Enum
from dataclasses import dataclass

DUMMY_TRAIN_SIZE = 200
DUMMY_TEST_SIZE = 50

DUMMY_MICROSCOPE_PARAMETERS_NUMBER = 4
DUMMY_MAGNETIC_FIELD_SIZE = 8

POS_X = 0
POS_Y = 1
POS_A = 2
POS_B = 3
POS_THETA = 4
DUMMY_ELLIPSE_PARAMETERS_NUMBER = 5

MODE_NO_TRAINING = 0
MODE_PRETRAINING = 1
MODE_REAL_TRAINING = 2
MODE_TRAINING_FULL = 3