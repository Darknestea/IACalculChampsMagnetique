from enum import Enum
from dataclasses import dataclass
import os

# Parameters
SAVE_PATH = os.path.dirname(os.path.realpath(__file__)) + "\\Models\\"
PRETRAINING_DATASET = None
TRAINING_DATASET = None
PRETRAINED_MODEL_NAME = None
TRAINED_MODEL_NAME = None

# Constants
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
MODE = MODE_TRAINING_FULL


RUN_SESSION = None
EXTENSION = None

MU_PARAM_N_THETA = 0
MU_PARAM_N_PHI = 1
MU_PARAM_V0 = 2
MU_PARAM_V1 = 3
MU_PARAM_RATIO = 4
MU_PARAM_C1 = 5
MU_PARAM_C2 = 6
MU_PARAM_C3 = 7
MU_PARAM_I1 = 8
MU_PARAM_I2 = 9
MU_PARAM_I3 = 10
MU_PARAM_P1 = 11
MU_PARAM_P2 = 12
MU_PARAM_OBJ = 14
MU_PARAM_BPC = 15
MU_PARAM_BP1 = 16
MU_PARAM_BP2 = 17
MU_PARAM_MODE = 18
MU_PARAM_SCREEN = 19


def run_session():
    return RUN_SESSION


def current_extension():
    return EXTENSION


def set_run_session(extension, filename=SAVE_PATH + "run_counter.dat"):
    global EXTENSION
    EXTENSION = extension
    with open(filename, "a+") as f:
        f.seek(0)
        global RUN_SESSION
        RUN_SESSION = int(f.read() or 0) + 1
        f.seek(0)
        f.truncate()
        f.write(str(RUN_SESSION))

