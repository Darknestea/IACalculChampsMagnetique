from enum import Enum
from dataclasses import dataclass
import os

# Parameters

# To modify

VERBOSE_NONE = 0
VERBOSE_DEBUG = 1
VERBOSE_INFO = 2
VERBOSE_ALL = 3

VERBOSE = VERBOSE_NONE

MODE_NO_TRAINING = 0
MODE_PRETRAINING = 1
MODE_REAL_TRAINING = 2
MODE_TRAINING_FULL = 3

MODE = MODE_TRAINING_FULL

# Default

SAVE_PATH = os.path.dirname(os.path.realpath(__file__)) + "\\"

SAVE_DATA_PATH = SAVE_PATH + "Data\\"
SAVE_RAW_CONFIGURATION_PATH = SAVE_DATA_PATH + "Raw\\Configurations\\Raw\\"
SAVE_YAML_CONFIGURATION_PATH = SAVE_DATA_PATH + "Raw\\Configurations\\Yaml\\"
SAVE_CLEAN_CONFIGURATION_PATH = SAVE_DATA_PATH + "Cleaned\\Configurations\\"
SAVE_CLEAN_SIMULATION_DATA_PATH = SAVE_DATA_PATH + "Cleaned\\SimulationImages\\"

SAVE_MODEL_PATH = SAVE_PATH + "Models\\"
IMAGE_TO_ELLIPSE_MODEL_PATH = SAVE_MODEL_PATH + "image_to_ellipse\\"

ELLIPSES_PATH = "F:\\DaVinci\\Images"

SUFFIX_CLEAN_CONFIGURATIONS_DF = '_configurations.csv'
SUFFIX_CLEAN_CONFIGURATIONS_DEFAULT = '_default.yaml'

PRETRAINING_DATASET = None
TRAINING_DATASET = None
PRETRAINED_MODEL_NAME = None
TRAINED_MODEL_NAME = None

THRESHOLD_ELLIPSE_DETECTION = 40

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

APERTURE_OPEN_DIAMETER = 1.

SCREEN_POSITION = 2052.5
SCREEN_POSITION_CORRECTED = SCREEN_POSITION + 433.45
SCREEN_SIZE = 32
SCREEN_DPI = 8
SCREEN_WIDTH = SCREEN_SIZE * SCREEN_DPI
SCREEN_SHAPE = (SCREEN_WIDTH, SCREEN_WIDTH)
SCREEN_CENTER = (SCREEN_WIDTH/2, SCREEN_WIDTH/2)

CAMERA_POSITION = 2150
CAMERA_POSITION_CORRECTED = CAMERA_POSITION + 433.45


ELLIPSE_PARAMETER_NAMES = [
    "X",
    "Y",
    "A",
    "B",
    "THETA"
]


RUN_SESSION = None
EXTENSION = None

MU_PARAM_N_THETA =      0
MU_PARAM_N_PHI =        1

MU_PARAM_V0 =           2
MU_PARAM_V1 =           3
MU_PARAM_RATIO =        4

MU_PARAM_C1 =           5
MU_PARAM_C2 =           6
MU_PARAM_C3 =           7

MU_PARAM_I1 =           8
MU_PARAM_I2 =           9
MU_PARAM_I3 =           10

MU_PARAM_P1 =           11
MU_PARAM_P2 =           12

MU_PARAM_OBJ =          13

MU_PARAM_LENSES = range(MU_PARAM_C1, MU_PARAM_OBJ+1)

MU_PARAM_BPC =          14
MU_PARAM_BPC_X =        15
MU_PARAM_BPC_Y =        16
MU_PARAM_BPC_A =        17
MU_PARAM_BPC_V =        18
MU_PARAM_BPC_P =        19

MU_PARAM_BP1 =          20
MU_PARAM_BP1_X =        21
MU_PARAM_BP1_Y =        22
MU_PARAM_BP1_A =        23
MU_PARAM_BP1_V =        24
MU_PARAM_BP1_P =        25

MU_PARAM_BP2 =          26
MU_PARAM_BP2_X =        27
MU_PARAM_BP2_Y =        28
MU_PARAM_BP2_A =        29
MU_PARAM_BP2_V =        30
MU_PARAM_BP2_P =        31


MU_PARAM_BIPRISMS = range(MU_PARAM_BPC, MU_PARAM_BP2_P+1, 6)

MU_PARAM_APP_CO =       32
MU_PARAM_APP_CO_X =     33
MU_PARAM_APP_CO_Y =     34

MU_PARAM_APP_ST =       35
MU_PARAM_APP_ST_X =     36
MU_PARAM_APP_ST_Y =     37

MU_PARAM_APP_OB =       38
MU_PARAM_APP_OB_X =     39
MU_PARAM_APP_OB_Y =     40

MU_PARAM_APP_SA =       41
MU_PARAM_APP_SA_X =     42
MU_PARAM_APP_SA_Y =     43

MU_PARAM_APERTURES = range(MU_PARAM_APP_CO, MU_PARAM_APP_SA_Y+1, 3)

MU_PARAM_DEF_GH_X =     44
MU_PARAM_DEF_GH_Y =     45
MU_PARAM_DEF_GHX_R =    46
MU_PARAM_DEF_GHX_Z =    47
MU_PARAM_DEF_GHY_R =    48
MU_PARAM_DEF_GHY_Z =    49

MU_PARAM_DEF_GT_X =     50
MU_PARAM_DEF_GT_Y =     51
MU_PARAM_DEF_GTX_R =    52
MU_PARAM_DEF_GTX_Z =    53
MU_PARAM_DEF_GTY_R =    54
MU_PARAM_DEF_GTY_Z =    55

MU_PARAM_DEF_GA_X =     56
MU_PARAM_DEF_GA_Y =     57

MU_PARAM_DEF_CS3_X =    58
MU_PARAM_DEF_CS3_Y =    59
MU_PARAM_DEF_CS3X_R =   60
MU_PARAM_DEF_CS3X_Z =   61
MU_PARAM_DEF_CS3Y_R =   62
MU_PARAM_DEF_CS3Y_Z =   63

MU_PARAM_DEF_CS_X =     64
MU_PARAM_DEF_CS_Y =     65
MU_PARAM_DEF_CSX_R =    66
MU_PARAM_DEF_CSX_Z =    67
MU_PARAM_DEF_CSY_R =    68
MU_PARAM_DEF_CSY_Z =    69

MU_PARAM_DEF_BH_X =     70
MU_PARAM_DEF_BH_Y =     71
MU_PARAM_DEF_BHU_X =    72
MU_PARAM_DEF_BHU_Y =    73
MU_PARAM_DEF_BHX_R =    74
MU_PARAM_DEF_BHX_Z =    75
MU_PARAM_DEF_BHY_R =    76
MU_PARAM_DEF_BHY_Z =    77

MU_PARAM_DEF_BT_X =     78
MU_PARAM_DEF_BT_Y =     79
MU_PARAM_DEF_BTU_X =    80
MU_PARAM_DEF_BTU_Y =    81
MU_PARAM_DEF_BTX_R =    82
MU_PARAM_DEF_BTX_Z =    83
MU_PARAM_DEF_BTY_R =    84
MU_PARAM_DEF_BTY_Z =    85

MU_PARAM_DEF_OS_X =     86
MU_PARAM_DEF_OS_Y =     87
MU_PARAM_DEF_OSX_R =    88
MU_PARAM_DEF_OSX_Z =    89
MU_PARAM_DEF_OSY_R =    90
MU_PARAM_DEF_OSY_Z =    91

MU_PARAM_DEF_OSP_X =    92
MU_PARAM_DEF_OSP_Y =    93

MU_PARAM_DEF_ISF_X =    94
MU_PARAM_DEF_ISF_Y =    95
MU_PARAM_DEF_ISFX_R =   96
MU_PARAM_DEF_ISFX_Z =   97
MU_PARAM_DEF_ISFY_R =   98
MU_PARAM_DEF_ISFY_Z =   99

MU_PARAM_DEF_IS_X =     100
MU_PARAM_DEF_IS_Y =     101

MU_PARAM_DEF_IA_X =     102
MU_PARAM_DEF_IA_Y =     103

MU_PARAM_DEF_PA_X =     104
MU_PARAM_DEF_PA_Y =     105

MU_PARAM_DEFLECTORS = range(MU_PARAM_DEF_GH_X, MU_PARAM_DEF_PA_Y+1, 2)

MU_PARAM_SAMPLE =       106

MU_PARAM_REF_PLANE =    107

MU_PARAM_MAG =          108

MU_PARAM_LENS_MODE =    109

MU_PARAM_PROB_MODE =    110

MU_EMISSION_CURRENT =   111

MU_PARAM_LAST =         112


# Offsets

MU_OFFSET_BP = 0
MU_OFFSET_BP_X = 1
MU_OFFSET_BP_Y = 2
MU_OFFSET_BP_A = 3
MU_OFFSET_BP_V = 4
MU_OFFSET_BP_P = 5
MU_OFFSET_BP_LAST = 6

MU_OFFSET_APP = 0
MU_OFFSET_APP_X = 1
MU_OFFSET_APP_Y = 2
MU_OFFSET_AP_LAST = 3


MU_PARAMS = range(MU_PARAM_LAST)
MU_USEFUL_PARAMS = [i for i in range(MU_PARAM_V0, MU_PARAM_OBJ+1)]
MU_USEFUL_PARAMS += [i for i in range(MU_PARAM_APP_CO, MU_PARAM_SAMPLE)]
MU_REDUCED_PARAMS = [
        MU_PARAM_C1,
        MU_PARAM_C2,
        MU_PARAM_C3,
        MU_PARAM_OBJ,
        MU_PARAM_I1,
        MU_PARAM_I2,
        MU_PARAM_I3,
        MU_PARAM_P1,
        MU_PARAM_P2
    ]
MU_NYTCHE_PARAMS = [
        MU_PARAM_V0,
        MU_PARAM_C1,
        MU_PARAM_C2,
        MU_PARAM_C3,
        MU_PARAM_OBJ,
        MU_PARAM_I1,
        MU_PARAM_I2,
        MU_PARAM_I3,
        MU_PARAM_P1,
        MU_PARAM_P2,
        MU_PARAM_BPC_V,
        MU_PARAM_BPC_P,
        MU_PARAM_BP1_V,
        MU_PARAM_BP1_P,
        MU_PARAM_BP2_V,
        MU_PARAM_BP2_P,
        MU_PARAM_APP_OB,
        MU_PARAM_APP_SA,
        MU_PARAM_APP_CO,
        MU_PARAM_APP_ST
    ]

MU_PARAM_NAMES = [
    "MU_PARAM_PHI",
    "MU_PARAM_THETA",

    "MU_PARAM_V0",
    "MU_PARAM_V1",
    "MU_PARAM_RATIO",

    "MU_PARAM_C1",
    "MU_PARAM_C2",
    "MU_PARAM_C3",

    "MU_PARAM_I1",
    "MU_PARAM_I2",
    "MU_PARAM_I3",

    "MU_PARAM_P1",
    "MU_PARAM_P2",

    "MU_PARAM_OBJ",

    "MU_PARAM_BPC",
    "MU_PARAM_BPC_X",
    "MU_PARAM_BPC_Y",
    "MU_PARAM_BPC_A",
    "MU_PARAM_BPC_V",
    "MU_PARAM_BPC_P",

    "MU_PARAM_BP1",
    "MU_PARAM_BP1_X",
    "MU_PARAM_BP1_Y",
    "MU_PARAM_BP1_A",
    "MU_PARAM_BP1_V",
    "MU_PARAM_BP1_P",

    "MU_PARAM_BP2",
    "MU_PARAM_BP2_X",
    "MU_PARAM_BP2_Y",
    "MU_PARAM_BP2_A",
    "MU_PARAM_BP2_V",
    "MU_PARAM_BP2_P",

    "MU_PARAM_APP_CO",
    "MU_PARAM_APP_CO_X",
    "MU_PARAM_APP_CO_Y",

    "MU_PARAM_APP_ST",
    "MU_PARAM_APP_ST_X",
    "MU_PARAM_APP_ST_Y",

    "MU_PARAM_APP_OB",
    "MU_PARAM_APP_OB_X",
    "MU_PARAM_APP_OB_Y",

    "MU_PARAM_APP_SA",
    "MU_PARAM_APP_SA_X",
    "MU_PARAM_APP_SA_Y",

    "MU_PARAM_DEF_GH_X",
    "MU_PARAM_DEF_GH_Y",
    "MU_PARAM_DEF_GHX_R", 
    "MU_PARAM_DEF_GHX_Z", 
    "MU_PARAM_DEF_GHY_R", 
    "MU_PARAM_DEF_GHY_Z",

    "MU_PARAM_DEF_GT_X", 
    "MU_PARAM_DEF_GT_Y", 
    "MU_PARAM_DEF_GTX_R", 
    "MU_PARAM_DEF_GTX_Z", 
    "MU_PARAM_DEF_GTY_R", 
    "MU_PARAM_DEF_GTY_Z", 
    "MU_PARAM_DEF_GA_X", 
    "MU_PARAM_DEF_GA_Y", 

    "MU_PARAM_DEF_CS3_X", 
    "MU_PARAM_DEF_CS3_Y", 
    "MU_PARAM_DEF_CS3X_R", 
    "MU_PARAM_DEF_CS3X_Z", 
    "MU_PARAM_DEF_CS3Y_R", 
    "MU_PARAM_DEF_CS3Y_Z", 
    
    "MU_PARAM_DEF_CS_X", 
    "MU_PARAM_DEF_CS_Y", 
    "MU_PARAM_DEF_CSX_R", 
    "MU_PARAM_DEF_CSX_Z",
    "MU_PARAM_DEF_CSY_R", 
    "MU_PARAM_DEF_CSY_Z", 
    
    "MU_PARAM_DEF_BH_X", 
    "MU_PARAM_DEF_BH_Y", 
    "MU_PARAM_DEF_BHU_X", 
    "MU_PARAM_DEF_BHU_Y", 
    "MU_PARAM_DEF_BHX_R", 
    "MU_PARAM_DEF_BHX_Z", 
    "MU_PARAM_DEF_BHY_R", 
    "MU_PARAM_DEF_BHY_Z", 
    
    "MU_PARAM_DEF_BT_X", 
    "MU_PARAM_DEF_BT_Y", 
    "MU_PARAM_DEF_BTU_X", 
    "MU_PARAM_DEF_BTU_Y", 
    "MU_PARAM_DEF_BTX_R", 
    "MU_PARAM_DEF_BTX_Z", 
    "MU_PARAM_DEF_BTY_R",
    "MU_PARAM_DEF_BTY_Z",
    
    "MU_PARAM_DEF_OS_X",
    "MU_PARAM_DEF_OS_Y", 
    "MU_PARAM_DEF_OSX_R",
    "MU_PARAM_DEF_OSX_Z",
    "MU_PARAM_DEF_OSY_R",
    "MU_PARAM_DEF_OSY_Z",
    
    "MU_PARAM_DEF_OSP_X",
    "MU_PARAM_DEF_OSP_Y",
    
    "MU_PARAM_DEF_ISF_X",
    "MU_PARAM_DEF_ISF_Y",
    "MU_PARAM_DEF_ISFX_R",
    "MU_PARAM_DEF_ISFX_Z",
    "MU_PARAM_DEF_ISFY_R",
    "MU_PARAM_DEF_ISFY_Z",
    
    "MU_PARAM_DEF_IS_X",
    "MU_PARAM_DEF_IS_Y",
    
    "MU_PARAM_DEF_IA_X",
    "MU_PARAM_DEF_IA_Y",
    
    "MU_PARAM_DEF_PA_X",
    "MU_PARAM_DEF_PA_Y",

    "MU_PARAM_SAMPLE",

    "MU_PARAM_REF_PLANE",

    "MU_PARAM_MAG",

    "MU_PARAM_LENS_MODE",

    "MU_PARAM_PROB_MODE",

    "MU_EMISSION_CURRENT"
]

# MU_PARAM_RANGES = [
#     (0.00665630645849, 0.00665630645849),
#     (-0.6, -0.6),
#
#     (60000, 300000),
#     "MU_PARAM_V1",
#     "MU_PARAM_RATIO",
#
#     "MU_PARAM_C1",
#     "MU_PARAM_C2",
#     "MU_PARAM_C3",
#
#     "MU_PARAM_I1",
#     "MU_PARAM_I2",
#     "MU_PARAM_I3",
#
#     "MU_PARAM_P1",
#     "MU_PARAM_P2",
#
#     "MU_PARAM_OBJ",
#
#     "MU_PARAM_BPC",
#     "MU_PARAM_BPC_X",
#     "MU_PARAM_BPC_Y",
#     "MU_PARAM_BPC_A",
#
#     "MU_PARAM_BP1",
#     "MU_PARAM_BP1_X",
#     "MU_PARAM_BP1_Y",
#     "MU_PARAM_BP1_A",
#
#     "MU_PARAM_BP2",
#     "MU_PARAM_BP2_X",
#     "MU_PARAM_BP2_Y",
#     "MU_PARAM_BP2_A",
#
#     "MU_PARAM_APP_CO",
#     "MU_PARAM_APP_CO_X",
#     "MU_PARAM_APP_CO_Y",
#
#     "MU_PARAM_APP_ST",
#     "MU_PARAM_APP_ST_X",
#     "MU_PARAM_APP_ST_Y",
#
#     "MU_PARAM_APP_OB",
#     "MU_PARAM_APP_OB_X",
#     "MU_PARAM_APP_OB_Y",
#
#     "MU_PARAM_APP_SA",
#     "MU_PARAM_APP_SA_X",
#     "MU_PARAM_APP_SA_Y",
#
#     "MU_PARAM_SAMPLE",
#
#     "MU_PARAM_DEF_GH_X",
#     "MU_PARAM_DEF_GH_Y",
#     "MU_PARAM_DEF_GHX_R",
#     "MU_PARAM_DEF_GHX_Z",
#     "MU_PARAM_DEF_GHY_R",
#     "MU_PARAM_DEF_GHY_Z",
#
#     "MU_PARAM_DEF_GT_X",
#     "MU_PARAM_DEF_GT_Y",
#     "MU_PARAM_DEF_GTX_R",
#     "MU_PARAM_DEF_GTX_Z",
#     "MU_PARAM_DEF_GTY_R",
#     "MU_PARAM_DEF_GTY_Z",
#     "MU_PARAM_DEF_GA_X",
#     "MU_PARAM_DEF_GA_Y",
#
#     "MU_PARAM_DEF_CS3_X",
#     "MU_PARAM_DEF_CS3_Y",
#     "MU_PARAM_DEF_CS3X_R",
#     "MU_PARAM_DEF_CS3X_Z",
#     "MU_PARAM_DEF_CS3Y_R",
#     "MU_PARAM_DEF_CS3Y_Z",
#
#     "MU_PARAM_DEF_CS_X",
#     "MU_PARAM_DEF_CS_Y",
#     "MU_PARAM_DEF_CSX_R",
#     "MU_PARAM_DEF_CSX_Z",
#     "MU_PARAM_DEF_CSY_R",
#     "MU_PARAM_DEF_CSY_Z",
#
#     "MU_PARAM_DEF_BH_X",
#     "MU_PARAM_DEF_BH_Y",
#     "MU_PARAM_DEF_BHU_X",
#     "MU_PARAM_DEF_BHU_Y",
#     "MU_PARAM_DEF_BHX_R",
#     "MU_PARAM_DEF_BHX_Z",
#     "MU_PARAM_DEF_BHY_R",
#     "MU_PARAM_DEF_BHY_Z",
#
#     "MU_PARAM_DEF_BT_X",
#     "MU_PARAM_DEF_BT_Y",
#     "MU_PARAM_DEF_BTU_X",
#     "MU_PARAM_DEF_BTU_Y",
#     "MU_PARAM_DEF_BTX_R",
#     "MU_PARAM_DEF_BTX_Z",
#     "MU_PARAM_DEF_BTY_R",
#     "MU_PARAM_DEF_BTY_Z",
#
#     "MU_PARAM_DEF_OS_X",
#     "MU_PARAM_DEF_OS_Y",
#     "MU_PARAM_DEF_OSX_R",
#     "MU_PARAM_DEF_OSX_Z",
#     "MU_PARAM_DEF_OSY_R",
#     "MU_PARAM_DEF_OSY_Z",
#
#     "MU_PARAM_DEF_OSP_X",
#     "MU_PARAM_DEF_OSP_Y",
#
#     "MU_PARAM_DEF_ISF_X",
#     "MU_PARAM_DEF_ISF_Y",
#     "MU_PARAM_DEF_ISFX_R",
#     "MU_PARAM_DEF_ISFX_Z",
#     "MU_PARAM_DEF_ISFY_R",
#     "MU_PARAM_DEF_ISFY_Z",
#
#     "MU_PARAM_DEF_IS_X",
#     "MU_PARAM_DEF_IS_Y",
#
#     "MU_PARAM_DEF_IA_X",
#     "MU_PARAM_DEF_IA_Y",
#
#     "MU_PARAM_DEF_PA_X",
#     "MU_PARAM_DEF_PA_Y",
#
#     "MU_PARAM_REF_PLANE",
# ]

MU_PARAM_YAML_NAMES = [
    "MU_PARAM_PHI",
    "MU_PARAM_THETA",

    "MU_PARAM_V0",
    "MU_PARAM_V1",
    "MU_PARAM_RATIO",

    "c1",
    "c2",
    "c3",

    "i1",
    "i2",
    "i3",

    "p1",
    "p2",

    "obj",


    "BPC",
    "bpc_x",
    "bpc_y",
    "bpc_rot",
    "bpc_voltage",
    "bpc_power",

    "BP1",
    "bp1_x",
    "bp1_y",
    "bp1_rot",
    "bp1_voltage",
    "bp1_power",

    "BP2",
    "bp2_x",
    "bp2_y",
    "bp2_rot",
    "bp2_voltage",
    "bp2_power",

    "Cond Ap",
    "cond_ap_x",
    "cond_ap_y",

    "STEM Ap",
    "stem_ap_x",
    "stem_ap_y",

    "Obj Ap",
    "obj_ap_x",
    "obj_ap_y",

    "SA Ap",
    "sa_ap_x",
    "sa_ap_y",

    "gh",
    "MU_PARAM_DEF_GH_Y",
    "ghx",
    "MU_PARAM_DEF_GHX_Z",
    "ghy",
    "MU_PARAM_DEF_GHY_Z",

    "gt",
    "MU_PARAM_DEF_GT_Y",
    "gtx",
    "MU_PARAM_DEF_GTX_Z",
    "gty",
    "MU_PARAM_DEF_GTY_Z",
    "ga",
    "MU_PARAM_DEF_GA_Y",

    "cs3",
    "MU_PARAM_DEF_CS3_Y",
    "cs3x",
    "MU_PARAM_DEF_CS3X_Z",
    "cs3y",
    "MU_PARAM_DEF_CS3Y_Z",

    "cs",
    "MU_PARAM_DEF_CS_Y",
    "csx",
    "MU_PARAM_DEF_CSX_Z",
    "csy",
    "MU_PARAM_DEF_CSY_Z",

    "bh",
    "MU_PARAM_DEF_BH_Y",
    "bhu",
    "MU_PARAM_DEF_BHU_Y",
    "bhx",
    "MU_PARAM_DEF_BHX_Z",
    "bhy",
    "MU_PARAM_DEF_BHY_Z",

    "bt",
    "MU_PARAM_DEF_BT_Y",
    "btu",
    "MU_PARAM_DEF_BTU_Y",
    "btx",
    "MU_PARAM_DEF_BTX_Z",
    "bty",
    "MU_PARAM_DEF_BTY_Z",

    "os",
    "MU_PARAM_DEF_OS_Y",
    "osx",
    "MU_PARAM_DEF_OSX_Z",
    "osy",
    "MU_PARAM_DEF_OSY_Z",

    "osp",
    "MU_PARAM_DEF_OSP_Y",

    "isf",
    "MU_PARAM_DEF_ISF_Y",
    "isfx",
    "MU_PARAM_DEF_ISFX_Z",
    "isfy",
    "MU_PARAM_DEF_ISFY_Z",

    "is",
    "MU_PARAM_DEF_IS_Y",

    "ia",
    "MU_PARAM_DEF_IA_Y",

    "pa",
    "MU_PARAM_DEF_PA_Y",

    "MU_PARAM_SAMPLE",

    "screen_pos",

    "mag",

    "lens_mode",

    "probe_mode",

    "emission_current"
]

MU_SPECIAL_PARAM_ACC_VOLTAGE = 0
MU_SPECIAL_PARAM_EXT_VOLTAGE = 1
MU_SPECIAL_PARAM_CTRL_VOLTAGE = 2

MU_SPECIAL_PARAM_YAML_NAMES = [
    "acc_voltage",
    "extractor_voltage",
    "control_voltage"
]

MU_PARAM_SPECIAL_YAML_FUNCTIONS = [
    lambda x, file: (x*1000, MU_PARAM_V0),
    lambda x, file: (x*1000, MU_PARAM_V1),
    lambda x, file: (x/file["extractor_voltage"], MU_PARAM_RATIO)
]


def run_session():
    return RUN_SESSION


def current_extension():
    return EXTENSION


def set_run_session(extension, filename=SAVE_DATA_PATH + "run_counter.dat"):
    global EXTENSION
    EXTENSION = extension
    with open(filename, "a+") as f:
        f.seek(0)
        global RUN_SESSION
        RUN_SESSION = int(f.read() or 0) + 1
        f.seek(0)
        f.truncate()
        f.write(str(RUN_SESSION))

