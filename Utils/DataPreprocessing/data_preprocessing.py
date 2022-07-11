from os.path import exists

import numpy as np
import pandas as pd
import yaml
from markdown.util import deprecated

from Utils.constants import DUMMY_MICROSCOPE_PARAMETERS_NUMBER, DUMMY_TRAIN_SIZE, \
    DUMMY_TEST_SIZE, MU_REDUCED_PARAMS, \
    MU_PARAM_LAST, \
    MU_PARAM_YAML_NAMES, MU_SPECIAL_PARAM_YAML_NAMES, MU_PARAM_SPECIAL_YAML_FUNCTIONS, MU_PARAM_NAMES, VERBOSE, \
    VERBOSE_INFO, EXPERIMENT_3, CONFIGURATIONS_TXT, CONFIGURATIONS_CSV, \
    DUMMY_ELLIPSE_PARAMETERS_NUMBER, VERBOSE_ALL, CONFIGURATIONS_YAML, RAW_CONFIGURATIONS_CSV, CURRENT_EXPERIMENT, \
    USER_ID, RAW_CONFIG, RAW_CONFIG_YAML_ONLY, RAW_CONFIG_CSV_AND_YAML, RAW_CONFIGURATIONS_INCOMPLETE_CSV, \
    MU_PARAM_YAML_FACTORS


# Parse a txt file to get a Dataframe
def parse_txt(experiment_name):
    configurations = []
    finished = False
    number_params = len(MU_REDUCED_PARAMS)
    with open(CONFIGURATIONS_TXT(experiment_name)) as file:
        while not finished:
            configuration = np.zeros(number_params)
            lines = [file.readline() for _ in MU_REDUCED_PARAMS]
            if '' in lines:
                assert lines == [''] * 9
                break
            for i, line in enumerate(lines):
                configuration[i] = float(line)
            configurations.append(configuration)
    df = pd.DataFrame(data=configurations, columns=[MU_PARAM_NAMES[index_p] for index_p in MU_REDUCED_PARAMS])
    return df


def get_csv_from_txt_and_one_yaml(experiment_name):
    df = parse_txt(experiment_name)
    with open(CONFIGURATIONS_YAML(experiment_name)) as file:
        default = parse_yaml_config(file)[0]
        df = add_default_configuration(df, default)
    df.to_csv(RAW_CONFIGURATIONS_CSV(experiment_name), index=False)


def get_csv_from_yaml(experiment_name):
    with open(CONFIGURATIONS_YAML(experiment_name)) as file:
        data = parse_yaml_config(file)
        df = pd.DataFrame(data, columns=MU_PARAM_NAMES)
    df.to_csv(RAW_CONFIGURATIONS_CSV(experiment_name), index=False)


def get_csv_from_incomplete_csv_and_yaml(experiment_name):
    df = pd.read_csv(RAW_CONFIGURATIONS_INCOMPLETE_CSV(experiment_name))
    with open(CONFIGURATIONS_YAML(experiment_name)) as file:
        default = parse_yaml_config(file)[0]
        df = add_default_configuration(df, default)
    df.to_csv(RAW_CONFIGURATIONS_CSV(experiment_name), index=False)


def add_default_configuration(df, default_configuration):
    names_not_in_df = [(i, name) for i, name in enumerate(MU_PARAM_NAMES) if name not in df.columns]

    df = pd.concat([
        df,
        pd.DataFrame(
            [[default_configuration[v[0]] for v in names_not_in_df]],
            index=df.index,
            columns=[v[1] for v in names_not_in_df]
        )
    ], axis=1)
    return df


# Parse a yaml file to form a numpy array
def parse_yaml_config(file):
    yaml_file = yaml.safe_load_all(file)
    configurations = []
    for yaml_conf in yaml_file:
        configuration = np.zeros(MU_PARAM_LAST)
        for key, value in yaml_conf[1].items():
            update_configuration_yaml(configuration, key, value, yaml_conf[1])
        configurations.append(configuration)

    return np.array(configurations)


# Parse single yaml configuration to a numpy format
def update_configuration_yaml(configuration, key, value, yaml_conf):
    if key in MU_PARAM_YAML_NAMES:
        index = MU_PARAM_YAML_NAMES.index(key)
        if VERBOSE & VERBOSE_INFO:
            print(f"{key} : {value} of type {type(value)}")
        if type(value) == list:
            for i, v in enumerate(value):
                configuration[index + i] = to_si(index+i, v)
        else:
            configuration[index] = to_si(index, value)

    elif key in MU_SPECIAL_PARAM_YAML_NAMES:
        special_index = MU_SPECIAL_PARAM_YAML_NAMES.index(key)
        real_value, index = MU_PARAM_SPECIAL_YAML_FUNCTIONS[special_index](value, yaml_conf)
        configuration[index] = real_value
    else:
        if VERBOSE & VERBOSE_INFO:
            print(f"{key} discarded")


# Modify value if needed to be in international metric system and equivalent s, m, A, V, kg, hz, m/s etc
def to_si(index, value):
    if index not in MU_PARAM_YAML_FACTORS:
        return value
    else:
        return value * MU_PARAM_YAML_FACTORS[index]


# Generate the cleaned configurations csv file
def generate_clean_configurations(experiment_name=CURRENT_EXPERIMENT, RAW_CONFIG_YAML_AND_TXT=None):
    # Generate the raw configurations.csv if not existing
    if not exists(RAW_CONFIGURATIONS_CSV(experiment_name)):
        if exists(CONFIGURATIONS_YAML(experiment_name)):
            if RAW_CONFIG == RAW_CONFIG_YAML_ONLY:
                get_csv_from_yaml(experiment_name)
            elif RAW_CONFIG == RAW_CONFIG_YAML_AND_TXT:
                get_csv_from_txt_and_one_yaml(experiment_name)
            elif RAW_CONFIG == RAW_CONFIG_CSV_AND_YAML:
                get_csv_from_incomplete_csv_and_yaml(experiment_name)
        else:
            print(
                f"No {CONFIGURATIONS_YAML(experiment_name)} or {RAW_CONFIGURATIONS_CSV(experiment_name)} files found"
            )
            raise FileNotFoundError

    # Duplicate it in the cleaned folder
    df = pd.read_csv(RAW_CONFIGURATIONS_CSV(experiment_name))
    df.to_csv(CONFIGURATIONS_CSV(experiment_name), index=False)


# Get the configuration if full is set to True then all fields are fetched else only those not in default
def get_cleaned_configuration(experiment_name=CURRENT_EXPERIMENT, user_id=USER_ID, session=None, full=True, parameters=MU_REDUCED_PARAMS):
    df = pd.read_csv(CONFIGURATIONS_CSV(experiment_name, user_id, session))
    if not full:
        # todo change to MU_USEFUL_PARAMS to get all the parameters or MU_NYTCHE_PARAM to get only those used by nytche
        df = df[parameters]
    return df


def get_dummy_pretraining_data():
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


def get_pretraining_data(filename):
    # TODO replace dummy
    return get_dummy_pretraining_data()


def get_training_data(filename):
    # TODO replace dummy
    return get_dummy_training_data()


# if __name__ == "__main__":
#     experiment = EXPERIMENT_3
#     parse_raw_configuration(experiment)
#     with open(DEFAULT_CONFIGURATION_CSV(experiment)) as file:
#         configurations = parse_yaml_config(file)
#         for i, v in enumerate(configurations[0]):
#             if v == 0:
#                 print(f"{MU_PARAM_NAMES[i]} is not set or null")
#     for i, attr_name in enumerate(MU_PARAM_NAMES):
#         print(f"{attr_name}\n{MU_PARAM_YAML_NAMES[i]}\n")
#     print(get_cleaned_configuration(experiment))
#     pass
