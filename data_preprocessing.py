from typing import Tuple, List

import numpy as np
import pandas as pd
import yaml

from constants import DUMMY_MICROSCOPE_PARAMETERS_NUMBER, DUMMY_TRAIN_SIZE, \
    DUMMY_ELLIPSE_PARAMETERS_NUMBER, DUMMY_TEST_SIZE, MU_PARAMS, MU_REDUCED_PARAMS, SUFFIX_CLEAN_CONFIGURATIONS_DF, \
    SAVE_RAW_CONFIGURATION_PATH, SAVE_CLEAN_CONFIGURATION_PATH, SUFFIX_CLEAN_CONFIGURATIONS_DEFAULT, MU_PARAM_LAST, \
    MU_PARAM_YAML_NAMES, MU_PARAM_SPECIAL_YAML_NAMES, MU_PARAM_SPECIAL_YAML_FUNCTIONS, MU_PARAM_NAMES, VERBOSE, \
    VERBOSE_INFO


def parse_raw_configuration(filename, path=SAVE_RAW_CONFIGURATION_PATH, save_path=SAVE_CLEAN_CONFIGURATION_PATH):
    configurations = []
    finished = False

    number_params = len(MU_REDUCED_PARAMS)

    with open(path + filename) as file:
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
    df.to_csv(save_path + filename.rsplit('.', 1)[0] + SUFFIX_CLEAN_CONFIGURATIONS_DF, index=False)


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


def parse_yaml_config(file):
    yaml_file = yaml.safe_load(file)
    configurations = []
    for yaml_conf in yaml_file:
        configuration = np.zeros(MU_PARAM_LAST)
        for key, value in yaml_conf.items():
            update_configuration_yaml(configuration, key, value, yaml_conf)
        configurations.append(configuration)

    return np.array(configurations)


def update_configuration_yaml(configuration, key, value, yaml_conf):
    if key in MU_PARAM_YAML_NAMES:
        index = MU_PARAM_YAML_NAMES.index(key)
        if VERBOSE | VERBOSE_INFO:
            print(f"{key} : {value} of type {type(value)}")
        if type(value) == list:
            for i, v in enumerate(value):
                configuration[index + i] = v
        else:
            configuration[index] = value
    elif key in MU_PARAM_SPECIAL_YAML_NAMES:
        special_index = MU_PARAM_SPECIAL_YAML_NAMES.index(key)
        real_value, index = MU_PARAM_SPECIAL_YAML_FUNCTIONS[special_index](value, yaml_conf)
        configuration[index] = real_value
    else:
        if VERBOSE | VERBOSE_INFO:
            print(f"{key} discarded")


def get_cleaned_configuration(filename, save_path=SAVE_CLEAN_CONFIGURATION_PATH, full=True):
    df = pd.read_csv(save_path + filename.rsplit('.', 1)[0] + SUFFIX_CLEAN_CONFIGURATIONS_DF)
    with open(save_path + filename.rsplit('.', 1)[0] + SUFFIX_CLEAN_CONFIGURATIONS_DEFAULT) as file:
        default = parse_yaml_config(file)[0]
        if full:
            df = add_default_configuration(df, default)
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


if __name__ == "__main__":
    # parse_raw_configuration("Exp3log.txt")
    # with open(SAVE_CLEAN_CONFIGURATION_PATH + "Exp3log" + SUFFIX_CLEAN_CONFIGURATIONS_DEFAULT) as file:
    #     configurations = parse_yaml_config(file)
    #     for i, v in enumerate(configurations[0]):
    #         if v == 0:
    #             print(f"{MU_PARAM_NAMES[i]} is not set or null")
    # for i, name in enumerate(MU_PARAM_NAMES):
    #     print(f"{name}\n{MU_PARAM_YAML_NAMES[i]}\n")
    print(get_cleaned_configuration("Exp3log.txt"))
    pass
