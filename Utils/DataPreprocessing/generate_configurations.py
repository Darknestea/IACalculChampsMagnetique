# Should make the generation of a configurations.csv according to user parameters

import pandas as pd
import numpy as np

from Utils.constants import CURRENT_EXPERIMENT, RAW_PATH, EXPERIMENT_TEST, RAW_CONFIGURATIONS_CSV, \
    RAW_CONFIGURATIONS_RANGE_CSV


# Generate at random a float array between min and max
def random_float_array(size, minima=0., maxima=1.):
    return np.random.random(size) * (maxima - minima) + minima


# Uses a range csv file (can be generated from an existing configurations.csv)
# to create a new configurations.csv for another experiment
def generate_configurations(nb_example, experiment_name=CURRENT_EXPERIMENT):
    configurations_range_df = pd.read_csv(RAW_CONFIGURATIONS_RANGE_CSV(experiment_name))
    data = np.zeros((nb_example, len(configurations_range_df.columns)))
    for i, column in enumerate(configurations_range_df.columns):
        minimum = configurations_range_df[column].iloc[0]
        maximum = configurations_range_df[column].iloc[1]
        data[:, i] = random_float_array(nb_example, minimum, maximum)

    df_configurations = pd.DataFrame(data=data, columns=configurations_range_df.columns)
    df_configurations.to_csv(RAW_CONFIGURATIONS_CSV(experiment_name), index=False)


# Generate a configurations_range.csv from an existing configurations.csv
# it contains for each parameter the min and the max to sample from
def generate_configuration_range_from_file(path, experiment_name=CURRENT_EXPERIMENT):
    df = pd.read_csv(path)
    data = np.zeros((2, len(df.columns)))
    for i, column in enumerate(df.columns):
        df_series = df[column]
        data[0, i] = min(df_series)
        data[1, i] = max(df_series)
    df_range = pd.DataFrame(data=data, columns=df.columns)
    df_range.to_csv(RAW_CONFIGURATIONS_RANGE_CSV(experiment_name), index=False)


if __name__ == '__main__':
    # generate_configuration_range_from_file(RAW_CONFIGURATIONS_CSV(EXPERIMENT_TEST))
    generate_configuration_range_from_file(RAW_PATH(CURRENT_EXPERIMENT) + "configurations_modified.csv")
    generate_configurations(100)
