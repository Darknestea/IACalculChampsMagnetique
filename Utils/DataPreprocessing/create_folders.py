from os import mkdir
from os.path import exists

from Utils.constants import RAW_PATH, RAW_REAL_BEAM_SLICES_PATH, RAW_SIMULATED_BEAM_SLICES_PATH, RAW_INFO_TXT, \
    EXPERIMENT_FROM_CONFIGURATION, EXPERIMENT_FROM_GENERATED_ELLIPSES, RAW_GENERATED_ELLIPSES_PATH, CONFIGURATIONS_TXT, \
    CONFIGURATIONS_YAML, CLEANED_PATH, ELLIPSES_PARAMETERS_PATH, VERBOSE_NONE, VERBOSE, CURRENT_EXPERIMENT, MODEL_PATH, \
    RESULTS_PATH, DATA_PATH, MODEL_INFORMATION, MODEL

# Small wrapper for mkdir
from Utils.utils import main_specific_tasks


def create_folder(path):
    if not exists(path):
        mkdir(path)
    else:
        if VERBOSE != VERBOSE_NONE:
            print(f"{path} already exists")


# todo Generate an info file
def generate_info_txt(path):
    return


# Create the raw folder hierarchy for the given experiment
def create_raw_experiment_folders(name=CURRENT_EXPERIMENT, experiment_type=EXPERIMENT_FROM_CONFIGURATION):
    create_folder(DATA_PATH())
    create_folder(RAW_PATH(name))
    generate_info_txt(RAW_INFO_TXT(name))

    if experiment_type == EXPERIMENT_FROM_CONFIGURATION:
        create_folder(RAW_REAL_BEAM_SLICES_PATH(name))
        print(f"Experiment {name} raw data folder created, please set up:")
        # todo replace by csv/yaml
        print(f"\t- {CONFIGURATIONS_TXT(name)}")
        print(f"\t- {CONFIGURATIONS_YAML(name)}")

    elif experiment_type == EXPERIMENT_FROM_GENERATED_ELLIPSES:
        create_folder(RAW_GENERATED_ELLIPSES_PATH(name))
        print(f"Experiment {name} raw data folder created")


# Create the cleaned folder hierarchy for the given experiment as well as the raw folder for the simulated slice
# The last part is needed since it is Nytche dependent and newer version might change the results
def create_cleaned_experiment_folders(name=CURRENT_EXPERIMENT):
    create_folder(DATA_PATH())
    create_folder(CLEANED_PATH(name))
    create_folder(RAW_SIMULATED_BEAM_SLICES_PATH(name))
    create_folder(ELLIPSES_PARAMETERS_PATH(name))


def create_model_folder(model):
    create_folder(MODEL_PATH(model.name))
    model_path = MODEL(model.name)
    model_information_path = MODEL_INFORMATION(model.name)
    return model_path, model_information_path


if __name__ == '__main__':
    create_raw_experiment_folders()
