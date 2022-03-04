from Utils.DataPreprocessing.create_folders import create_cleaned_experiment_folders
from Utils.DataPreprocessing.data_preprocessing import generate_clean_configurations
from Utils.DataPreprocessing.generate_real_dataset import clean_real_dataset
from Utils.DataPreprocessing.generate_simulation_datasets import clean_simulation_dataset
from Utils.constants import CLEANING_OPERATIONS_TO_PERFORM, PERFORM_SIMULATION, PERFORM_REAL, \
    PERFORM_ELLIPSE_PARAMETERS_EXTRACTION, CURRENT_EXPERIMENT
from Utils.utils import main_specific_tasks

if __name__ == "__main__":
    main_specific_tasks()
    experiment = CURRENT_EXPERIMENT

    create_cleaned_experiment_folders()

    generate_clean_configurations()

    do_ellipse_parameter_extraction = CLEANING_OPERATIONS_TO_PERFORM & PERFORM_ELLIPSE_PARAMETERS_EXTRACTION

    if CLEANING_OPERATIONS_TO_PERFORM & PERFORM_SIMULATION:
        clean_simulation_dataset(do_ellipse_parameter_extraction=do_ellipse_parameter_extraction)

    if CLEANING_OPERATIONS_TO_PERFORM & PERFORM_REAL:
        clean_real_dataset(do_ellipse_parameter_extraction=do_ellipse_parameter_extraction)



# if MODE | MODE_PRETRAINING:
#     # Do pretraining from Command to field
#     pass
# else:
#     pretrained_model = load_model(ROOT_PATH, PRETRAINED_MODEL_NAME)
#
# if MODE | MODE_REAL_TRAINING:
#     x_train, y_train, x_test, y_test = get_training_data(TRAINING_DATASET)
#     trained_model = train(pretrained_model, x_train, y_train)
#     save_model(trained_model, ROOT_PATH)
#     record_trained_model(trained_model, x_test, y_test)
#
# else:
#     trained_model = load_model(ROOT_PATH, TRAINED_MODEL_NAME)

