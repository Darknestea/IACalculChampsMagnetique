import random
import numpy as np

from Training.ConfigurationToBeam.Models.simple_deconvnet import simple_deconvolution
from Training.ConfigurationToBeam.Models.simple_sequential import simple_sequential
from Training.ConfigurationToBeam.train import load_training_config, load_training_images
from Utils.constants import MODEL, SUMMARY_FIELD, HISTORY_FIELD, EXPERIMENT_TEST2, IMAGES_SHAPE_FIELD, EXPERIMENT_TEST
from Utils.model_loader import load_model
from Utils.record_model_performances import evaluate_model
from Utils.utils import get_model_name, main_specific_tasks
from my_id import MY_ID


def execute_model(
        experiment_name, user_id, session, model, images_shape, use_real_images,
        number_samples=10, random_state=None, record_images=False):

    x, columns = load_training_config(experiment_name, user_id, session)
    y = load_training_images(experiment_name, user_id, session, use_real_images, images_shape)

    if random_state is not None:
        random.seed(random_state)

    r = random.choices(range(x.shape[0]), k=number_samples)

    x = np.array(x[r], dtype='float32')
    y = np.array(y[r], dtype='float32')

    print(model_information[SUMMARY_FIELD])

    evaluate_model(model, model_information[HISTORY_FIELD], x, y, number_samples, record_images)


if __name__ == '__main__':
    main_specific_tasks()
    experiment = EXPERIMENT_TEST
    model, model_information = load_model(simple_deconvolution.__name__, MY_ID, 138)
    model.experiment_name = experiment
    execute_model(experiment, MY_ID, 123, model, tuple(model_information[IMAGES_SHAPE_FIELD]),
                  use_real_images=True, number_samples=50, record_images=True
                  )
