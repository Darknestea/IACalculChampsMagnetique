from os import chdir
from os.path import curdir, basename
from pathlib import Path

import tensorflow as tf
import cv2 as cv
import numpy as np
from PIL import Image as PilIm

from Utils.constants import set_run_session, SCREEN_SHAPE, run_session, MODEL_PATH

# Main specific task
from my_id import MY_ID


def main_specific_tasks(filename=None):
    # Generate unique identifier to avoid data loss
    if filename is not None:
        set_run_session(filename)
    else:
        set_run_session()

    print("TensorFlow version: ", tf.__version__)

    # Test GPU
    device_name = tf.test.gpu_device_name()
    if not device_name:
        print("GPU device not found")
    print("Found GPU at: {}".format(device_name))


def load_image(image_path):
    path = Path(image_path)
    current_path = Path(curdir)
    chdir(path.parent.absolute())
    img = cv.imread(basename(path))
    chdir(current_path)
    return img[:, 304:1744]


def gray_from_image(img):
    return cv.cvtColor(img, cv.COLOR_BGR2GRAY)


def load_gray_image(image_path):
    return cv.resize(gray_from_image(load_image(image_path)), SCREEN_SHAPE)


def normalize_image(image):
    return image / 255.


def denormalize_image(image):
    return image * 255.


def normalize_by_column(data):
    # TODO should change to be consistent for each value type
    for j in range(data.shape[1]):
        max_j = max(data[:, j])
        min_j = min(data[:, j])
        if min_j == max_j:
            data[:, j] = 1.
        else:
            data[:, j] = (data[:, j] - min(data[:, j])) / (max(data[:, j])-min(data[:, j]))
    return data


def reshape_images(images, image_shape):
    reshaped_images = np.zeros(((images.shape[0],) + image_shape), dtype=float)
    for i in range(images.shape[0]):
        pil_img = PilIm.fromarray(images[i]).resize(image_shape)
        reshaped_images[i] = np.array(pil_img)
    return reshaped_images


def give_name(base_name):
    return f"{base_name}_{MY_ID}_{run_session()}"


def get_model_name(base_name, user_id, session):
    return f"{base_name}_{user_id}_{session}"


def get_model_summary(model):
    summary_list = []
    model.summary(print_fn=lambda s: summary_list.append(s))
    return "\n".join(summary_list)


def print_to_string(s):
    return s
