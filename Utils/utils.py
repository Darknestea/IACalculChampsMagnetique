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
    img = cv.imread(image_path)
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
