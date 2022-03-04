import tensorflow as tf
import cv2 as cv
import numpy as np

from Utils.constants import set_run_session


# Main specific task
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
    img = cv.cvLoadImage(image_path)
    if img is None:
        print(image_path)
    return img[:, 304:1744]


def gray_from_image(img):
    return cv.cvtColor(img, cv.COLOR_BGR2GRAY)


def load_gray_image(image_path):
    return gray_from_image(load_image(image_path))
