from os import listdir

import numpy as np
import tensorflow as tf
import cv2 as cv
from keras.layers import Conv2D, Dense
from keras.models import Sequential

from constants import ELLIPSES_PATH
from data_preprocessing import get_cleaned_configuration

IMAGE_WIDTH = 32
IMAGE_HEIGHT = 32
IMAGE_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH)


def load_images():
    paths = listdir(ELLIPSES_PATH)
    paths = [path for path in paths if path[len(path)-4:] == ".png"]

    y = np.zeros((len(paths), IMAGE_WIDTH, IMAGE_HEIGHT))

    for i, img_path in enumerate(paths):
        img = cv.imread(img_path)[:, 304:1744]  # Img 1440*1080 since opencv is reversed

        # convert to grayscale
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        y[i] = cv.resize(gray, IMAGE_SHAPE)

        cv.namedWindow(f"Image {i}", cv.WINDOW_NORMAL)
        cv.imshow(f"Image {i}", gray)

    return y


if __name__ == '__main__':
    # Load images
    y = load_images()

    # Load parameters
    x = get_cleaned_configuration("Exp3log.txt")

    # Preprocessing (normalization)
    y = y/255.

    # Create model
    model = Sequential()
    model.add(Conv2D(256, (4, 4), activation='relu', bias_initializer='random_normal', input_shape=IMAGE_SHAPE))
    model.add(Conv2D(64), (2, 2), activation='relu')
    model.add(Dense(32))
    model.add(Dense(x.shape[1:]))
    model.compile('adam', loss='rmse', metrics='rmse')

    # Train model
    model.train(x, y, train_test_split=0.8)

    # Evaluate model
    # TODO once first test are ok
