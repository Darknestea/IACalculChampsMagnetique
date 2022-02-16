from os import listdir

import numpy as np
import tensorflow as tf
import cv2 as cv
from keras.layers import Conv2D, Dense
from keras.layers.convolutional import Deconv2D
from keras.models import Sequential

from constants import ELLIPSES_PATH, MU_USEFUL_PARAMS, MU_PARAM_NAMES
from data_preprocessing import get_cleaned_configuration

IMAGE_WIDTH = 32
IMAGE_HEIGHT = 32
IMAGE_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH)


def load_images():
    paths = listdir(ELLIPSES_PATH)
    paths = [path for path in paths if path[len(path)-4:] == ".png"]

    y = np.zeros((len(paths), IMAGE_WIDTH, IMAGE_HEIGHT))

    for i, img_path in enumerate(paths):
        img = cv.imread(ELLIPSES_PATH + "\\" + img_path)[:, 304:1744]  # Img 1440*1080 since opencv is reversed

        # convert to grayscale
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        y[i] = cv.resize(gray, IMAGE_SHAPE)

        if False:
            cv.namedWindow(f"Image {i}", cv.WINDOW_NORMAL)
            cv.imshow(f"Image {i}", gray)
            cv.waitKey()

    return y


def create_model(x):
    model = Sequential()
    model.add(Dense(32, inputshape=x.shape[1]))
    model.add(Dense(64))
    model.add(Deconv2D(256, (4, 4), activation='relu', bias_initializer='random_normal'))
    # model.add(Conv2D(64), (2, 2), activation='relu')
    model.compile('adam', loss='rmse', metrics='rmse')
    return model


if __name__ == '__main__':
    # Load parameters
    df, _ = get_cleaned_configuration("Exp3log.txt")

    df = df[[MU_PARAM_NAMES[param] for param in MU_USEFUL_PARAMS][3:12]]
    x = df.to_numpy(dtype=float)

    # Load images
    y = load_images()

    # Preprocessing (normalization)
    y = y/255.
    y.reshape(-1, IMAGE_WIDTH, IMAGE_HEIGHT, 1)

    # Create model
    print(x.shape, y.shape)
    model = create_model(x)

    # Train model
    model.fit(x, y, validation_split=0.8)
    # model.fit(x, y, validation_split=0.8, batch_size=100)
    print(model.summary())

    # Evaluate model
    # TODO once first test are ok
