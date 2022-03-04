from os import listdir

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Dense, Activation, BatchNormalization, Reshape, Dropout
from keras.layers.convolutional import UpSampling2D, Conv2DTranspose
from keras.models import Sequential, load_model
from sklearn.model_selection import train_test_split

from Utils.constants import ELLIPSES_PATH, MU_USEFUL_PARAMS, MU_PARAM_NAMES, IMAGE_TO_ELLIPSE_MODEL_PATH, \
    DATA_PATH, SAVE_CLEAN_SIMULATION_DATA_PATH
from Utils.DataPreprocessing.data_preprocessing import get_cleaned_configuration

IMAGE_WIDTH = 28
IMAGE_HEIGHT = 28
IMAGE_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH)


def load_images(number_sample):
    paths = listdir(ELLIPSES_PATH)
    paths = [path for path in paths if path[len(path)-4:] == ".png"][:number_sample]

    y = np.zeros((len(paths), IMAGE_WIDTH, IMAGE_HEIGHT), dtype=float)

    for i, img_path in enumerate(paths):
        if number_sample < 20 or i % (number_sample // 20) == 0:
            print(f"{i} on {number_sample}")
        img = cv.imread(ELLIPSES_PATH + "\\" + img_path)[:, 304:1744]  # Img 1440*1080 since opencv is reversed

        # convert to grayscale
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        y[i] = cv.resize(gray, IMAGE_SHAPE)

        if i == 0:
            cv.namedWindow(f"Image {i}", cv.WINDOW_NORMAL)
            cv.imshow(f"Image {i}", np.concatenate([gray, cv.resize(cv.resize(gray, IMAGE_SHAPE), gray.shape).T]).T)
            cv.waitKey()

    return normalize_image(y)


def create_model(input_dim, output_width):
    dim = int(output_width / 8)
    depth = output_width * 2
    dropout = 0.9

    model = Sequential()
    model.add(Dense(32, input_shape=(input_dim,)))
    model.add(Activation('relu'))

    model.add(Dense(output_width))
    model.add(Activation('relu'))

    model.add(Dense(dim * dim * depth))
    model.add(BatchNormalization(momentum=0.9))
    model.add(Activation('relu'))

    model.add(Reshape((dim, dim, depth)))
    model.add(Dropout(dropout))

    model.add(UpSampling2D())

    model.add(Conv2DTranspose(int(depth / 2), 5, padding='same'))
    model.add(BatchNormalization(momentum=0.9))
    model.add(Activation('relu'))

    model.add(Conv2DTranspose(int(depth / 4), 5, padding='same'))
    model.add(BatchNormalization(momentum=0.9))
    model.add(Activation('relu'))

    model.add(UpSampling2D())

    model.add(Conv2DTranspose(int(depth / 8), 5, padding='same'))
    model.add(BatchNormalization(momentum=0.9))
    model.add(Activation('relu'))

    model.add(UpSampling2D())

    model.add(Conv2DTranspose(1, 5, padding='same'))
    model.add(Activation('sigmoid'))

    # Out: grayscale image [0.0,1.0] per pix
    model.add(Reshape((output_width, output_width)))

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])

    return model


def create_basic_model(input_dim, output_width):

    model = Sequential()
    model.add(Dense(32, input_dim=input_dim))
    model.add(Activation('relu'))

    model.add(Dense(64))
    model.add(Activation('relu'))

    model.add(Dense(output_width*output_width))
    model.add(Activation('sigmoid'))

    model.add(Reshape((output_width, output_width)))

    model.compile(optimizer='adam', loss='mse', metrics=['mse'])

    return model


def test(simulation_data=False):
    y_path = DATA_PATH + "Cleaned\\Exp_28x28_ellipse\\simulated_beam.npy"
    model_path = IMAGE_TO_ELLIPSE_MODEL_PATH + "my_test_model.h5"

    #Computed is valid only for
    already_computed = True
    already_trained = True

    # already_computed = False
    already_trained = False

    # Load parameters
    if simulation_data:
        x = np.load(SAVE_CLEAN_SIMULATION_DATA_PATH + "Exp3_simulation_configurations.npy")[:, :9]
        y_256 = np.load(SAVE_CLEAN_SIMULATION_DATA_PATH + "simulated_beam.npy")
        y = np.zeros((y_256.shape[0], 128, 128), dtype=float)
        for i in range(y.shape[0]):
            y[i] = cv.resize(y_256[i], (128, 128))

    else:
        start_range = 3
        end_range = 12
        number_sample = 3600
        df, _ = get_cleaned_configuration("Exp_3.txt")

        df = df[[MU_PARAM_NAMES[param] for param in MU_USEFUL_PARAMS][start_range:end_range]]
        x = df.to_numpy(dtype=float)[:number_sample]

        if already_computed:
            y = np.load(y_path)
        else:
            # Load images
            y = load_images(number_sample)
            np.save(y_path, y)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    print(f"Shapes:\n"
          f"\tx_train : {x_train.shape}\n"
          f"\ty_train : {y_train.shape}\n"
          f"\tx_test : {x_test.shape}\n"
          f"\ty_test : {y_test.shape}"
          )

    # Train model
    if already_trained:
        model = load_model(model_path)
        model.summary()
    else:
        # Create model
        model = create_basic_model(x.shape[1], 128)
        model.summary()
        history = model.fit(x_train, y_train, epochs=50, batch_size=32, validation_data=(x_test, y_test))
        model.save(model_path)

        # Evaluate model
        plt.title('Loss')
        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='test')
        plt.legend()
        plt.show()

    # Evaluate model
    print(f"Evaluation = {model.evaluate(x_test, y_test)}")

    # train dataset results
    images_to_show = range(10)
    y_predicted = model.predict(x_train)
    for i in images_to_show:
        cv.namedWindow(f"Training: Ground truth (left), Image predicted (right)", cv.WINDOW_NORMAL)
        cv.imshow(f"Training: Ground truth (left), Image predicted (right)",
                  np.concatenate([y_train[i], y_predicted[i]]).T
                  )
        cv.waitKey()

    # Test dataset results
    images_to_show = range(x_test.shape[0])
    y_predicted = model.predict(x_test)
    for i in images_to_show:
        cv.namedWindow(f"Ground truth (left), Image predicted (right)", cv.WINDOW_NORMAL)
        cv.imshow(f"Ground truth (left), Image predicted (right)",
                  np.concatenate([y_test[i], y_predicted[i]]).T
                  )
        cv.waitKey()


def normalize_image(image):
    return image / 255.


def denormalize_image(image):
    return image * 255.


def test_network(y_is_x_dependant=True):
    size = 36000
    x = np.random.random(size * 9).reshape(-1, 9)
    if y_is_x_dependant:
        x2 = np.concatenate([x, x, x, np.zeros((size, 5))], axis=1)
        y = x2.reshape(size, 1, 32) * x2.reshape(size, 32, 1)
    else:
        y = np.random.random(size * 32 * 32).reshape(size, 32, 32)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    model = create_model(9, 32)
    model.summary()
    history = model.fit(x_train, y_train, epochs=15, batch_size=32, validation_data=(x_test, y_test))

    # Evaluate model
    plt.title('Loss')
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()
    # Evaluate model
    print(f"Evaluation = {model.evaluate(x_test, y_test)}")

    images_to_show = range(x_test.shape[0])
    y_predicted = model.predict(x_test)

    for i in images_to_show:
        cv.namedWindow(f"Ground truth (left), Image predicted (right)", cv.WINDOW_NORMAL)
        cv.imshow(f"Ground truth (left), Image predicted (right)",
                  np.concatenate([y_test[i], y_predicted[i]]).T
                  )
        cv.waitKey()


if __name__ == '__main__':
    # test_network(True)
    # load_images(5)
    test(True)
