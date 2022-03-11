import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
from PIL import Image

from Utils.DataPreprocessing.create_folders import create_folder
from Utils.constants import EXECUTE_RESULTS_IMAGES_PATH


def record_pretrained_model(model, x, y):
    return record_trained_model(model, x, y)


def record_trained_model(model, x, y):
    print(model.predict(x))
    print(y)
    return


def evaluate_model(model, history, x_test, y_test, number_samples=10, record_images=False, show_images=False):
    # Evaluate model
    plt.title('Loss')
    plt.plot(history['loss'], label='train')
    plt.plot(history['val_loss'], label='test')
    plt.legend()

    if record_images and model.experiment_name is not None:
        path = EXECUTE_RESULTS_IMAGES_PATH(model.name, model.experiment_name, number_samples)
        create_folder(path)
        plt.savefig(f"{path}history.png")

    if show_images:
        plt.show(block=False)
        plt.pause(5)
        plt.close()

    y_predicted = model.predict(x_test)

    if show_images:
        display_images(number_samples, y_predicted, y_test)

    if record_images and model.experiment_name is not None:
        save_images(model, number_samples, y_predicted, y_test, x_test)

    return model.evaluate(x_test, y_test)


def display_images(number_samples, y_predicted, y_test):
    for i in range(number_samples):
        print(f"{i}/{number_samples}", end="\r")
        cv.namedWindow(f"Ground truth (left), Image predicted (right)", cv.WINDOW_NORMAL)
        cv.imshow(f"Ground truth (left), Image predicted (right)",
                  np.concatenate([y_test[i].T, y_predicted[i].T]).T
                  )
        cv.waitKey(1000)


def save_images(model, number_samples, y_predicted, y, x):
    path = EXECUTE_RESULTS_IMAGES_PATH(model.name, model.experiment_name, number_samples)
    for i in range(number_samples):
        print(f"{i+1}/{number_samples}", end="\r")
        true = (y[i] * 255).astype(np.uint8)
        predicted = (y_predicted[i] * 255).astype(np.uint8)
        Image.fromarray(true).save(f"{path}{i}_true.png")
        Image.fromarray(predicted).save(f"{path}{i}_predicted.png")
    print("\n")
    np.save(f"{path}x.npy", x)
