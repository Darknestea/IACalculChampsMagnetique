import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from matplotlib.patches import Ellipse
from dataclasses import dataclass

IM_SIZE_DEFAULT = 256
IM_SHAPE_DEFAULT = (IM_SIZE_DEFAULT, IM_SIZE_DEFAULT)

A_MAX = IM_SIZE_DEFAULT
B_MAX = IM_SIZE_DEFAULT
U_MAX = IM_SIZE_DEFAULT
V_MAX = IM_SIZE_DEFAULT
THETA_MIN = -np.pi
THETA_MAX = np.pi


@dataclass
class Ellipse:
    x: int
    y: int
    a: float
    b: float
    theta: float


# Return True if the point (x,y) is in the ellipse
def is_xy_in_ellipse(x, y, ellipse):
    return ((((x - ellipse.x) * np.cos(ellipse.theta) + (y - ellipse.y) * np.sin(ellipse.theta)) / ellipse.a) ** 2 +
            ((((x - ellipse.x) * np.sin(ellipse.theta) - (y - ellipse.y) * np.cos(ellipse.theta)) / ellipse.b) ** 2)) \
           < 1


def image_from_ellipse(ellipse):
    return np.fromfunction(lambda x, y:
                           is_xy_in_ellipse(x, y, ellipse),
                           IM_SHAPE_DEFAULT,
                           dtype=float)


def generate_random_ellipse():
    a = np.random.random() * A_MAX
    b = np.random.random() * B_MAX
    x = np.random.randint(U_MAX)
    y = np.random.randint(V_MAX)
    theta = np.random.random() * (THETA_MAX - THETA_MIN) + THETA_MIN
    return Ellipse(x, y, a, b, theta)


def generate_set(dataset_size, im_size=IM_SIZE_DEFAULT, im_shape=IM_SHAPE_DEFAULT):
    dataset_shape = (dataset_size, IM_SHAPE_DEFAULT, IM_SIZE_DEFAULT)
    dataset = np.zeros(dataset_shape)
    for i in range(dataset_size):
        dataset[i] = image_from_ellipse(generate_random_ellipse())
    return dataset
