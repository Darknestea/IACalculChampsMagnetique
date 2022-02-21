import ellipse as el
import numpy as np
import matplotlib.pyplot as plt
from ellipse import LsqEllipse
from matplotlib.patches import Ellipse
from scipy.signal import savgol_filter


def generate_ellipse(x, y, r, R, theta):
    t = np.linspace(0, 360, 500)
    X = x + (r * np.cos(np.radians(t)) * np.cos(np.radians(theta)) - R * np.sin(np.radians(t)) * np.sin(
        np.radians(theta)))
    Y = y + (r * np.cos(np.radians(t)) * np.sin(np.radians(theta)) + R * np.sin(np.radians(t)) * np.cos(
        np.radians(theta)))

    X = (X + 0.5).astype(int).astype(float)
    Y = (Y + 0.5).astype(int).astype(float)
    return X, Y


def trim(X):
    return X


def rework(x):
    # Make 100-150 a line
    a = x[100]
    b = x[150]
    x_ab = np.linspace(a[0], b[0], 50)
    y_ab = np.linspace(a[1], b[1], 50)
    x[100:150] = np.dstack([x_ab, y_ab])

    # Make 200-216 a bug
    a = x[200]
    b = x[216]
    c = (a + b)/2 - np.array([(b - a)[1], (a - b)[0]])

    print(a, b, c)

    x_ac = np.linspace(a[0], c[0], 8)
    y_ac = np.linspace(a[1], c[1], 8)
    x_cb = np.linspace(c[0], b[0], 8)
    y_cb = np.linspace(c[1], b[1], 8)

    x[200:208] = np.dstack([x_ac, y_ac])
    x[208:216] = np.dstack([x_cb, y_cb])

    # Make 250-420 a line
    a = x[250]
    b = x[420]
    x_ab = np.linspace(a[0], b[0], 170)
    y_ab = np.linspace(a[1], b[1], 170)
    x[250:420] = np.dstack([x_ab, y_ab])

    # Add noise
    noise = np.random.normal(0, 0.5, x.size)
    x = x + noise.reshape(x.shape)

    return x


def test():
    x = np.random.randint(10, 240)
    y = np.random.randint(10, 240)
    b = np.random.randint(10, 50)
    a = np.random.randint(b, b * 2)
    theta = np.random.randint(-90, 90)

    x1, x2 = generate_ellipse(x, y, b, a, theta)

    x = np.array(list(zip(x1, x2)))

    x = rework(x)

    x = trim(x)

    reg = LsqEllipse().fit(x)
    center, width, height, phi = reg.as_parameters()

    x_smooth = np.dstack([x[:, 0], savgol_filter(x[:, 1], 51, 3)])[0]
    print(x_smooth.shape)

    plt.close('all')
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    ax.axis('equal')
    ax.plot(x[:, 0], x[:, 1], 'r', label='raw', zorder=1)

    ax.plot(x_smooth[:, 0], x_smooth[:, 1], 'b', label='smooth', zorder=1)

    ellipse = Ellipse(xy=center, width=2 * width, height=2 * height, angle=np.rad2deg(phi),
                      edgecolor='g', fc='None', lw=2, label='Fit', zorder=2)
    ax.add_patch(ellipse)

    plt.legend()
    plt.show()

    # plt.close('all')
    # fig = plt.figure(figsize=(6, 6))
    # ax = fig.add_subplot(111)
    # ax.axis('equal')
    #
    # print(X)
    # # grad = np.array(np.gradient(X, axis=0))
    # grad = np.diff(X[:, 0]) * np.diff(X[:, 1]).T
    #
    # print("gradient:\n", grad, "\n", grad.shape)
    #
    # ax.plot(np.arange(grad.shape[0]), grad, 'red', label='grad_x', zorder=1)
    #
    # plt.legend()
    # plt.show()




if __name__ == '__main__':
    test()