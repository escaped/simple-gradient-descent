import math

import numpy as np


def error(value, expected):
    return 1/2 * (value - expected) ** 2

def sigmoid(value):
    return 1 / (1 + np.exp(-value))


def sigmoid_1(value):
    return sigmoid(value) * (1 - sigmoid(value))


def neuron(input, weight, bias):
    return sigmoid(weight * input + bias)


def plot_surface(x, y_true):
    n = 256
    w = np.linspace(-3.0, 3.0, n)
    b = np.linspace(-3.0, 3.0, n)
    W, B = np.meshgrid(w, b)
    Z = error(sigmoid(W * x + B), y_true)

    import matplotlib.cm as cm
    import matplotlib.pyplot as plt
    CSF = plt.contourf(W, B, Z, alpha=0.8, cmap=cm.jet, zorder=0)
    CS = plt.contour(
        W, B, Z,
        # 10,
        [0.0001, .001, .005, .01, .025, .045],
        colors='black', zorder=1,
    )
    plt.clabel(CS, inline=1, fontsize=10)
    plt.title('Simplest default with labels')
    return plt


if __name__ == '__main__':
    threshold = 0.00001
    lr = 0.5

    x = -3.5
    y_true = 0.9
    print(f'expected: {y_true}')
    plt = plot_surface(x, y_true)

    w = 0.5  # initial weight
    b = 0.3  # initial bias
    y = neuron(x, w, b)
    err = error(y, y_true)
    count = 0
    while err > threshold:
        print(f'{count}| error: {err}, y: {y}, w: {w}), b: {b}')
        z = w * x + b
        gradient_w = (y - y_true) * sigmoid_1(z) * x
        gradient_b = (y - y_true) * sigmoid_1(z)
        delta_w = -lr * gradient_w
        delta_b = -lr * gradient_b
        print(f'{count}| error: {err}, y: {y}, w: {w}), b: {b}, dw: {delta_w}, db: {delta_b}')

        plt.quiver(w, b, delta_w, delta_b, color ='red',
                   width=0.005, scale_units='xy',
                   scale=1.0, angles='xy', zorder=2)

        w += delta_w
        b += delta_b
        y = neuron(x, w, b)
        err = error(y, y_true)
        count += 1
    print(f'{count}| error: {err}, y: {y}, w: {w}), b: {b}')

    plt.show()
