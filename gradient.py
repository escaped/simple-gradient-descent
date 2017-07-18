import math

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np


def error(X, Y_true, w, b):
    err = None
    for x, y_true in zip(X, Y_true):
        y = sigmoid(w * x + b)
        if err is None:
            err = 1/2 * (y - y_true) ** 2
        else:
            err += 1/2 * (y - y_true) ** 2
    return err

def sigmoid(value):
    return 1 / (1 + np.exp(-value))


def sigmoid_1(value):
    return sigmoid(value) * (1 - sigmoid(value))


def neuron(input, weight, bias):
    return sigmoid(weight * input + bias)


def plot_surface(X, Y_true):
    n = 256
    w = np.linspace(-3.0, 3.0, n)
    b = np.linspace(-3.0, 3.0, n)
    W, B = np.meshgrid(w, b)
    Z = error(X, Y_true, W, B)

    plt.figure()
    CSF = plt.contourf(W, B, Z, alpha=0.8, cmap=cm.jet, zorder=0)
    CS = plt.contour(
        W, B, Z,
        # 10,
        [0.0001, .001, .005, .01, .025, .045],
        colors='black', zorder=1,
    )
    plt.clabel(CS, inline=1, fontsize=10)


def plot_sigma(X, Y, w, b):
    plt.figure()
    n = 256
    X_sigma = np.linspace(-5.0, 5.0, n)
    Y_sigma = sigmoid(w * X_sigma + b)
    plt.plot(X_sigma, Y_sigma)
    for x, y in zip(X, Y):
        plt.plot(x, y, marker='o', color='red')


if __name__ == '__main__':
    threshold = 0.0001
    lr = 0.1

    X = np.asarray([1, -3.5])
    Y_true = np.asarray([0.6, 0.5])
    print(f'expected: {Y_true}')

    w = 0.5  # initial weight
    b = 0.3  # initial bias

    plot_sigma(X, Y_true, w, b)
    plot_surface(X, Y_true)

    Y = neuron(X, w, b)
    err = error(X, Y_true, w, b)
    count = 0
    while err > threshold:
        print(f'{count}| error: {err}, y: {Y}, y_true: {Y_true}, w: {w}), b: {b}')
        gradient_w = gradient_b = 0
        for x, y, y_true in zip(X, Y, Y_true):
            z = w * x + b
            gradient_w += (y - y_true) * sigmoid_1(z) * x
            gradient_b += (y - y_true) * sigmoid_1(z)
        delta_w = -lr * gradient_w
        delta_b = -lr * gradient_b
        print(f'{count}| gradient_w: {gradient_w}, gradient_b: {gradient_b}')
        print(f'{count}| dw: {delta_w}, db: {delta_b}')

        plt.quiver(w, b, delta_w, delta_b, color='red',
                   width=0.005, scale_units='xy',
                   scale=1.0, angles='xy', zorder=2)

        w += delta_w
        b += delta_b
        Y = neuron(X, w, b)
        err = error(X, Y_true, w, b)
        count += 1
    print(f'{count}| error: {err}, w: {w}), b: {b}')

    plot_sigma(X, Y_true, w, b)
    plt.show()
