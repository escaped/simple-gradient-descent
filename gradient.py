import click
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np


def error(X, Y_true, w, b):
    """
    Calculates the error for the given weight and bias for each point.
    """
    err = None
    for x, y_true in zip(X, Y_true):
        y = sigmoid(w * x + b)
        if err is None:
            err = 1/2 * (y - y_true) ** 2
        else:
            err += 1/2 * (y - y_true) ** 2
    return err


def sigmoid(value):
    """
    Implements the sigmoid activation function.
    """
    return 1 / (1 + np.exp(-value))


def sigmoid_1(value):
    """
    Implements the derivation of the sigmoid function.
    """
    return sigmoid(value) * (1 - sigmoid(value))


def neuron(input, weight, bias):
    """
    Calulates the output of an artificial neuron.
    """
    return sigmoid(weight * input + bias)


def plot_surface(X, Y_true):
    """
    Plots the error surface.
    """
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
    """
    Plots the learned function of our artifical neuron and the example points.
    """
    plt.figure()
    n = 256
    X_sigma = np.linspace(-5.0, 5.0, n)
    Y_sigma = sigmoid(w * X_sigma + b)
    plt.plot(X_sigma, Y_sigma)
    for x, y in zip(X, Y):
        plt.plot(x, y, marker='o', color='red')


@click.command()
@click.option('--threshold', default=0.0001, type=click.FLOAT,
              help="Train until the error reaches this threshold.")
@click.option('--lr', default=0.1, type=click.FLOAT, help="Learning rate")
@click.option('--initial_weight', default=0.5, type=click.FLOAT)
@click.option('--initial_bias', default=0.3, type=click.FLOAT)
@click.option('--point', nargs=2,
              type=click.Tuple([float, click.FloatRange(0, 1, clamp=True)]),
              multiple=True, required=True,
              help="Supply at least one data point to approximate.")
def train(threshold, lr, initial_weight, initial_bias, point):
    X = np.asarray([x for x, _ in point])
    Y_true = np.asarray([y for _, y in point])
    print(f'Input data: {X}')
    print(f'Expected data: {Y_true}')

    w = initial_weight
    b = initial_bias

    plot_sigma(X, Y_true, w, b)
    plot_surface(X, Y_true)

    Y = neuron(X, w, b)
    err = error(X, Y_true, w, b)
    count = 0
    while err > threshold:
        print(f'{count}| error: {err}, y: {Y}, y_true: {Y_true}, w: {w}), b: {b}')

        # calculate the gradient of w and b
        gradient_w = gradient_b = 0
        for x, y, y_true in zip(X, Y, Y_true):
            z = w * x + b
            gradient_w += (y - y_true) * sigmoid_1(z) * x
            gradient_b += (y - y_true) * sigmoid_1(z)
        delta_w = -lr * gradient_w
        delta_b = -lr * gradient_b

        # plot movement through the error space
        plt.quiver(w, b, delta_w, delta_b, color='red',
                   width=0.005, scale_units='xy',
                   scale=1.0, angles='xy', zorder=2)

        # update w and b
        w += delta_w
        b += delta_b
        Y = neuron(X, w, b)
        err = error(X, Y_true, w, b)
        count += 1
    print(f'{count}| error: {err}, w: {w}), b: {b}')

    plot_sigma(X, Y_true, w, b)
    plt.show()


if __name__ == '__main__':
    train()
