import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(x, 0)


def d_relu_dz(z):
    z[z <= 0] = 0
    z[z > 0] = 1
    return z


def d_sigmoid_dz(z):
    return sigmoid(z) * (1 - sigmoid(z))


def mse(Y_hat, Y):
    return np.sum((np.square(Y_hat - Y)))


def d_mse_da(a, y):
    return 2 * (a - y)
