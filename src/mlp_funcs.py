import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def d_sigmoid_dz(z):
    return sigmoid(z) * (1 - sigmoid(z))


def mse(Y_hat, Y):
    return np.sum((np.square(Y - Y_hat)))


def d_mse_da(a, y):
    return 2 * (a - y)
