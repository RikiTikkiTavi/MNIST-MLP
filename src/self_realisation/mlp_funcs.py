import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(x, 0)


def d_relu_dz(z):
    return (z > 0) * 1


def leaky_relu(Z):
    return np.maximum(0.1 * Z, Z)


def d_leaky_relu_dz(z):
    return 1 if z >= 0 else 0.01


def tanh(Z):
    return np.tanh(Z)


def d_tanh_dz(z):
    return 1 - (np.square(np.tanh(z)))


def d_sigmoid_dz(z):
    return sigmoid(z) * (1 - sigmoid(z))


def mse(Y_hat, Y):
    return np.sum((np.square(Y_hat - Y)))


def d_mse_da(a, y):
    return 2*(a - y)
