import math

import sklearn as skl
import pandas as pd
import numpy as np

import matplotlib
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def mse(Y_hat, Y):
    return (np.square(Y - Y_hat)).mean()


class MLP:

    def __init__(self, activation_func, error_func):
        layer_0_size = 784
        layer_1_size = 16
        layer_2_size = 16
        layer_3_size = 10
        self.weights = {
            'W_1': np.random.rand(layer_1_size, layer_0_size),
            'W_2': np.random.rand(layer_2_size, layer_1_size),
            'W_3': np.random.rand(layer_3_size, layer_2_size)
        }
        self.biases = {
            'B_1': np.random.rand(layer_1_size),
            'B_2': np.random.rand(layer_2_size),
            'B_3': np.random.rand(layer_3_size)
        }
        self.layers_outputs = {
            'A1': None,
            'A2': None,
            'A3': None
        }
        self.activation_func = activation_func
        self.error_func = error_func

    def calculate_a(self, A_prev, W):
        Z_next = np.matmul(W, A_prev)
        return self.activation_func(Z_next)

    def forward_propagation(self, A_0):
        self.layers_outputs['A_1'] = self.calculate_a(A_0, self.weights['W_1'])
        self.layers_outputs['A_2'] = self.calculate_a(self.layers_outputs['A_1'], self.weights['W_2'])
        self.layers_outputs['A_3'] = self.calculate_a(self.layers_outputs['A_2'], self.weights['W_3'])
        return self.layers_outputs['A_3']

    def backward_propagation(self, Y):
        print('Y: ', Y)
        Y_hat = self.layers_outputs['A_3']
        cost = self.error_func(Y_hat, Y)
        print('COST: ', cost)


def construct_y(y):
    Y = np.array([i if y == i else 0 for i in range(0, 10)])
    return Y


def main():
    df = pd.read_csv('./data/train.csv')
    mlp = MLP(sigmoid, mse)

    for index, row in df.iterrows():
        vector = row.to_numpy()
        mlp.forward_propagation(vector[1:])
        mlp.backward_propagation(construct_y(vector[0]))


np.random.seed(0)
main()
