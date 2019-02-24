import math

import sklearn as skl
import pandas as pd
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

np.random.seed(0)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class MLP:

    def __init__(self, activation):
        layer_0_size = 784
        layer_1_size = 16
        layer_2_size = 16
        layer_3_size = 10
        self.weights = {
            'W_1': np.random.rand(layer_1_size, layer_0_size),
            'W_2': np.random.rand(layer_2_size, layer_1_size),
            'W_3': [layer_3_size, layer_2_size]
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
        self.activation = activation

    def calculate_a(self, A_prev, W):
        Z_next = np.matmul(W, A_prev)
        return self.activation(Z_next)

    def forward_propagation(self, A_0):
        self.layers_outputs['A_1'] = self.calculate_a(A_0, self.weights['W_1'])
        self.layers_outputs['A_2'] = self.calculate_a(self.layers_outputs['A_1'], self.weights['W_2'])
        self.layers_outputs['A_3'] = self.calculate_a(self.layers_outputs['A_3'], self.weights['W_3'])
        return self.layers_outputs['A_3']

    def backward_propagation(self, Y):
        error_vector = self.layers_outputs['A_3'] - Y


def main():
    df = pd.read_csv('./data/train.csv')
    df = pd.read_csv('./data/train.csv')
