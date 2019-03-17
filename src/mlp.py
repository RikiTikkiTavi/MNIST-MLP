import math
import sklearn as skl
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import src.mlp_funcs as mlp_funcs
from src.utils import construct_y


class MLP:

    def __init__(self):
        layer_0_size = 784
        layer_1_size = 16
        layer_2_size = 16
        layer_3_size = 10
        self.layers_sizes = [layer_0_size, layer_1_size, layer_2_size, layer_3_size]
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
            'A_0': None,
            'A_1': None,
            'A_2': None,
            'A_3': None
        }

        self.layers_z = {
            'Z_1': None,
            'Z_2': None,
            'Z_3': None
        }

        self.error_signals = {
            'ES_1': np.empty(layer_1_size),
            'ES_2': np.empty(layer_2_size),
            'ES_3': np.empty(layer_3_size)
        }

        self.act_f = mlp_funcs.sigmoid
        self.d_act_f_dz = mlp_funcs.d_sigmoid_dz
        self.err_f = mlp_funcs.mse
        self.d_err_f_da = mlp_funcs.d_mse_da

        self.Y = None

    def __get_A(self, index):
        return self.layers_outputs[f'A_{index}']

    def __set_A(self, index, A):
        self.layers_outputs[f'A_{index}'] = A

    def __get_B(self, index):
        return self.biases[f'B_{index}']

    def __get_W(self, index):
        return self.weights[f'W_{index}']

    def __get_Z(self, index):
        return self.error_signals[f'Z_{index}']

    def __set_Z(self, index, Z):
        self.layers_outputs[f'Z_{index}'] = Z

    def __get_ES(self, index):
        return self.error_signals[f'ES_{index}']

    def __set_ES(self, index, ES):
        self.error_signals[f'ES_{index}'] = ES

    # noinspection PyTypeChecker
    def __calculate_Z(self, l_index):
        return np.matmul(self.__get_W(l_index), self.__get_A(l_index - 1)) - self.__get_B(l_index)

    def __calculate_A(self, Z):
        return self.act_f(Z)

    def __handle_update_layer(self, i):
        Z = self.__calculate_Z(i)
        self.__set_Z(i, Z)
        self.__set_A(i, self.__calculate_A(Z))

    def handle_prop_forward(self, A_0):
        self.__set_A(0, A_0)
        self.__handle_update_layer(1)
        self.__handle_update_layer(2)
        self.__handle_update_layer(3)

    def __calc_error_signal_outer_shortcut(self, i, l_i):
        self.__calc_error_signal_outer(
            (self.__get_A(l_i))[i],
            (self.__get_Z(l_i))[i],
            self.Y[i]
        )

    def __handle_update_error_signals_last_layer(self, l_i, l_size):
        self.__set_ES(
            l_i,
            np.fromiter(
                map(
                    lambda i: self.__calc_error_signal_outer_shortcut(i, l_i),
                    range(l_size)
                ),
                dtype=float
            )
        )

    def handle_prop_backward(self, Y):
        self.Y = Y
        self.__handle_update_error_signals_last_layer(3, 10)

    def __calc_error_signal_outer(self, a, z, y):
        return self.d_err_f_da(a, y) * self.d_act_f_dz(z)

    def __calc_error_signal_inner(self, z, ES_next, W_next, z_index):
        return np.sum([es * W_next[index][z_index] for index, es in enumerate(ES_next)]) * self.d_act_f_dz(z)

    def handle_calc_error_signals(self, Y):
        # Calc outer
        pass
