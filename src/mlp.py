import math
import sklearn as skl
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import src.mlp_funcs as mlp_funcs
from src.utils import construct_y
from src.utils import create_gradient_accum
from src.utils import log


class MLP:

    def __init__(self):
        layer_0_size = 784
        layer_1_size = 16
        layer_2_size = 16
        layer_3_size = 10

        self.layers_sizes = (layer_0_size, layer_1_size, layer_2_size, layer_3_size)

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

        self.gradient_accumulator = create_gradient_accum(self.layers_sizes)

        self.act_f = mlp_funcs.sigmoid
        self.d_act_f_dz = mlp_funcs.d_sigmoid_dz
        self.err_f = mlp_funcs.mse
        self.d_err_f_da = mlp_funcs.d_mse_da

        self.Y = None
        self.batch_size = None

        self.learning_rate = 0.5

    # -----------------
    # Getters / Setters
    # -----------------

    def __get_A(self, index):
        return self.layers_outputs[f'A_{index}']

    def __set_A(self, index, A):
        self.layers_outputs[f'A_{index}'] = A

    def __get_B(self, index):
        return self.biases[f'B_{index}']

    def __set_B(self, index, B):
        self.biases[f'B_{index}'] = B

    def __get_W(self, index):
        return self.weights[f'W_{index}']

    def __set_W(self, index, W):
        self.weights[f'W_{index}'] = W

    def __get_Z(self, index):
        return self.layers_z[f'Z_{index}']

    def __set_Z(self, index, Z):
        self.layers_z[f'Z_{index}'] = Z

    def __get_ES(self, index):
        return self.error_signals[f'ES_{index}']

    def __set_ES(self, index, ES):
        self.error_signals[f'ES_{index}'] = ES

    def __add_W_to_gradient_accum(self, l_i, W):
        self.gradient_accumulator['weights'][f'W_{l_i}'] += W

    def __add_B_to_gradient_accum(self, l_i, B):
        self.gradient_accumulator['biases'][f'B_{l_i}'] += B

    def __set_gradient_accumulator(self, g_a):
        self.gradient_accumulator = g_a

    # -----------
    # Calculators
    # -----------

    def __calc_gradient_step(self):
        for p_type, params_dict in self.gradient_accumulator.items():
            for label, params_arr in params_dict:
                self.gradient_accumulator[p_type][label] /= self.batch_size
                self.gradient_accumulator[p_type][label] *= self.learning_rate

    # noinspection PyTypeChecker
    def __calculate_Z(self, l_index):
        return np.matmul(self.__get_W(l_index), self.__get_A(l_index - 1)) - self.__get_B(l_index)

    def __calculate_A(self, Z):
        return self.act_f(Z)

    def __calc_dC_dw(self, l_i, k, j):
        return self.__get_ES(l_i)[k] * self.__get_A(l_i - 1)[j]

    def __calc_dC_db(self, l_i, k):
        return self.__get_ES(l_i)[k]

    def __calc_error_signal_outer(self, a, z, y):
        return self.d_err_f_da(a, y) * self.d_act_f_dz(z)

    def __calc_error_signal_inner(self, z, ES_next, W_next, z_index):
        return np.sum([es * W_next[index][z_index] for index, es in enumerate(ES_next)]) * self.d_act_f_dz(z)

    # -----------------
    # Internal Handlers
    # -----------------

    def __handle_update_layer(self, i):
        Z = self.__calculate_Z(i)
        self.__set_Z(i, Z)
        self.__set_A(i, self.__calculate_A(Z))

    def __calc_error_signal_outer_shortcut(self, i, l_i):
        return self.__calc_error_signal_outer(
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

    def __handle_update_error_signals_layer(self, l_i, l_size):
        W_next = self.__get_W(l_i + 1)
        ES_next = self.__get_ES(l_i + 1)
        Z = self.__get_Z(l_i)
        self.__set_ES(
            l_i,
            np.fromiter(
                map(
                    lambda i: self.__calc_error_signal_inner(Z[i], ES_next, W_next, i),
                    range(l_size)
                ),
                dtype=float
            )
        )

    def __update_error_signals(self):
        self.__handle_update_error_signals_last_layer(3, 10)
        log(f"Calculated last layer ES:\n {self.error_signals['ES_3']}")
        self.__handle_update_error_signals_layer(2, 16)
        self.__handle_update_error_signals_layer(1, 16)

    def __handle_add_dC_dw_to_accum(self, l_i):
        W = self.__get_W(l_i)  # To get an array of correct shape
        for k, row in enumerate(W):
            for j, weight in enumerate(row):
                W[k][j] = self.__calc_dC_dw(l_i, k, j)
        self.__add_W_to_gradient_accum(l_i, W)

    def __handle_add_dC_db_to_accum(self, l_i):
        B = self.__get_B(l_i)  # To get an array of correct shape
        for k, b in enumerate(B):
            B[k] = self.__calc_dC_db(l_i, k)
        self.__add_B_to_gradient_accum(l_i, B)

    def __handle_add_to_accum_params_layer(self, l_i):
        self.__handle_add_dC_dw_to_accum(l_i)
        self.__handle_add_dC_db_to_accum(l_i)

    def __update_accum(self):
        self.__handle_add_to_accum_params_layer(3)
        self.__handle_add_to_accum_params_layer(2)
        self.__handle_add_to_accum_params_layer(1)

    def __handle_apply_gradient_accumulator(self):
        for p_type, params_dict in self.gradient_accumulator.items():
            for label, params_arr in params_dict.items():
                if p_type == 'weights':
                    self.weights[label] -= self.gradient_accumulator[p_type][label]
                elif p_type == 'biases':
                    self.biases[label] -= self.gradient_accumulator[p_type][label]

    def __handle_clear_gradient_accumulator(self):
        self.__set_gradient_accumulator(create_gradient_accum(self.layers_sizes))

    def __handle_prop_backward(self, Y):
        self.Y = Y
        cost = self.err_f(Y_hat=self.__get_A(3), Y=Y)
        log(f"Current cost computed: {cost}")
        log(f"Current Y set:\n {Y}")
        self.__update_error_signals()
        log(f"Error signals calculated:\n {self.error_signals}")
        self.__update_accum()
        log(f"Gradient accum updated!")

    def __handle_prop_forward(self, A_0):
        self.__set_A(0, A_0)
        self.__handle_update_layer(1)
        self.__handle_update_layer(2)
        self.__handle_update_layer(3)

    def __handle_single_vector(self, vector):
        log(f"Processing vector:\n {vector}")
        self.__handle_prop_forward(vector[1:])
        log(f"Forward prop completed: output:\n {self.layers_outputs['A_3']}")
        self.__handle_prop_backward(construct_y(vector[0]))

    def __handle_train_batch(self, batch):
        self.batch_size = batch.shape[0]
        np.apply_along_axis(self.__handle_single_vector, 1, batch)
        self.__handle_apply_gradient_accumulator()
        self.__handle_clear_gradient_accumulator()

    # -----------------
    # External handlers
    # -----------------

    def train(self, df_train, epochs, batch_size):

        matrix_train = df_train.to_numpy()
        q_batches = matrix_train.shape[0] // batch_size

        for e in range(epochs):
            print(f"--------------- EPOCH {e} ---------------")
            np.random.shuffle(matrix_train)
            batches = np.array(np.array_split(matrix_train, q_batches))

            for batch in batches:
                print('<BATCH>')

                self.__handle_train_batch(batch)

                print('</BATCH>\n\n')

            print(f"--------------- END OF EPOCH {e} ---------------")
