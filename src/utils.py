import numpy as np


def construct_y(y):
    Y = np.array([1 if y == i else 0 for i in range(0, 10)])
    return Y

def create_gradient_accum(layers_sizes):
    layer_0_size, layer_1_size, layer_2_size, layer_3_size = layers_sizes
    return {
        'weights': {
            'W_1': np.zeros((layer_1_size, layer_0_size)),
            'W_2': np.zeros((layer_2_size, layer_1_size)),
            'W_3': np.zeros((layer_3_size, layer_2_size))
        },
        'biases': {
            'B_1': np.zeros(layer_1_size),
            'B_2': np.zeros(layer_2_size),
            'B_3': np.zeros(layer_3_size)
        }
    }


def log(m):
    DEBUG = False
    if DEBUG:
        print("")
        print(m)
        print('-----------')
