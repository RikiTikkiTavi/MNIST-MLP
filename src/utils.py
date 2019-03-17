import numpy as np


def construct_y(y):
    Y = np.array([i if y == i else 0 for i in range(0, 10)])
    return Y
