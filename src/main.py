import math

import sklearn as sk
import pandas as pd
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

from src.mlp import MLP


def main():
    df = pd.read_csv('./data/train.csv')
    mlp = MLP()

    for index, row in df.iterrows():
        vector = row.to_numpy()
        # mlp.forward_propagation(vector[1:])
        # mlp.backward_propagation(construct_y(vector[0]))


if __name__ == 'main':
    np.random.seed(0)
    main()
