import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from src.mlp import MLP


def main():
    df = pd.read_csv('./data/train.csv')
    mlp = MLP()
    mlp.train(df, 1, 100)


if __name__ == '__main__':
    np.random.seed(0)
    main()
