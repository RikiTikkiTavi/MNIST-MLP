import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from src.mlp import MLP


def main():
    df = pd.read_csv('./data/train.csv')
    X_train = (df.drop(['label'], axis=1)).to_numpy()
    Y_Train = (df[['label']]).to_numpy()
    print(Y_Train.shape)
    print(X_train.shape)
    mlp = MLP()
    mlp.train(X_train, Y_Train, 1, 50)


if __name__ == '__main__':
    np.random.seed(0)
    main()
