import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from src.mlp import MLP


def main():
    df = pd.read_csv('./data/train.csv')
    df = df.head(n=1000)
    X_train = (df.drop(['label'], axis=1))
    Y_Train = (df[['label']])
    X_train, X_test, y_train, y_test = train_test_split(X_train, Y_Train, test_size=0.33, random_state=42)
    mlp = MLP()
    mlp.train(X_train.to_numpy(), y_train.to_numpy(), 4, 100)
    result = mlp.predict(X_test.to_numpy())
    y_test = (y_test.to_numpy()).flatten()
    print(f"Result\n: {result}")
    print(f"Result shape: {result.shape}")
    print(f"Correct result\n: {y_test}")
    print(f"Correct result shape: {y_test.shape}")

    print(np.sum(result == y_test)/result.shape[0])


if __name__ == '__main__':
    np.random.seed(0)
    main()
