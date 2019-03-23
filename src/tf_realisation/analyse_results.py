import matplotlib
import matplotlib.pyplot as plt
import pandas as pd


def get_best(filename):
    data = pd.read_csv(filename)
    best_params = data.loc[data['val_categorical_accuracy'].idxmax(), :]
    print(best_params.to_dict())


def main():
    get_best('p1.csv')
    get_best('p2.csv')


main()
