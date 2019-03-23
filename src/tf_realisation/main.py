import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras
import talos
import os

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
import pandas as pd

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def get_data():
    df = pd.read_csv('../data/train.csv')
    X_train = (df.drop(['label'], axis=1))
    Y_Train = (df[['label']])
    X_train, X_test, y_train, y_test = train_test_split(X_train, Y_Train, test_size=0.2, random_state=42)
    return X_train.to_numpy(), X_test.to_numpy(), y_train.to_numpy().flatten(), y_test.to_numpy().flatten()


def _one_hot(x):
    return [1 if i == x else 0 for i in range(10)]


def encode_y(Y):
    return np.array(list(map(_one_hot, Y)))


def mnist_model(x_train, y_train, x_test, y_test, params):
    model = keras.Sequential([
        keras.layers.Dense(units=397, input_dim=784, activation=tf.nn.tanh),
        keras.layers.Dropout(rate=0.15),
        keras.layers.Dense(units=397, activation=tf.nn.relu),
        keras.layers.Dropout(rate=0.15),
        keras.layers.Dense(units=10, activation=tf.nn.tanh)
    ])
    sgd = keras.optimizers.SGD(lr=params['lr'], decay=params['decay'], momentum=params['momentum'], nesterov=True)
    model.compile(optimizer=sgd,
                  loss='mse',
                  metrics=['categorical_accuracy'])

    history = model.fit(x=x_train,
                        y=y_train,
                        validation_data=(x_test, y_test),
                        epochs=150,
                        shuffle=True,
                        use_multiprocessing=True,
                        batch_size=params['batch_size'],
                        verbose=1)

    return history, model


def _create_param_list(p_d):
    return [p_d[0] - p_d[1], p_d[0], p_d[0] + p_d[1]]


def _handle_create_param_list(p_d):
    return _create_param_list(p_d) if p_d[2] else [p_d[0]]


def create_params(params_bases):
    # 0 - base, 1 - range, 3 - tune
    lr_data, decay_data, momentum_data, batch_size_data = params_bases
    return {
        'lr': _handle_create_param_list(lr_data),
        'decay': _handle_create_param_list(decay_data),
        'momentum': _handle_create_param_list(momentum_data),
        'batch_size': _handle_create_param_list(batch_size_data)
    }


def get_best_from_scan_results(filename):
    data = pd.read_csv(filename)
    best_params = data.loc[data['val_categorical_accuracy'].idxmax(), :]
    return best_params.to_dict()


def _get_data():
    x_train, x_test, y_train, y_test = get_data()
    y_train = encode_y(y_train)
    y_test = encode_y(y_test)
    return x_train, x_test, y_train, y_test


def _do_scan(data, params, name):
    x_train, x_test, y_train, y_test = data
    print(params)
    talos.Scan(model=mnist_model,
               x=x_train,
               y=y_train,
               x_val=x_test,
               y_val=y_test,
               params=params,
               dataset_name=name,
               print_params=True,
               clear_tf_session=False)


def handle_pipeline_1(start_params_bases):
    params_bases = start_params_bases
    data = _get_data()
    for i in range(4):
        name = f'e{i}'
        _do_scan(data=data, params=create_params(tuple(params_bases.values())), name=name)
        best_params = get_best_from_scan_results(f'{name}_.csv')
        print(f"\nBEST PARAMS:\n {best_params} \n")
        params_bases = {
            'lr': (best_params['lr'], best_params['lr'] / 10, True),
            'decay': (best_params['decay'], best_params['decay'] / 10, True),
            'momentum': (best_params['momentum'], best_params['momentum'] / 100, True if i < 1 else False),
            'batch_size': (int(best_params['batch_size']), 5, True if i < 1 else False)
        }


def handle_pipeline_2(best_params):
    ta_params = {
        'lr': _handle_create_param_list((best_params['lr'], 0.001, True)),
        'decay': _handle_create_param_list((best_params['decay'], 10 ** -6, True)),
        'momentum': [best_params['momentum']],
        'batch_size': _handle_create_param_list((int(best_params['batch_size']), 3, True))
    }
    _do_scan(_get_data(), ta_params, 'a0')
    print(get_best_from_scan_results('a0_.csv'))


def get_experiments_data():
    exp_data = pd.read_csv(f'e0_.csv')
    for i in range(1, 4):
        pd.concat([exp_data, pd.read_csv(f'e{i}_.csv')])
    return exp_data


def handle_result_of_pipeline_2():
    best_params = get_best_from_scan_results('a0_.csv')
    best_params['batch_size'] = int(best_params['batch_size'])
    print(best_params)
    x_train, x_test, y_train, y_test = _get_data()
    history, model = mnist_model(x_train, y_train, x_test, y_test, best_params)
    model.evaluate(x=x_test, y=y_test, verbose=1, use_multiprocessing=True)
    model.save('mnist_mlp_sgd_mse_model.h5')


def main():
    start_params_bases = {
        'lr': (0.01, 0.001, True),
        'decay': (0.0001, 0.00002, True),
        'momentum': (0.945, 0.005, True),
        'batch_size': (100, 10, True)
    }
    # handle_pipeline_1(start_params_bases)
    exp_data = get_experiments_data()
    best_params = (exp_data.loc[exp_data['val_categorical_accuracy'].idxmax(), :]).to_dict()
    best_params['batch_size'] = int(best_params['batch_size'])
    print(best_params)
    handle_pipeline_2(best_params)


handle_result_of_pipeline_2()
