import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras
import talos

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
import pandas as pd


def get_data():
    df = pd.read_csv('../data/train.csv')
    X_train = (df.drop(['label'], axis=1))
    Y_Train = (df[['label']])
    X_train, X_test, y_train, y_test = train_test_split(X_train, Y_Train, test_size=0.33, random_state=42)
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
                        epochs=25,
                        shuffle=True,
                        use_multiprocessing=True,
                        batch_size=params['batch_size'],
                        verbose=1)

    # Plot training & validation accuracy values
    plt.plot(history.history['categorical_accuracy'])
    plt.title(f'Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    return history, model


def main():
    x_train, x_test, y_train, y_test = get_data()
    y_train = encode_y(y_train)
    y_test = encode_y(y_test)

    params = {
        'lr': [0.001, 0.01, 0.1],
        'decay': [0.00001, 0.0001, 0.001],
        'momentum': [0.9, 0.945, 0.99],
        'batch_size': [100]
    }

    talos.Scan(model=mnist_model,
               x=x_train,
               y=y_train,
               x_val=x_test,
               y_val=y_test,
               params=params,
               print_params=True,
               clear_tf_session=False)


main()
