import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
import pandas as pd


def get_data():
    df = pd.read_csv('../data/train.csv')
    X_train = (df.drop(['label'], axis=1))
    Y_Train = (df[['label']])
    X_train, X_test, y_train, y_test = train_test_split(X_train, Y_Train, test_size=0.3, random_state=42)
    return X_train.to_numpy(), X_test.to_numpy(), y_train.to_numpy().flatten(), y_test.to_numpy().flatten()


def _one_hot(x):
    return [1 if i == x else 0 for i in range(10)]


def encode_y(Y):
    return np.array(list(map(_one_hot, Y)))


x_train, x_test, y_train, y_test = get_data()
y_train = encode_y(y_train)
y_test = encode_y(y_test)

model = keras.Sequential([
    keras.layers.Dense(units=397, input_shape=(784,), activation=tf.nn.tanh),
    keras.layers.Dense(units=397, activation=tf.nn.relu),
    keras.layers.Dense(units=10, activation=tf.nn.tanh)
])

sgd = keras.optimizers.SGD(lr=0.01, decay=0, momentum=0.2, nesterov=False)

model.compile(optimizer=sgd, loss='mse', metrics=['categorical_accuracy'])

model.summary()

history = model.fit(x=x_train, y=y_train, epochs=50, use_multiprocessing=True, shuffle=True, batch_size=100)

# Plot training & validation accuracy values
plt.plot(history.history['categorical_accuracy'])
plt.title('Model accuracy')
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

test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test accuracy:', test_acc)
