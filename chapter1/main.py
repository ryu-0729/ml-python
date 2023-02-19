import tensorflow as tf
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense


def check_version():
    print(tf.__version__)


def create_model():
    ten = Dense(units=1, input_shape=[1])
    model = Sequential([ten])
    model.compile(optimizer='sgd', loss='mean_squared_error')

    xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
    ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

    model.fit(xs, ys, epochs=500)

    print(model.predict([10.0]))
    print(f'Here is what I learned: {ten.get_weights()}')


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    check_version()
    create_model()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
