import numpy as np
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.utils import plot_model


def Autoencoder(series_length):
    """
    Return a keras model of autoencoder
    :param series_length:
    :return:
    """
    model = Sequential()
    model.add(LSTM(100, activation='relu', input_shape=(series_length, 1)))
    model.add(RepeatVector(series_length))
    model.add(LSTM(100, activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(1)))
    model.compile(optimizer='adam', loss='mse')
    return model


if __name__ == '__main__':
    # define input sequence
    sequence = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    # reshape input into [samples, timesteps, features]
    n_in = len(sequence)
    sequence = sequence.reshape((1, n_in, 1))
    # define model
    model = Autoencoder(series_length=n_in)
    # fit model
    model.fit(sequence, sequence, epochs=300, verbose=0)
    plot_model(model, show_shapes=True, to_file='reconstruct_lstm_autoencoder.png')
    # demonstrate recreation
    yhat = model.predict(sequence, verbose=0)
    print(yhat[0, :, 0])
