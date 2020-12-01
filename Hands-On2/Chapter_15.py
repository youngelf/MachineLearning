import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_datasets as tfds

import numpy as np

# Chapter 15: RNNs and CNNs for time series data.

# Creating time series
def generate_time_series(batch_size, n_steps):
    freq1, freq2, offset1, offset2 = np.random.random.rand(4, batch_size, 1)
    time = np.linspace(0, 1, n_steps)

    series = 0.5 * np.sin((time - offset1) * (freq1 * 10 + 10)) # wave 1
    series += 0.2 * np.sin((time - offset2) * (freq2 * 20 + 20)) # wave 2
    series += 0.1 * (np.random.rand(batch_size, n_steps) - 0.5) # noise        

    return series[..., np.newaxis].astype(np.float32)


def time_series_data():
    n_steps = 50
    series = generate_time_series(10000, n_steps + 1)
    X_train, y_train = series[:7000,:n_steps], series[:7000, -1]
    X_valid, y_valid = series[7000:9000,:n_steps], series[7000:9000, -1]
    X_test, y_test = series[9000:,:n_steps], series[9000:, -1]    

    return X_train, y_train, X_valid, y_valid, X_test, y_test

