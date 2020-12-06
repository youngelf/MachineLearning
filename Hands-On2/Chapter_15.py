import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_datasets as tfds

import numpy as np
import Chapter_10 as c10

# Chapter 15: RNNs and CNNs for time series data.

# Creating time series
def generate_time_series(batch_size, n_steps):
    """Create a mixture of sinusoidal data that we can generate easily.

    Meant to be called from time_series_data()
    series = generate_time_series()

    But then it needs to be split into training/test and features versus labels.
    """
    freq1, freq2, offset1, offset2 = np.random.rand(4, batch_size, 1)
    time = np.linspace(0, 1, n_steps)

    series = 0.5 * np.sin((time - offset1) * (freq1 * 10 + 10)) # wave 1
    series += 0.2 * np.sin((time - offset2) * (freq2 * 20 + 20)) # wave 2
    series += 0.1 * (np.random.rand(batch_size, n_steps) - 0.5) # noise

    return series[..., np.newaxis].astype(np.float32)


def time_series_data():
    """Generate time series data for experimenting with.

    This gives a series of 7k labeled training data with a single label to predict.
    The validation data has 2k observations, and the test data has 1k observations.

    All are drawn from the sinusoidal time series from generate_time_series

    Call with:
     X_train, y_train, X_valid, y_valid, X_test, y_test = time_series_data()
    """
    n_steps = 50
    series = generate_time_series(10000, n_steps + 1)
    X_train, y_train = series[:7000,:n_steps], series[:7000, -1]
    X_valid, y_valid = series[7000:9000,:n_steps], series[7000:9000, -1]
    X_test, y_test = series[9000:,:n_steps], series[9000:, -1]

    return X_train, y_train, X_valid, y_valid, X_test, y_test


def naive_estimate(X_valid, y_valid):
    """Find out the accuracy of just returning the last value observed.

    Call with:
    mse = naive_estimate(X_valid, y_valid)
    """
    # Prediction is the last observation. Discard the rest
    y_pred = X_valid[:, -1]
    mse = np.mean(keras.losses.mean_squared_error(y_valid, y_pred))
    print ("Mean Squared Error = ", mse)
    return mse


# Yup, this is good. With a few epochs of training, MSE is .0073, and
# that's similar to the validation loss.  Would be fun to try making
# this a deep model and see how far we can get without have to use
# RNNs.

# Epoch 10/10
# 219/219 [==============================] - 1s 3ms/step - loss: 0.0073 - mse: 0.0073 - val_loss: 0.0073 - val_mse: 0.0073

def linear_model(input_shape=[50, 1], output_length=1, optimizer="sgd", testing=False):
    """Generate a naive Linear Estimate model

    This feeds all input data to a single Dense Linear Model.
    Call with:
    model = linear_model()

    model = linear_model(input_shape=[100, 1])
    """
    model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=input_shape),
        keras.layers.Dense(output_length)
    ])

    # MeanSquaredError loss because this is a regression problem, not categorical.
    model.compile(loss=tf.keras.losses.MeanSquaredError(),
                  optimizer=optimizer,
                  metrics=["mse"])

    return model

# Let's try a model that's a tad more complicated that what we've got.

# The intuition was good. Losses reduce! 0.0037 on test and validation!
# Epoch 10/10
# 219/219 [==============================] - 4s 17ms/step - loss: 0.0037 - mse: 0.0037 - val_loss: 0.0037 - val_mse: 0.0037

# More training didn't help. Error reduced to .0031. This is within
# range of the RNN models described later in the book.
#
# Epoch 500/500
# 219/219 [==============================] - 4s 17ms/step - loss: 0.0029 - mse: 0.0029 - val_loss: 0.0031 - val_mse: 0.0031

def involved_linear_model(input_shape=[50, 1], output_length=1, optimizer="sgd", testing=False):
    """Generate a naive Linear Estimate model

    This feeds all input data to a single Dense Linear Model.
    Call with:
    model = involved_linear_model()

    model = involved_linear_model(input_shape=[100, 1])
    """
    model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=input_shape),
        # Try doing a CNN style expansion and contraction. What happens?
        keras.layers.Dense(100),
        keras.layers.Dense(200),
        keras.layers.Dense(100),
        keras.layers.Dense(output_length)
    ])

    # MeanSquaredError loss because this is a regression problem, not categorical.
    model.compile(loss=tf.keras.losses.MeanSquaredError(),
                  optimizer=optimizer,
                  metrics=["mse"])

    return model

# Epoch 10/10
# 219/219 [==============================] - 9s 42ms/step - loss: 0.0047 - mse: 0.0047 - val_loss: 0.0043 - val_mse: 0.0043
#
# 10 epochs produced 0.0043 error. Not good enough. Let's try more epochs.

# Epoch 57/100
# 219/219 [==============================] - 9s 42ms/step - loss: 0.0029 - mse: 0.0029 - val_loss: 0.0029 - val_mse: 0.0029
#
# At 67 epochs (10 earlier, 57 now, it is at 0.0029 validation mse which is pretty reasonable.
# Epoch 100/100
# 219/219 [==============================] - 9s 42ms/step - loss: 0.0028 - mse: 0.0028 - val_loss: 0.0029 - val_mse: 0.0029
# After 110, mse is 0.0029. Decent
def basic_rnn_model(input_shape=[None, 1], output_length=1, optimizer="sgd", testing=False):
    """Generate a naive RNN model

    This feeds all input data to a single Dense Linear Model.
    Call with:
    model = basic_rnn_model()

    model = basic_rnn_model(input_shape=[100, 1])

    This model takes in an input shape of [None, 1] because it is a sequence to vector model.
    """
    model = keras.models.Sequential([
        # Grab the input element, and return all the sequences to the next layer
        keras.layers.SimpleRNN(20, return_sequences=True, input_shape=input_shape),
        # Next layer is an RNN, so return all the sequences to the next layer
        keras.layers.SimpleRNN(20, return_sequences=True),
        # Return the single element, this is the output so don't return sequences.
        keras.layers.SimpleRNN(output_length),
    ])

    # MeanSquaredError loss because this is a regression problem, not categorical.
    model.compile(loss=tf.keras.losses.MeanSquaredError(),
                  optimizer=optimizer,
                  metrics=["mse"])

    return model

# After 50 epochs, validation mse is 0.0031. Great.
# Epoch 40/40
# 219/219 [==============================] - 6s 29ms/step - loss: 0.0032 - mse: 0.0032 - val_loss: 0.0031 - val_mse: 0.0031
# 
def basic_rnn_withDense(input_shape=[None, 1], output_length=1, optimizer="sgd", testing=False):
    """Generate a naive RNN model with a dense last layer. This model is
    similar to the basic_rnn_model() method's output

    This feeds all input data to a single Dense Linear Model.
    Call with:
    model = basic_rnn_withDense()

    model = basic_rnn_withDense(input_shape=[100, 1])

    This model takes in an input shape of [None, 1] because it is a sequence to vector model.

    """
    model = keras.models.Sequential([
        # Grab the input element, and return all the sequences to the next layer
        keras.layers.SimpleRNN(20, return_sequences=True, input_shape=input_shape),
        # Next layer is a dense layer, so don't return sequences
        keras.layers.SimpleRNN(20),
        # Return the single element, this is the output so don't return sequences.
        keras.layers.Dense(output_length),
    ])

    # MeanSquaredError loss because this is a regression problem, not categorical.
    model.compile(loss=tf.keras.losses.MeanSquaredError(),
                  optimizer=optimizer,
                  metrics=["mse"])

    return model


# Generate sequence to sequence training data: n_steps of training, and 10 steps of output
def time_series_10label_data():
    """Generate time series data for experimenting with. This produces
    labeled data that has 10 elements that are to be guessed

    This gives a series of 7k labeled training data with a single label to predict.
    The validation data has 2k observations, and the test data has 1k observations.

    All are drawn from the sinusoidal time series from generate_time_series

    Call with:
     X_train, y_train, X_valid, y_valid, X_test, y_test = time_series_10label_data()

    """
    n_steps = 50
    series = generate_time_series(10000, n_steps + 10)
    X_train, y_train = series[:7000,:n_steps], series[:7000, n_steps:, 0]
    X_valid, y_valid = series[7000:9000,:n_steps], series[7000:9000, n_steps:, 0]
    X_test, y_test = series[9000:,:n_steps], series[9000:, n_steps:, 0]

    return X_train, y_train, X_valid, y_valid, X_test, y_test



def fit_model(model, X_train, y_train,
              X_valid, y_valid, epochs,
              batch_size=32, verbose=0,
              test_shapes=True):
    """Fit a model

    Call with:
    history = fit_model(model, X_train, y_train, X_valid,
                        y_valid, epochs=10, verbose=1, test_shapes=True)
    """

    # This fails after a day, when the validation data is incorrectly shaped.
    # This is a terrible idea. Failures should be early.

    # The best way to guard against it is to run a small fit run with
    # a tiny data size, and a tiny validation data size to ensure that
    # the data is correctly shaped.

    if (test_shapes):
        print ("Testing with the first 10 elements of the input")
        X_small = X_train[:10,:,:]
        y_small = y_train[:10]
        X_valid_small = X_valid[:10,:,:]
        y_valid_small = y_valid[:10]
        # Call ourselves again with a smaller input. This confirms
        # that the two methods are calling the same .fit() method, and
        # that the input is correctly shaped in the original method too.

        # The output is discarded, and that's ok. We don't care about
        # the model training or the history. If this passed, the input
        # must be correctly shaped and the correct dtype.
        fit_model(model, X_small, y_small,
                     X_valid_small, y_valid_small,
                     epochs=epochs, verbose=verbose,
                     batch_size=batch_size,
                     test_shapes=False)

    # If that worked, then do a full run.
    history_conv = model.fit(x=X_train, y=y_train, batch_size=batch_size,
                             validation_data=(X_valid, y_valid),
                             epochs=epochs, verbose=verbose)
    return history_conv

def plot_history(history, name):
    c10.plot_training(history, name, show=False)
