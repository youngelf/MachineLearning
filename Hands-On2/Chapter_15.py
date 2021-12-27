import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_datasets as tfds

import numpy as np
import Chapter_10 as c10
import pandas as pd

from pathlib import Path
from IPython.display import Audio


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

# And on the sequence of 10 output, 50 epochs gives a low 0.0076 mse
# Epoch 50/50
# 219/219 [==============================] - 6s 30ms/step - loss: 0.0082 - mse: 0.0082 - val_loss: 0.0076 - val_mse: 0.0076

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

def basic_lstm_distributed(input_shape=[None, 1], output_length=10,
                           optimizer="adam", testing=False, only_last_mse=True):
    """Generate a naive RNN model with a dense last layer. This model is
    similar to the basic_rnn_model() method's output

    This feeds all input data to a single Dense Linear Model.
    Call with:
    model = basic_lstm_distributed()

    model = basic_lstm_distributed(input_shape=[100, 1])

    This model takes in an input shape of [None, 1] because it is a sequence to vector model.

    """


    # When feeding the sequence to sequence models, then we don't care
    # about the loss of the full training run, just the final output.
    # So we can use this loss function where only the last MSE is
    # retained, all else are discarded.
    def last_time_step_mse(Y_true, Y_pred):
        """ Mean squared error of the last step only """
        return keras.metrics.mean_squared_error(Y_true[:, -1], Y_pred[:, -1])

    model = keras.models.Sequential([
        # Grab the input element, and return all the sequences to the next layer
        keras.layers.LSTM(50, return_sequences=True, input_shape=input_shape),
        # Next layer is a dense layer, so don't return sequences
        keras.layers.LSTM(50, return_sequences=True),
        # Return the single element, this is the output so don't return sequences.
        keras.layers.TimeDistributed(keras.layers.Dense(output_length)),
    ])

    # MeanSquaredError loss because this is a regression problem, not categorical.
    if(only_last_mse):
        metrics = last_time_step_mse
    else:
        metrics = "mse"

    model.compile(loss=tf.keras.losses.MeanSquaredError(),
                  optimizer=optimizer,
                  metrics=[metrics])

    return model

# This is the best model. A 1D convolution, and two layers of GRU units.
# To fit, you have to run this:
#
# Epoch 50/50
# 219/219 [==============================] - 9s 43ms/step - loss: 0.0121 - last_time_step_mse: 0.0032 - val_loss: 0.0124 - val_last_time_step_mse: 0.0033
# MSE is 0.0033, very impressive.

# To fit, run this:
# history = fit_model(model, X_train, y_train[:, 3::2], X_valid, y_valid[:,3::2], epochs=50, verbose =1, test_shapes=True)
def basic_gru_distributed(input_shape=[None, 1], output_length=10,
                          optimizer="adam", only_last_mse=True):
    """Generate a naive RNN model with a dense last layer. This model is
    similar to the basic_rnn_model() method's output

    This feeds all input data to a single Dense Linear Model.
    Call with:
    model = basic_gru_distributed()
    history = fit_model(model, X_train, y_train[:, 3::2], X_valid, y_valid[:,3::2], epochs=50, verbose =1, test_shapes=True)


    model = basic_gru_distributed(input_shape=[100, 1])

    This model takes in an input shape of [None, 1] because it is a sequence to vector model.

    """


    # When feeding the sequence to sequence models, then we don't care
    # about the loss of the full training run, just the final output.
    # So we can use this loss function where only the last MSE is
    # retained, all else are discarded.
    def last_time_step_mse(Y_true, Y_pred):
        """ Mean squared error of the last step only """
        return keras.metrics.mean_squared_error(Y_true[:, -1], Y_pred[:, -1])

    model = keras.models.Sequential([
        # Grab the input element, and return all the sequences to the next layer
        keras.layers.Conv1D(filters=20, kernel_size=4, strides=2,
                            padding="valid", input_shape=input_shape),
        keras.layers.GRU(20, return_sequences=True, input_shape=input_shape),
        # Next layer is a dense layer, so don't return sequences
        keras.layers.GRU(20, return_sequences=True),
        # Return the single element, this is the output so don't return sequences.
        keras.layers.TimeDistributed(keras.layers.Dense(output_length)),
    ])

    # MeanSquaredError loss because this is a regression problem, not categorical.
    if(only_last_mse):
        metrics = last_time_step_mse
    else:
        metrics = "mse"

    model.compile(loss="mse",
                  optimizer=optimizer,
                  metrics=[metrics])

    return model


# Generate sequence to vector training data: n_steps of training, and 10 steps of output
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


# Generate sequence to sequence training data: n_steps of training,
# and 10 steps of output at *every* input batch. This is different
# from the previous run in that time_series_10label_data outputs 10
# values at the end of all of the input. This method outputs 10
# valueas at every batch.
def sequence_to_sequence_data():
    """Generate time series data for experimenting with. This produces
    labeled data that has 10 elements that are to be guessed

    This gives a series of 7k labeled training data with a single label to predict.
    The validation data has 2k observations, and the test data has 1k observations.

    All are drawn from the sinusoidal time series from generate_time_series

    Call with:
     X_train, y_train, X_valid, y_valid, X_test, y_test = sequence_to_sequence_data()

    """
    n_steps = 50
    series = generate_time_series(10000, n_steps + 10)
    Y = np.empty((10000,n_steps, 10)) # Each target is a sequence of 10D vectors

    for step_ahead in range(1, 10+1):
        Y[:, :, step_ahead - 1] = series[:, step_ahead:step_ahead + n_steps, 0]

    X_train = series[:7000,:n_steps]
    X_valid = series[7000:9000,:n_steps]
    X_test = series[9000:,:n_steps]

    y_train = Y[:7000]
    y_valid = Y[7000:9000]
    y_test = Y[9000:]

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


def run_all_15():
    """ Run all Chapter 15 code """

    # Simple time series data. 1 label after n_step(50 by default)
    X_train, y_train, X_valid, y_valid, X_test, y_test = time_series_data()

    # Generate an RNN model
    model = basic_rnn_model()
    history = fit_model(model, X_train, y_train, X_valid,y_valid, epochs=100, verbose=1, test_shapes=True)

    # This has a dense layer at the end rather than a single RNN output.
    model = basic_rnn_withDense()
    history = fit_model(model, X_train, y_train, X_valid,y_valid, epochs=50, verbose=1, test_shapes=True)


    # Now get data with 10 elements as the output, and fit that
    X_train, y_train, X_valid, y_valid, X_test, y_test = time_series_10label_data()
    model = basic_rnn_withDense(output_length=10, optimizer="adam")
    history = fit_model(model, X_train, y_train, X_valid,y_valid, epochs=50, verbose=1, test_shapes=True)


    # This data is 10 elements at every time step. The LSTM and GRU
    # model generation code above expects this
    X_train, y_train, X_valid, y_valid, X_test, y_test = sequence_to_sequence_data()

    model = basic_lstm_distributed()
    history = fit_model(model, X_train, y_train, X_valid,y_valid, epochs=50, verbose=1, test_shapes=True)

    # Best model.
    model = basic_gru_distributed()
    history = fit_model(model, X_train, y_train[:, 3::2], X_valid, y_valid[:,3::2], epochs=50, verbose=1, test_shapes=True)

print ("All done. Exercises are next")


# Exercises.

# E1.
# sequence to sequence: translating English to Greek.
# sequence to vector: Predicting movement of stock price
# Vector to sequence: Generate music for a genre


# E2.
# Dimensions of input RNN layer: [observations, time steps, input
# features]. Output is [observations, output features]


# E3.
# sequence to sequence
# All should set return_sequences=True.

# sequence to vector
# All but the last RNN layer should set return_sequences=True. If the
# last layer is a Dense layer, then its prior layer should not
# return_sequences=True.


# E4. If the series is stationary, I'd use a sequence to vector RNN
# with LSTMs or GRU cells. The last layer would be a Dense layer with
# 7 outputs.

# E5. Vanishing gradients are the biggest difficulty. Either choosing
# to normalize over features or choosing to normalize over time is one
# technique. Choosing simpler models rather than complicated models,
# or using Convolutional layers is another mechanism to increase
# information flow without increasing complexity.

# E6. on paper

# E7. To capture longer-term information, multigrid-style in a single
# convolution that can have a variety of filters for the longer-term
# pattern.

# E8. This sounds complicated.

# I would probably use a CNN on individual frames, probably extract
# critical frames. The difficulty stems from the 2d frames, and the
# additional dimension of time. I guess this is one additional input
# to the CNN, where there is CNN to extract the spatial information
# and an RNN to extract the temporal information. Can the CNN layers
# be used in a time-distributed way?

# E9. Trying to classify the SketchRNN dataset. I guess the author
# wants the QuickDraw dataset that corresponds to the SketchRNN
# model. I really hate such mistakes that passed through in the book.
# I guess the pull request is still stuck, but it is available as a tfrecord.

# From:
# https://github.com/ageron/handson-ml2/blob/master/15_processing_sequences_using_rnns_and_cnns.ipynb


def load_sketch_dataset():
    """Loads the sketch dataset.

    Call with:
    train_set, valid_set, test_set = load_sketch_dataset()
    """
    DOWNLOAD_ROOT = "http://download.tensorflow.org/data/"
    FILENAME = "quickdraw_tutorial_dataset_v1.tar.gz"

    # Information about the dataset is provided here:
    # https://github.com/googlecreativelab/quickdraw-dataset

    # The magic to read this dataset comes from here:
    # https://raw.githubusercontent.com/tensorflow/models/r1.13.0/tutorials/rnn/quickdraw/train_model.py
    # Again, a completely opaque dataset with a magic set of instructions to get the data out.
    
    # Download the 1G dataset
    filepath = keras.utils.get_file(FILENAME,
                                    DOWNLOAD_ROOT + FILENAME,
                                    cache_subdir="datasets/quickdraw",
                                    extract=True)

    # Get names of all the files
    quickdraw_dir = Path(filepath).parent
    train_files = sorted([str(path) for path in quickdraw_dir.glob("training.tfrecord-*")])
    eval_files = sorted([str(path) for path in quickdraw_dir.glob("eval.tfrecord-*")])

    # Get all the training and test category (classes) names
    with open(quickdraw_dir / "eval.tfrecord.classes") as test_classes_file:
        test_classes = test_classes_file.readlines()

    with open(quickdraw_dir / "training.tfrecord.classes") as train_classes_file:
        train_classes = train_classes_file.readlines()
                    
    # Make sure they are identical, otherwise we have a problem
    assert train_classes == test_classes
    # Normalize the class names, and show them
    class_names = [name.strip().lower() for name in train_classes]
    print(sorted(class_names))

    # Return tfrecord data as sketches, lengths and the labels
    def parse(data_batch):
        feature_descriptions = {
            "ink": tf.io.VarLenFeature(dtype=tf.float32),
            "shape": tf.io.FixedLenFeature([2], dtype=tf.int64),
            "class_index": tf.io.FixedLenFeature([1], dtype=tf.int64)
        }
        examples = tf.io.parse_example(data_batch, feature_descriptions)
        flat_sketches = tf.sparse.to_dense(examples["ink"])
        sketches = tf.reshape(flat_sketches, shape=[tf.size(data_batch), -1, 3])
        # How long the 
        lengths = examples["shape"][:, 0]
        # This is what we have to predict
        labels = examples["class_index"][:, 0]
        return sketches, lengths, labels

    # Create a tf dataset from the files
    def quickdraw_dataset(filepaths, batch_size=32, shuffle_buffer_size=None,
                          n_parse_threads=5, n_read_threads=5, cache=False):
        dataset = tf.data.TFRecordDataset(filepaths,
                                          num_parallel_reads=n_read_threads)
        if cache:
            dataset = dataset.cache()
        if shuffle_buffer_size:
            dataset = dataset.shuffle(shuffle_buffer_size)
            dataset = dataset.batch(batch_size)
            # Map the individual data to sketches, lengths and the class labels
            dataset = dataset.map(parse, num_parallel_calls=n_parse_threads)
            return dataset.prefetch(1)

    # Keep the full training set, and don't split out the validation here.
    train_set = quickdraw_dataset(train_files, shuffle_buffer_size=10000)

    # Take the first five eval files for validation
    valid_set = quickdraw_dataset(eval_files[:5])
    # And the rest for the actual test set
    test_set = quickdraw_dataset(eval_files[5:])
    
    return train_set, valid_set, test_set

def draw_sketch(sketch, label=None):
    origin = np.array([[0., 0., 0.]])
    sketch = np.r_[origin, sketch]
    stroke_end_indices = np.argwhere(sketch[:, -1]==1.)[:, 0]
    coordinates = np.cumsum(sketch[:, :2], axis=0)
    strokes = np.split(coordinates, stroke_end_indices + 1)
    title = class_names[label.numpy()] if label is not None else "Try to guess"
    plt.title(title)
    plt.plot(coordinates[:, 0], -coordinates[:, 1], "y:")
    for stroke in strokes:
        plt.plot(stroke[:, 0], -stroke[:, 1], ".-")
        plt.axis("off")
        
def draw_sketches(sketches, lengths, labels):
    n_sketches = len(sketches)
    n_cols = 4
    n_rows = (n_sketches - 1) // n_cols + 1
    plt.figure(figsize=(n_cols * 3, n_rows * 3.5))
    for index, sketch, length, label in zip(range(n_sketches), sketches, lengths, labels):
        plt.subplot(n_rows, n_cols, index + 1)
        draw_sketch(sketch[:length], label)
        plt.show()

def sketch_a_few(train_set):
    for sketches, lengths, labels in train_set.take(1):
        draw_sketches(sketches, lengths, labels)


def load_chorales():
    """Load Bach Chorales dataset

    Call with:
    train_set, valid_set, test_set = load_chorales()
    """
    DOWNLOAD_ROOT = "https://github.com/ageron/handson-ml2/raw/master/datasets/jsb_chorales/"
    FILENAME = "jsb_chorales.tgz"
    filepath = keras.utils.get_file(FILENAME,
                                    DOWNLOAD_ROOT + FILENAME,
                                    cache_subdir="datasets/jsb_chorales",
                                    extract=True)

        
    jsb_chorales_dir = Path(filepath).parent
    train_files = sorted(jsb_chorales_dir.glob("train/chorale_*.csv"))
    valid_files = sorted(jsb_chorales_dir.glob("valid/chorale_*.csv"))
    test_files = sorted(jsb_chorales_dir.glob("test/chorale_*.csv"))

    def load_chorales(filepaths):
            return [pd.read_csv(filepath).values.tolist() for filepath in filepaths]

    train_chorales = load_chorales(train_files)
    valid_chorales = load_chorales(valid_files)
    test_chorales = load_chorales(test_files)

    # Validate the ends of the spectrum of notes in the database.
    notes = set()
    for chorales in (train_chorales, valid_chorales, test_chorales):
        for chorale in chorales:
            for chord in chorale:
                notes |= set(chord)
                
    n_notes = len(notes)
    min_note = min(notes - {0})
    max_note = max(notes)

    assert min_note == 36
    assert max_note == 81
                                        
    train_set = bach_dataset(train_chorales, min_note=min_note, shuffle_buffer_size=1000)
    valid_set = bach_dataset(valid_chorales, min_note=min_note)
    test_set = bach_dataset(test_chorales, min_note=min_note)
    
    return train_set, valid_set, test_set
    
def notes_to_frequencies(notes):
    # Frequency doubles when you go up one octave; there are 12 semi-tones
    # per octave; Note A on octave 4 is 440 Hz, and it is note number 69.
    return 2 ** ((np.array(notes) - 69) / 12) * 440

def frequencies_to_samples(frequencies, tempo, sample_rate):
    note_duration = 60 / tempo # the tempo is measured in beats per minutes
    # To reduce click sound at every beat, we round the frequencies to try to
    # get the samples close to zero at the end of each note.
    frequencies = np.round(note_duration * frequencies) / note_duration
    n_samples = int(note_duration * sample_rate)
    time = np.linspace(0, note_duration, n_samples)
    sine_waves = np.sin(2 * np.pi * frequencies.reshape(-1, 1) * time)
    # Removing all notes with frequencies ≤ 9 Hz (includes note 0 = silence)
    sine_waves *= (frequencies > 9.).reshape(-1, 1)
    return sine_waves.reshape(-1)

def chords_to_samples(chords, tempo, sample_rate):
    freqs = notes_to_frequencies(chords)
    freqs = np.r_[freqs, freqs[-1:]] # make last note a bit longer
    merged = np.mean([frequencies_to_samples(melody, tempo, sample_rate)
                     for melody in freqs.T], axis=0)
    n_fade_out_samples = sample_rate * 60 // tempo # fade out last note
    fade_out = np.linspace(1., 0., n_fade_out_samples)**2
    merged[-n_fade_out_samples:] *= fade_out
    return merged

def play_chords(chords, tempo=160, amplitude=0.1, sample_rate=44100, filepath=None):
    samples = amplitude * chords_to_samples(chords, tempo, sample_rate)
    if filepath:
        from scipy.io import wavfile
        samples = (2**15 * samples).astype(np.int16)
        wavfile.write(filepath, sample_rate, samples)
        return display(Audio(filepath))
    else:
        return display(Audio(samples, rate=sample_rate))
def create_target(batch):
    X = batch[:, :-1]
    Y = batch[:, 1:] # predict next note in each arpegio, at each step
    return X, Y

def preprocess(window, min_note=36):
    window = tf.where(window == 0, window, window - min_note + 1) # shift values
    return tf.reshape(window, [-1]) # convert to arpegio

def bach_dataset(chorales, min_note, batch_size=32, shuffle_buffer_size=None,
                 window_size=32, window_shift=16, cache=True):
    def batch_window(window):
        return window.batch(window_size + 1)

    def to_windows(chorale):
        dataset = tf.data.Dataset.from_tensor_slices(chorale)
        dataset = dataset.window(window_size + 1, window_shift, drop_remainder=True)
        return dataset.flat_map(batch_window)

    chorales = tf.ragged.constant(chorales, ragged_rank=1)
    dataset = tf.data.Dataset.from_tensor_slices(chorales)
    dataset = dataset.flat_map(to_windows).map(preprocess)
    if cache:
        dataset = dataset.cache()
    if shuffle_buffer_size:
        dataset = dataset.shuffle(shuffle_buffer_size)
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(create_target)
    return dataset.prefetch(1)

