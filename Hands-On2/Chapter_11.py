#!/usr/bin/env python
# coding: utf-8

# Chapter 11: Training deep neural networks

# Common imports
import matplotlib.cm as cm
from matplotlib.image import imread
import matplotlib as mpl
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

from sklearn.metrics import accuracy_score
from sklearn.metrics import silhouette_samples
from sklearn.metrics import silhouette_score

from sklearn.datasets import fetch_california_housing
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow import keras

print("TF version ", tf.__version__)
print("Keras version ", keras.__version__)

def load_cifar():
    """ Loads the cifar data and returns it.
    Call with:
    X_train, X_valid, testX, y_train, y_valid, testy = load_cifar()
    """

    # Private to this kind of code, just a size checker
    def size_check(name, expected, observed):
        """Utility method to confirm that expected == observed.
        Call with:
        size_check("X_train", X_train.shape[0], 40000)
        """
        if (observed != expected):
            print ("%s is not the expected size. Expecting: %d, Got: %d",
                   name, expecting, observed)
            return False
        return True


    # Load the actual dataset
    (X, y), (testX, testy) = keras.datasets.cifar10.load_data()

    # Split into training and validation.
    # Also normalize to [0.0, 1.0], dividing by 255.0
    X_train, X_valid = X[:40000] / 255.0, X[40000:] / 255.0

    # Remove the extraneous dimension on the predictions turning the
    # predictions into simple scalars.
    y = y.reshape(50000)
    testy.reshape(10000)
    y_train, y_valid = y[:40000]        , y[40000:]

    # Rudimentary checking on the shapes here. This is what you don't
    # get with the Python notebook style of writing fast and loose
    # cowboy code. Every nontrivial code should confirm its outputs.
    success = (
        size_check ("X_train", X_train.shape[0], 40000) and
        size_check ("X_valid", X_valid.shape[0], 10000) and
        size_check ("testX", testX.shape[0], 10000) and
        size_check ("y_train", y_train.shape[0], 40000) and
        size_check ("y_valid", y_valid.shape[0], 10000) and
        size_check ("testy", testy.shape[0], 10000))

    if (not success):
        print ("Rudimentary checks on data failed. Please examine the earlier output.")
        return False

    # Now, and only now, return the values. The order is unfortunately
    # critical
    return X_train, X_valid, testX, y_train, y_valid, testy


def confirm_cifar(X_train, X_valid, testX, y_train, y_valid, testy):
    """ Print all the variables' shapes.

    Call with:
    confirm_cifar(X_train, X_valid, testX, y_train, y_valid, testy)
    """
    print("Training: ", X_train.shape)
    print("Validation: ", X_valid.shape)
    print("Test: ", testX.shape)

    print("Labels validation: ", y_valid.shape)
    print("Labels training: ", y_train.shape)
    print("Labels test: ", testy.shape)



# A simple model creator, to be provided to KerasClassifier(build_fn=...)
def twenty_dense(n_classes=100):
    """Keras multinomial logistic regression creation model

    Args:
        n_classes(int): Number of classes to be classified

    Returns:
        Compiled keras model

    """
    # create model
    model = keras.models.Sequential()

    # The input: we get 32x32 pixels, each with 3 colors (rgb)
    model.add(keras.layers.Flatten(input_shape=[32,32,3]))
    # Then the hidden layers, fully connected (100 by default)
    for i in range(20):
        model.add(keras.layers.Dense(
            n_classes,
            activation="elu",
            kernel_initializer=tf.keras.initializers.HeNormal()
        ))
    # Now add the output layer: 10 classes in CIFAR10, so 10 outputs.
    model.add(keras.layers.Dense(10, activation="softmax"))

    # print(model.summary())
    # Compile model
    nadam = keras.optimizers.Nadam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07)

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=nadam,
        metrics=["accuracy"]
    )
    return model


# With regularization
def L2regularized(n_classes=100, normalization=False, dropout_rate=-1):
    """Keras multinomial logistic regression creation model

    Args:
        n_classes(int): Number of classes to be classified

        normalization(boolean): True if you want L2 normalization

        dropout_rate(int): -1 for no dropout, positive for dropout
                           rate as a fraction. 0.2 is 20% dropout. If
                           dropout_rate is specified, normalization is done.

    Returns:
        Compiled keras model

    Call with:
    regularized = L2regularized(n_classes=100, normalization=True, dropout_rate=0.2)

    """
    # create model
    model = keras.models.Sequential()

    # The input: we get 32x32 pixels, each with 3 colors (rgb)
    model.add(keras.layers.Flatten(input_shape=[32,32,3]))
    # Then the hidden layers, fully connected (100 by default)
    for i in range(20):
        model.add(keras.layers.Dense(
            n_classes,
            activation="elu",
            kernel_initializer=tf.keras.initializers.HeNormal(),
            kernel_regularizer=tf.keras.regularizers.l2(0.01),
        ))
        if (dropout_rate > 0):
            keras.layers.Dropout(rate=dropout_rate)

        # If dropout rate is specified, then perform normalization
        if (normalization or dropout_rate > 0):
            model.add(keras.layers.BatchNormalization())
            model.add(keras.layers.Activation("elu"))

    # Now add the output layer: 10 classes in CIFAR10, so 10 outputs.
    model.add(keras.layers.Dense(10, activation="softmax"))

    # print(model.summary())
    # Compile model
    nadam = keras.optimizers.Nadam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-07)

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=nadam,
        metrics=["accuracy"]
    )
    return model


estimator = keras.wrappers.scikit_learn.KerasClassifier(
    build_fn=twenty_dense,
    n_classes=10,
    class_weight={0: 1, 1:3})

simplest = twenty_dense(100)
print ("Model built: ", simplest)

def fit(name, model, X_train, y_train, X_valid, y_valid, epochs=30, verbose=0, showFig=False):
    """ Fit a model provided here with number of epochs, and verbosity

    Call with:
    fit("simplest", simplest, X_train, y_train, X_valid, y_valid, epochs=30, verbose=0, showFig=False)
    same as:
    fit("simplest", simplest, X_train, y_train, X_valid, y_valid)

    epochs: Number of epochs to train (integer)
    verbose: whether to print verbose training information (0 or 1)

    Plots the training history in images/<name>.png
    """

    # Fit a model, and train it.
    history = model.fit(X_train, y_train, epochs=epochs, verbose=verbose,
                        validation_data=(X_valid, y_valid))

    # Now plot the history to file
    pd.DataFrame(history.history).plot(figsize=(8,5))
    plt.grid(True)
    plt.gca().set_ylim(0,1) # Y axis set to [0,1]

    if (showFig):
        plt.show()
    else:
        plt.savefig('images/' + name + '.png')

    return history


def evaluate(name, model, testX, testy):
    """ Evaluate the model on the test data.

    Call with:
    loss, accuracy = evaluate("simplest", simplest, testX, testy)

    """

    (loss, accuracy) = model.evaluate(testX, testy)

    print ("Loss is %f" % loss)
    print ("Accuracy is %f" % accuracy)

    return loss, accuracy


def flattened_model(n_classes=100, normalization=False, dropout_rate=-1):
    """Creates multinomial logistic regression creation model with
    flattened input data (3072 pixels rather than row, column, color)

    Args:
        n_classes(int): Number of classes to be classified

        normalization(boolean): True if you want L2 normalization

        dropout_rate(int): -1 for no dropout, positive for dropout
                           rate as a fraction. 0.2 is 20% dropout. If
                           dropout_rate is specified, normalization is done.

    Returns:
        Compiled keras model

    Call with:
    regularized = flattened_model(n_classes=100, normalization=True, dropout_rate=0.2)

    """
    # create model
    model = keras.models.Sequential()

    # The input: we get 32x32 pixels, each with 3 colors
    # (rgb). StandardScalar wants the dimensions flattened, so now
    # this gets the input directly.
    model.add(keras.layers.Flatten(input_shape=[3072]))

    if (dropout_rate > 0):
        # Apply dropout and then normalize
        keras.layers.Dropout(rate=dropout_rate)

    if (normalization or dropout_rate > 0):
        # Batch normalization after the input output.
        model.add(keras.layers.BatchNormalization())

    # Then the hidden layers, fully connected (100 by default)
    for i in range(20):
        model.add(keras.layers.Dense(
            n_classes,
            kernel_initializer="lecun_normal",
        ))
        if (dropout_rate > 0):
            keras.layers.Dropout(rate=dropout_rate)

        # If dropout rate is specified, then perform normalization
        if (normalization or dropout_rate > 0):
            model.add(keras.layers.BatchNormalization())
            model.add(keras.layers.Activation("selu"))

    # Now add the output layer: 10 classes in CIFAR10, so 10 outputs.
    model.add(keras.layers.Dense(10, activation="softmax"))

    # print(model.summary())
    # Compile model
    nadam = keras.optimizers.Nadam(
        learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-07)

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=nadam,
        metrics=["accuracy"]
    )
    return model

def standard_scale(X_train, X_valid, testX):
    """Performs standard scaling on the data, which requires flattening
    the input so it is a single feature vector of 3072 elements,
    rather than a row containing other arrays.

    Call with:
    X_train_reshape, X_valid_reshape, testX_reshape = standard_scale(X_train, X_valid, testX)
    """
    scaler = StandardScaler()
    X_train_reshape = X_train.reshape(40000, 3072)
    X_valid_reshape = X_valid.reshape(10000, 3072)
    testX_reshape = testX.reshape(10000, 3072)

    X_train_ss = scaler.fit_transform(X_train_reshape)
    # Always use the same scaler for the X_validation and X_test!
    X_valid_ss = scaler.transform(X_valid_reshape)
    testX_ss = scaler.transform(testX_reshape)

    return X_train_ss, X_valid_ss, testX_ss




# Get how well a model works
def model_quality(model, testX_reshape, testy):
    """ Calculate the quality of a flattened model

    Call with:
    model_quality(scaled, testX_reshape, testy)
    """

    # Calculate probabilities of each class
    y_pred = mm_drop.predict(testX_reshape)

    # Find the highest probability class
    prediction = np.argmax(y_pred_nomc, axis=1)

    # Now compute the accuracy
    return accuracy_score(prediction, testy)


# Monte-Carlo stacking of a model.
def stacked_model(model, testX_reshape, testy, iterations=100):
    """Calculate the predictions for a monte carlo version of the mode

    Call with:
    stacked_model(scaled, testX_reshape, testy, iterations=30)
    """

    # The probability of each class. training=True gets you a new
    # model every time due to dropout
    y_probas = np.stack([model(testX_reshape, training=True)
                         for sample in range(iterations)])

    # Find the mean probabilities for each class
    y_proba = y_probas.mean(axis=0)

    print_debug = False
    if (print_debug):
        np.round(y_proba[1], 2)

    # Find the highest probability class
    prediction = np.argmax(y_proba, axis=1)

    return accuracy_score(prediction, testy)




class OneCycleScheduler(keras.callbacks.Callback):
    """ My own 1-cycle scheduler """
    def __init__(self, iterations, max_rate, start_rate=None,
                 last_iterations=None, last_rate=None):
        self.iterations = iterations
        self.max_rate = max_rate
        self.start_rate = start_rate or max_rate / 10
        self.last_iterations = last_iterations or iterations // 10 + 1
        self.half_iteration = (iterations - self.last_iterations) // 2
        self.last_rate = last_rate or self.start_rate / 1000
        self.iteration = 0

    def _interpolate(self, iter1, iter2, rate1, rate2):
        return ((rate2 - rate1) * (self.iteration - iter1)
                / (iter2 - iter1) + rate1)

    def on_batch_begin(self, batch, logs):
        if self.iteration < self.half_iteration:
            rate = self._interpolate(0, self.half_iteration, self.start_rate, self.max_rate)
        elif self.iteration < 2 * self.half_iteration:
            rate = self._interpolate(self.half_iteration, 2 * self.half_iteration,
                                     self.max_rate, self.start_rate)
        else:
            rate = self._interpolate(2 * self.half_iteration, self.iterations,
                                     self.start_rate, self.last_rate)
            rate = max(rate, self.last_rate)
        self.iteration += 1
        keras.backend.set_value(self.model.optimizer.lr, rate)


# Create and test a model that uses the 1-cycle schedule
def create_one_cycle(X_train_reshape, X_valid_reshape, testX_reshape,
                     y_train, y_valid, testy):
    """
    Call with:
    create_one_cycle(X_train_reshape, X_valid_reshape, testX_reshape, y_train, y_valid, testy)
    """
    n_epochs = 100
    batch_size = 32
    onecycle = OneCycleScheduler(len(X_train_reshape) // batch_size * n_epochs, max_rate=0.05)
    debug=False

    # Create the model here, with batch normalization
    model = flattened_model(100, normalization=True, dropout_rate=0.2)
    if (debug):
        print ("Model built: ", model)

    # callbacks=[onecycle] call my one-cycle scheduler.
    history = model.fit(X_train_reshape, y_train, epochs=n_epochs,
                        verbose=0, batch_size=batch_size,
                        validation_data=(X_valid_reshape, y_valid),
                        callbacks=[onecycle])

    return history



