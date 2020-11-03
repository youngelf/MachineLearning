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


