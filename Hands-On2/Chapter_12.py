#!/usr/bin/env python
# coding: utf-8

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


# Creating a new loss function allows you to store the config, load
# from a config and apply ('call') the method.
# 
# Initializers, Regularizers, Constraings can be overwriten. A
# kernel_constraint allows you to overwrite the edge weights.
# 
# 

# Sanity check on core dependencies: things are working well if this
# prints out the version
print("TF version ", tf.__version__)
print("Keras version ", keras.__version__)

# "Loss" functions are used during training, and their gradient is
# what is optimized.
# 
# By contrast, "metrics" are used to evaluate a model, they can be
# anything arbitrary. They have no expectation of having nonzero
# values or existence of gradients.


# This is a custom loss function
class HuberLoss(keras.losses.Loss):
    "A custom loss function that will be used later. Just an example"

    def __init(self, threshold=1.0, **kwargs):
        self.threshold = threshold
        super().__init__(**kwargs)

    def call(self, y_true, y_pred):
        "Evaluate the loss at this stage"
        error = y_true - y_pred
        is_small_error = tf.abs(error) < self.threshold
        squared_loss = tf.square(error) / 2
        linear_loss = self.threshold * tf.abs(error) - self.threshold ** 2 / 2
        return tf.where(is_small_error, squared_loss, linear_loss)
    
    def get_config(self):
        """Called when model is saved to preserve existing config. This class
           will save its parent class' config too."""
        base_config = super().get_config()
        return {**base_config, "threshold": self.threshold}


# Here are other custom functions:
# 

def activation_softplus(z):
    "Used to return a probability of seeing this output"
    return tf.math.log(tf.exp(z) + 1.0)

def initializer_glorot(shape, dtype=tf.float32):
    "Used to initialize weights before training"
    stddev = tf.sqrt(2. / (shape[0] + shape[1]))
    return tf.random.normal(shape, stddev=stddev, dtype=dtype)

def regularizer_l1(weights):
    "Used to avoid over-fitting, and keep weights meaningful"
    return tf.reduce_sum(tf.abs(0.01 * weights))

def constraint_weights(weights):
    """Applied after the training to constrain the weights at the layer
       arbitrarily"""
    return tf.where(weights < 0., tf.zeros_like(weights), weights)


# The above methods can be used directly, but we can also create a class that inherits from

# keras.initializers.Initializer, keras.regularizers.Regularizer, and
# keras.constraints.Constraint appropriately.  The activation function
# usually has nothing to save, so if you want to have a parameter for
# the activation, you can create a new layer type.
# 
# Here's an example of extending just one of them, the Regularizer.
class VikiL1(keras.regularizers.Regularizer):
    def __init__(self, factor):
        "Create an L1 regularizer (with reg. factor provided)"
        self.factor = factor

    def __call__(self, weights):
        "Apply this regularizer with the weights at this layer"
        return tf.reduce_sum(tf.abs(self.factor * weights))
    
    def get_config(self):
        "Returns the configuration of this class for application later"
        # We don't look up the parent's config, because it has none.
        return {"factor": self.factor}
    


# A custom layer can be implemented that does add_weight() for all the
# values it needs to keep track of, and in the call() method, it
# provides the output from this layer. I don't quite understand how
# gradients are calculated at every layer. Perhaps the exercises make
# this clearer.


# Now you can %run Chapter_12.py in ipython3. This is so much better
# than the cumbersome loading in Jupyter all the time, and can be used
# elsewhere and elsewhen.


