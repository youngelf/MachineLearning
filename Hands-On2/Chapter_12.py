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


# Create my Normalization Layer

class VikiNormalizedLayer(keras.layers.Layer):
    def __init__(self, activation=None, **kwargs):
        # The general initialization routine, parse the normal args
        # and remember the units.
        super().__init__(**kwargs)
        # self.units = units
        self.activation = keras.activations.get(activation)

    def call(self, inputs):
        # Perform layer normalization here
        mean, variance = tf.nn.moments(inputs, axes=-1, keepdims=True)
        std_dev = tf.math.sqrt(variance)

        # Eps is a small smoothing factor, selected to be everyones
        # favorite: 0.001 here.
        eps = 0.001
        # * here is element-wise multiplication that gets written as
        # tf.math.multiply(). That is different from tf.mult() which
        # is matrix multiplication.
        return (self.alpha * (inputs - mean) / (std_dev + eps)) + self.beta


    def build(self, batch_input_shape):
        # Define two trainable weights: alpha and beta, which are the
        # same shape as the previous out and float32.
        self.alpha = self.add_weight(name="alpha", shape=[batch_input_shape[-1]],
                                     initializer="ones")
        print ("shape = ", batch_input_shape[-1])
        self.beta = self.add_weight(name="beta", shape=[batch_input_shape[-1]],
                                    initializer="zeros")


# This is how to test out the normalization layer
data = tf.constant(np.arange(10).reshape(5, 2) * 10, dtype=tf.float32)
print(data)

# Create the usual Keras LayerNormalization
layer = tf.keras.layers.LayerNormalization(axis=1)

print ("Using LayerNormalization")
output = layer(data)
print(output)

# Create my version of LayerNormalization
layer = VikiNormalizedLayer()

print ("Using VikiNormalizedLayer")
output = layer(data)
print(output)
# Based on that, let's create a model using my Normalization layer


def create_keras_classifier_model(n_classes=100):
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
    for i in range(3):
        model.add(keras.layers.Dense(
            n_classes, 
            activation="elu",
            kernel_initializer=tf.keras.initializers.HeNormal(),
            kernel_regularizer=tf.keras.regularizers.l2(0.01),
        ))
        model.add(VikiNormalizedLayer())
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

# Clear the errors, in case we observe them in the long run.
viki_stack_trace = ''

# Let's load the data
def 
(X, y), (testX, testy) = keras.datasets.cifar10.load_data()

# Split into training and testing
X_train, X_valid = X[:40000] / 255.0, X[40000:] / 255.0
y = y.reshape(50000)
testy.reshape(10000)

y_train, y_valid = y[:40000]        , y[40000:]

print("Validation: ", X_valid.shape)
print("Training: ", X_train.shape)
print("Labels validation: ", y_valid.shape)
print("Labels training: ", y_train.shape)

print("Test: ", testX.shape)
print("Labels test: ", testy.shape)


from sklearn.base import clone


# Got to remember them. mm_bn is the model with Batch normalization
mm_bn = create_keras_classifier_model(100)
print ("Model built: ", mm_bn)

history_bn = mm_bn.fit(X_train, y_train, epochs=10, verbose=0,
                 batch_size=32,
                 validation_data=(X_valid, y_valid))

import pandas as pd
pd.DataFrame(history_bn.history).plot(figsize=(8,5))
plt.grid(True)
plt.gca().set_ylim(0,1) # Y axis set to [0,1]
plt.show()

# Save the plot of loss history.
plt.savefig('VikiNormTraining.png')
