# Chapter 10: Deep neural models

from datetime import datetime
import matplotlib.cm as cm
import matplotlib

from matplotlib.image import imread

import numpy as np

import pandas as pd
import pydot

from scipy.stats import reciprocal

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
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler

from scipy.stats import reciprocal
from sklearn.model_selection import RandomizedSearchCV
from datetime import datetime


import sys

import tensorflow as tf
from tensorflow import keras


# Verify that the run worked and we have the imports working.
print("TF version ", tf.__version__)
print("Keras version ", keras.__version__)


def load_digits_mnist():
    """Load the (digits) MNIST data. Should only need to be called once.

    Call with:
    X_train, X_valid, X_test, y_train, y_valid, y_test, class_names = load_digits_mnist()
    """
    (X_train_full, y_train_full), (X_test, y_test) = keras.datasets.mnist.load_data()

    debug = False
    if (debug):
        print (X_train_full.shape)
        print (X_train_full.dtype)

    X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:] / 255.0
    y_valid, y_train = y_train_full[:5000]        , y_train_full[5000:]
    X_test = X_test / 255.0



def load_fashion_mnist():
    """Load the fashion MNIST data. Should only need to be called once.

    Call with:
    X_train, X_valid, X_test, y_train, y_valid, y_test, class_names = load_fashion_mnist()
    """

    fashion_mnist = keras.datasets.fashion_mnist

    (X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()

    debug = False
    if (debug):
        print (X_train_full.shape)
        print (X_train_full.dtype)

    # Let's split this into a validation set. First 5000 are kept aside for validation.
    #
    # Pixel densities in the attributes are 0-255, so scaling them down to [0,1].
    # Notice how we specify 255.0 otherwise integer division will lead to either 0 or 1.

    X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:] / 255.0
    y_valid, y_train = y_train_full[:5000]        , y_train_full[5000:]
    X_test = X_test / 255.0

    # These class-names are specified on the [Keras
    # website](https://www.tensorflow.org/tutorials/keras/classification)
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    return X_train, X_valid, X_test, y_train, y_valid, y_test, class_names



def print_debug(X_train, y_train, class_names):
    """Prints some debugging information about the dataset

    Call with:
    print_debug(X_train, y_train, class_names)
    """

    print ("Class name of the first value is: ", class_names[y_train[0]])
    # Let's look at a single image. By now I am good enough to print this stuff out.
    plt.imshow(1 - X_train[0])


def create_simplest_model(layers=[300, 100], optimizer="sgd"):
    """Create a dense model with 2 relu activation layers by default and softmax
output

    optimizer(optional): Provide an optimizer that will be used when
      compiling the model.

    Call with:
    simplest = create_simplest_model()

    You can make a deeper model specifying the hidden layers like this
    simplest = create_simplest_model([300, 300, 100])

    """

    # This is the other way of creating the default layer. It just
    # doesn't work when we want it to be dynamic

    # model = keras.models.Sequential([
    #   keras.layers.Flatten(input_shape=[28, 28]),
    #   keras.layers.Dense(300, activation="relu"),
    #   keras.layers.Dense(100, activation="relu"),
    #   keras.layers.Dense(10, activation="softmax")
    # ])

    # We use the imperitive method since we will pass an array of values
    model = keras.models.Sequential()
    model.add(keras.layers.Flatten(input_shape=[28, 28]))

    # Add all the layers provided as arguments. By default we will add
    # two layers, one with 300 hidden layers, and the second with 100
    # hidden layers.
    for i in layers:
        model.add(keras.layers.Dense(i, activation="relu"))

    # Finally add the output layer
    model.add(keras.layers.Dense(10, activation="softmax"))

    print ("Model created: ", model.summary())
    print ("Model layers: ", model.layers)

    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer=optimizer,
                  metrics=["accuracy"])

    return model


def fit_model(model, X_train, y_train, X_valid, y_valid, epochs=30, verbose=0):
    """
    Call with:
    history = fit_model(model, X_train, y_train, X_valid, y_valid, epochs=30)
    """
    history = model.fit(X_train, y_train, epochs=epochs, verbose=verbose,
                   validation_data=(X_valid, y_valid))

    return history


def plot_training(history, name=None):
    """Plots the model training history

    history: History obtained when training models

    name(String): A human-readable string used when saving models. If
    this value is provided, then the .

    Call with:

    plot_training(history, name="simplest") # Save to file called "simplest"

    plot_training(history) # show on screen
    """
    fig = None

    if (name != None):
        # Use the Anti-Grain-Geometry (file-based) backend
        matplotlib.use('Agg')
        print ("Using file-based backend")
        import matplotlib.pyplot as plt
    else:
        # Use the Tk (X-based) backend
        matplotlib.use('TkAgg')
        print ("Using X-based backend")
        import matplotlib.pyplot as plt

    fig = plt.figure()
    pd.DataFrame(history.history).plot(figsize=(8,5))
    plt.grid(True)
    plt.gca().set_ylim(0,1) # Y axis set to [0,1]
    plt.ylabel("Accuracy or Validation error")
    plt.xlabel("Training epoch")

    if (name == None):
        plt.show()
    else:
        plt.savefig("images/" + name + ".png")
        plt.close(fig)

def save_model(model, name):
    """ Save the model, and a plot of the model's internals

    Call with:
    save_model(simplest, "simplest")
    """
    keras.utils.plot_model(model, to_file='images/model_' + name + '.png',
                           show_shapes=False, show_layer_names=True,
                           rankdir='TB', expand_nested=False, dpi=96)

    model.save('saved_models/' + name)


def load_model(name):
    """Loads a previously saved model

    Call with:
    model = load_model('simplest')

    """
    # This is how you load a previously-saved model from disk.
    model = keras.models.load_model('saved_models/' + name)
    return model


def save_tflite(model):
    """Convert a model to TFlite

    Call with:
    tflite_model = save_tflite(model)
    """
    # Let's try converting these to Tensorflow Lite. The book does not
    # cover it, but could we use this model on an edge TPU device?

    # Many Edge TPU devices exist, but none of them work. Sigh
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    return tflite_model


def load_housing_data():
    """ Load California housing data, scaled

    Call with:
    X_train, y_train, X_valid, y_valid, testX, testy, scaler = load_housing_data()
    """

    housing = fetch_california_housing()
    X_train_full, testX, y_train_full, testy = train_test_split(
        housing.data, housing.target)

    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_full, y_train_full)

    # Scale the data before returning
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    # Always use the same scaler for the X_validation and X_test!
    X_valid = scaler.transform(X_valid)
    testX = scaler.transform(testX)

    return X_train, y_train, X_valid, y_valid, testX, testy, scaler


def create_housing_model(X_train, y_train, X_valid, y_valid, optimizer="sgd"):
    """ Creates a housing model

    Call with:
    model = create_housing_model(X_train, y_train, X_valid, y_valid, optimizer="sgd")
    """
    model = keras.models.Sequential([
        keras.layers.Dense(30, activation="relu", input_shape=X_train.shape[1:]),
        keras.layers.Dense(1)
    ])
    model.compile(loss="mean_squared_error", optimizer=optimizer)
    return model


def fit_housing(model, X_train, y_train, X_valid, y_valid, testX, testy, epochs=20):
    """ Fits the housing data, and also returns the training history and the mse error.
    Call with:
    history, mse_test = fit_housing(model, X_train, y_train, X_valid, y_valid, testX, testy, epochs=30)
    """
    history = model.fit(X_train, y_train, epochs=epochs,
                        validation_data = (X_valid, y_valid))
    mse_test = model.evaluate(testX, testy)
    return history, mse_test


def build_model(n_hidden=1, n_neurons=30, learning_rate=3e-3, input_shape=[8]):
    """Create a model with the specified parameters

    Call with:
    keras_reg = keras.wrappers.scikit_learn.KerasRegressor(build_model)
    keras_reg.fit(X_train, y_train, epochs=100,
             validation_data=(X_valid, y_valid),
              callbacks = [keras.callbacks.EarlyStopping(patience=10)])
    mse_test = keras_reg.score(X_test, y_test)
    y_pred = keras_reg.predict(X_new)

    """
    model = keras.models.Sequential()
    model.add(keras.layers.InputLayer(input_shape = input_shape))
    for layer in range(n_hidden):
        model.add(keras.layers.Dense(n_neurons, activation="selu"))
    model.add(keras.layers.Dense(1))
    optimizer = keras.optimizers.SGD(lr=learning_rate)
    model.compile(loss="mse", optimizer=optimizer)
    return model

def create_keras_classifier_model(n_classes):
    """Keras multinomial logistic regression creation model

    Args:
        n_classes(int): Number of classes to be classified

    Returns:
        Compiled keras model

    """
    # create model
    model = keras.models.Sequential()
    model.add(keras.layersDense(n_classes, activation="softmax"))
    # Compile model
    model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )
    return model


def run_all_10():
    """Run all Chapter 10 code.
    Call with:'
    run_all_10()

    Takes a few hours(!) to run since we are making many models
    """

    ignore = '''
    # Load the dataset and ensure it is fine
    X_train, X_valid, X_test, y_train, y_valid, y_test, class_names = load_fashion_mnist()

    # This needs X windows, let's ignore it
    # print_debug(X_train, y_train, class_names)


    # Now create all the datasets. This one is the simplest naive model
    simplest = create_simplest_model()
    history = fit_model(simplest, X_train, y_train, X_valid, y_valid, epochs=30)
    plot_training(history, "simplest-30")
    save_model(simplest, "simplest")

    # Run for longer. Same model as before, trained for longer.
    simplest = create_simplest_model()
    history = fit_model(simplest, X_train, y_train, X_valid, y_valid, epochs=300)
    # This should look overfitted. The accuracy is 100% but the validation accuracy is 91%.
    plot_training(history, "simplest-300")


    # Got a learning rate example from here: https://keras.io/api/optimizers/
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-2,
        decay_steps=10000,
        decay_rate=0.9)

    model_lr = create_simplest_model(optimizer=keras.optimizers.SGD(learning_rate=lr_schedule))
    history_lr = fit_model(model_lr, X_train, y_train, X_valid, y_valid, epochs=30)
    plot_training(history_lr, "lr-30")


    deeper = create_simplest_model([300, 300, 100])
    history_deeper = fit_model(deeper, X_train, y_train, X_valid, y_valid, epochs=30)
    plot_training(history_deeper, "deeper-30")
    save_model(deeper, 'fashion_deeper')

    constrained = create_simplest_model([300, 50, 100])
    history_constrained = fit_model(constrained, X_train, y_train, X_valid, y_valid, epochs=30)
    plot_training(history_deeper, "constrained-30")
    save_model(constrained, 'fashion_constrained')

    constrained = create_simplest_model([300, 50, 100])
    history_constrained = fit_model(constrained, X_train, y_train, X_valid, y_valid, epochs=300)
    plot_training(history_deeper, "constrained-300")


    superdeep = create_simplest_model([300, 100, 100, 100, 100, 100])
    history_superdeep = fit_model(superdeep, X_train, y_train, X_valid, y_valid, epochs=30)
    plot_training(history_deeper, "superdeep-30")
'''
    
    X_train, y_train, X_valid, y_valid, testX, testy, scaler = load_housing_data()
    housing = create_housing_model(X_train, y_train, X_valid, y_valid)
    history, mse_test = fit_housing(housing, X_train, y_train, X_valid, y_valid, testX, testy, epochs=30)


    keras_reg = keras.wrappers.scikit_learn.KerasRegressor(build_model)
    keras_reg.fit(X_train, y_train, epochs=100,
                  validation_data=(X_valid, y_valid),
                  callbacks = [keras.callbacks.EarlyStopping(patience=10)])
    mse_test = keras_reg.score(testX, testy)
    print ("Error of keras reg: ", mse_test)

    # Or, you can train a very computationally intensive Randomized or
    # GridSearch here. I don't fully understand why Randomized Search
    # is better here, but let's listen to the book and try it anyway
    param_distribs = {
        "n_hidden": [0, 1, 2, 3, 4, 5, 6, 7, 8],
        "n_neurons": np.arange(1, 200),
        "learning_rate": reciprocal(3e-4, 3e-2),
    }

    estimator = keras.wrappers.scikit_learn.KerasClassifier(
        build_fn=create_keras_classifier_model, n_classes=2, class_weight={0: 1, 1:3})

    # The clone(estimator) call fails. Detailed notes are in the
    # notebook, but the summary is that sklearn and TF/Keras don't
    # play well. So you can get an estimator you cannot clone. You can
    # run the full training, and get optimal parameters, but not the
    # saved estimator. Instead, you have to do a two-pass algorithm:
    # first getting the best params (refit=False) and then fitting
    # another estimator with those params you got earlier, and hope
    # you get a similar estimator: which you might not because model
    # training is not guaranteed to give you the same results on
    # subsequent runs.
    
    # clone(estimator)

    start_time = datetime.now()

    # The best estimator is only available if we 'refit=True', and we
    # cannot do that because it requires clone() to work. So we do
    # refit=False, and get the best params, and be satisfied with
    # that.
    rnd_cv = RandomizedSearchCV(keras_reg, param_distribs,
                                n_iter=500, cv=3, refit=False)

    # verbose=0 removes all the noisy output from training.
    rnd_cv.fit(X_train, y_train, epochs=200,
               validation_data=(X_valid, y_valid),
               callbacks=[keras.callbacks.EarlyStopping(patience=10)],
               verbose=0)

    end_time = datetime.now()
    print ("Total run time: ", end_time - start_time)

    print ("Best score: ", rnd_cv.best_score_)
    print ("Best params: ", rnd_cv.best_params_)


print ("Fully done! Thanks")

# -------------- All converted --------------------

