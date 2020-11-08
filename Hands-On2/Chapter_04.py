#!/usr/bin/env python
# coding: utf-8

# # Chapter 4: Training models and Mathematics
# The iris database

from sklearn import datasets
from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt
import numpy as np

def load_iris():
    """ Load the IRIS dataset.
    Call with:
    iris = load_iris()

    Does not separate data into training and test.
    """

    iris = datasets.load_iris()
    # Print all the keys
    list(iris.keys())

    # A description of the iris dataset  is stored in this magic variable.
    print (iris.DESCR)

    print ("Iris data shape = ", iris.data.shape)

    return iris

def create_model(iris):
    """ Create and fit a logistic regression model
    Call with:
    model_1d = create_model(iris)
    """
    # Create a logistic regression model, and fit it.
    log_reg = LogisticRegression()

    # Note that I am fitting the full dataset, I haven't set aside a test
    # set.  This is a bad idea

    # Features and categories to be classified
    X = (iris["data"][:,3:]) # Petal width
    y = (iris["target"] == 2).astype(np.int) # 1 if iris verginica, else 0
    log_reg.fit(X, y)

    return log_reg


def plot_1d_predictions(model, name="iris-space"):
    """Plot the space of all values, and corresponding predictions

    Call with:
    plot_1d_predictions(model_1d , name="iris-1d")
    """

    # Generate all the points in the 1-d space
    X_new = np.linspace(0, 3, 1000).reshape(-1, 1)
    y_proba = model.predict_proba(X_new)
    plt.plot(X_new, y_proba[:, 1], 'r-', label="Iris virginica")
    plt.plot(X_new, y_proba[:, 0], 'b--', label="Not Iris virginica")
    plt.legend()
    plt.grid()

    plt.savefig('images/' + name + '.png')


def create_2d_model(iris):
    """Fits a regression model with two variables

    Call with:
    create_2d_model(iris)

    """
    X_small = iris["data"][:,2:] # Petal length and width
    y_small = (iris["target"] == 2).astype(np.int) # 1 if iris verginica, else 0

    log_reg = LogisticRegression()
    log_reg.fit(X_small, y_small)

    return log_reg


def create_softmax_model(iris):
    """Fits a regression model with two variables

    Call with:
    create_softmax_model(iris)

    """

    # Logistic regression is capable of predicting multiple classes by
    # calculating a score for each class, and a corresponding
    # probability for each class.
    #
    # The cost function has a penalty for not predicting the target class.

    X_softmax = iris["data"][:, (2,3)] # petal length, petal width
    y_softmax = iris["target"]

    # The C parameter is the opposite of the alpha parameter for regularization.
    # Low C means more regularization, high C means less regularization.
    # TODO: find its range
    softmax_reg = LogisticRegression(multi_class="multinomial", solver="lbfgs", C=10)
    softmax_reg.fit(X_softmax, y_softmax)

    return softmax_reg


def run_all():
    """ Run the full set on this file.

    Call with:
    run_all()

    Full run takes a few minutes
    """

    iris = load_iris()
    model_1d = create_model(iris)
    plot_1d_predictions(model_1d, name="iris-1d")

    model_2d = create_2d_model(iris)

    softmax_model = create_softmax_model(iris)
