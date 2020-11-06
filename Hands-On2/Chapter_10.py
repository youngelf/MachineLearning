#!/usr/bin/env python
# coding: utf-8

## Chapter 10: Multi-Layer Perceptrons with Keras


# Common imports

import matplotlib.cm as cm
from matplotlib.image import imread
import matplotlib.pyplot as plt

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

import sys

print("TF version ", tf.__version__)
print("Keras version ", keras.__version__)


def load_fashion_mnist():
    """Load the fashion MNIST data
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



# -------------- converted till here --------------------
print ("Stopping early, still work left to do")
sys.exit(0)


class_names[y_train[0]]


# Let's look at a single image. By now I am good enough to print this stuff out.

# In[65]:


plt.imshow(1 - X_train[2])


# And let's create a Tensorflow network with Keras.

# In[66]:


# model = keras.models.Sequential()
# model.add(keras.layers.Flatten(input_shape=[28, 28]))
# model.add(keras.layers.Dense(300, activation="relu"))
# model.add(keras.layers.Dense(100, activation="relu"))
# model.add(keras.layers.Dense(10, activation="softmax"))

# This is another way of creating it:
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(300, activation="relu"),
    keras.layers.Dense(100, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])


# In[67]:


model.summary()


# In[19]:


model.layers


# In[21]:


model.compile(loss="sparse_categorical_crossentropy",
             optimizer="sgd",
             metrics=["accuracy"])

history = model.fit(X_train, y_train, epochs=30,
                   validation_data=(X_valid, y_valid))


# In[68]:


y_train.shape


# In[69]:


X_train.shape


# In[14]:


import pandas as pd

pd.DataFrame(history.history).plot(figsize=(8,5))
plt.grid(True)
plt.gca().set_ylim(0,1) # Y axis set to [0,1]
plt.show()


# This looks good, and convergence has been reached. Accuracy is close to validation accuracy.

# In[30]:


# The same model as earlier, but with a different learning rate.
model_lr = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(300, activation="relu"),
    keras.layers.Dense(100, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])

# Got a learning rate example from here: https://keras.io/api/optimizers/
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-2,
    decay_steps=10000,
    decay_rate=0.9)

model_lr.compile(loss="sparse_categorical_crossentropy",
                 optimizer=keras.optimizers.SGD(learning_rate=lr_schedule),
                 metrics=["accuracy"])
history_lr = model_lr.fit(X_train, y_train, epochs=30,
                   validation_data=(X_valid, y_valid))


# In[35]:


import pandas as pd

pd.DataFrame(history_lr.history).plot(figsize=(8,5))
plt.grid(True)
plt.gca().set_ylim(0,1) # Y axis set to [0,1]
plt.show()


# # Batch processing, for later
# 
# For later, run with many more epochs and see if the performance improves considerably

# In[31]:


# The same model as earlier, but with many more epochs than earlier.
model2 = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(300, activation="relu"),
    keras.layers.Dense(100, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])
model2.compile(loss="sparse_categorical_crossentropy",
             optimizer="sgd",
             metrics=["accuracy"])
history2 = model2.fit(X_train, y_train, epochs=300,
                   validation_data=(X_valid, y_valid))


# In[36]:


import pandas as pd

pd.DataFrame(history2.history).plot(figsize=(8,5))
plt.grid(True)
plt.gca().set_ylim(0,1) # Y axis set to [0,1]
plt.show()


# This looks overfitted. The accuracy is 100% but the validation accuracy is 91%.

# In[32]:


# A deeper model than earlier, but with as many epochs as the first.
model3 = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(300, activation="relu"),
    keras.layers.Dense(300, activation="relu"),
    keras.layers.Dense(100, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])
model3.compile(loss="sparse_categorical_crossentropy",
             optimizer="sgd",
             metrics=["accuracy"])
history3 = model3.fit(X_train, y_train, epochs=30,
                   validation_data=(X_valid, y_valid))


# In[38]:


import pandas as pd

pd.DataFrame(history3.history).plot(figsize=(8,5))
plt.grid(True)
plt.gca().set_ylim(0,1) # Y axis set to [0,1]
plt.show()


# This looks good, and convergence has been reached. Accuracy is close to validation accuracy.

# In[33]:


# A deeper model than earlier, but constraining the input layer and expanding again
model4 = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(300, activation="relu"),
    keras.layers.Dense(50, activation="relu"),
    keras.layers.Dense(100, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])
model4.compile(loss="sparse_categorical_crossentropy",
             optimizer="sgd",
             metrics=["accuracy"])
history4 = model4.fit(X_train, y_train, epochs=30,
                   validation_data=(X_valid, y_valid))


# In[37]:


import pandas as pd

pd.DataFrame(history4.history).plot(figsize=(8,5))
plt.grid(True)
plt.gca().set_ylim(0,1) # Y axis set to [0,1]
plt.show()


# This looks good, and convergence has been reached. Accuracy is close to validation accuracy.

# In[34]:


model4.summary()


# In[ ]:


# The same model as earlier, but with many more epochs than earlier.
model2 = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(300, activation="relu"),
    keras.layers.Dense(100, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])
model2.compile(loss="sparse_categorical_crossentropy",
             optimizer="sgd",
             metrics=["accuracy"])
history2 = model2.fit(X_train, y_train, epochs=300,
                   validation_data=(X_valid, y_valid))


# In[40]:


import pandas as pd

pd.DataFrame(history2.history).plot(figsize=(8,5))
plt.grid(True)
plt.gca().set_ylim(0,1) # Y axis set to [0,1]
plt.show()


# This looks overfitted. The accuracy is 100% but the validation accuracy is 91%.
# 
# Let's make a very deep model and see if that is better than model4, from which this is copied.

# In[41]:


# A deeper model than earlier, but constraining the input layer and expanding again
model4_deep = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(300, activation="relu"),
    keras.layers.Dense(100, activation="relu"),
    keras.layers.Dense(100, activation="relu"),
    keras.layers.Dense(100, activation="relu"),
    keras.layers.Dense(100, activation="relu"),
    keras.layers.Dense(100, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])
model4_deep.compile(loss="sparse_categorical_crossentropy",
             optimizer="sgd",
             metrics=["accuracy"])
history4_deep = model4_deep.fit(X_train, y_train, epochs=30,
                   validation_data=(X_valid, y_valid))


# In[42]:


import pandas as pd

pd.DataFrame(history4_deep.history).plot(figsize=(8,5))
plt.grid(True)
plt.gca().set_ylim(0,1) # Y axis set to [0,1]
plt.show()


# So adding extra layers didn't do much good. Let's try removing layers to see what we get.

# In[45]:


# A deeper model than earlier, but constraining the input layer and expanding again
model4_shallow = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(300, activation="relu"),
    keras.layers.Dense(100, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])
model4_shallow.compile(loss="sparse_categorical_crossentropy",
             optimizer="sgd",
             metrics=["accuracy"])
history4_shallow = model4_shallow.fit(X_train, y_train, epochs=30,
                   validation_data=(X_valid, y_valid))


# In[46]:


import pandas as pd

pd.DataFrame(history4_shallow.history).plot(figsize=(8,5))
plt.grid(True)
plt.gca().set_ylim(0,1) # Y axis set to [0,1]
plt.show()


# In[64]:


import pydot
keras.utils.plot_model(model4_shallow, to_file='model.png',
                       show_shapes=False, show_layer_names=True,
                       rankdir='TB', expand_nested=False, dpi=96)


# This is how you save a model to disk for reading later. This avoids the incredibly costly model training process.

# In[65]:


model4.save('saved_models/fashion_model4')


# This is how you load a previously-saved model from disk.

# In[68]:


new_model4 = keras.models.load_model('saved_models/fashion_model4')


# Let's try converting these to Tensorflow Lite. The book does not cover it, but could we use this model on an edge TPU device?

# In[47]:


converter = tf.lite.TFLiteConverter.from_keras_model(model4_shallow)
tflite_model = converter.convert()


# In[48]:


tflite_model


# Ok, that was easy, but now I need an edge TPU board to load that model and make sure it actually does something.
# 
# The way to load this is to use tflite, but I need to try that out on a board.
# 
# # Regression MLP using the Sequential API
# 
# You can use NNs for regression as well. The output is ordinal, and trained on ordinal data.

# In[2]:


from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

housing = fetch_california_housing()

X_train_full, X_test, y_train_full, y_test = train_test_split(
                                                housing.data, housing.target)


X_train, X_valid, y_train, y_valid = train_test_split(
                                                X_train_full, y_train_full)

scaler = StandardScaler()


# In[3]:


X_train = scaler.fit_transform(X_train)
# Always use the same scaler for the X_validation and X_test!
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)


# In[4]:


model = keras.models.Sequential([
    keras.layers.Dense(30, activation="relu", input_shape=X_train.shape[1:]),
    keras.layers.Dense(1)
])

model.compile(loss="mean_squared_error", optimizer="sgd")
history = model.fit(X_train, y_train, epochs=20,
                   validation_data = (X_valid, y_valid))

mse_test = model.evaluate(X_test, y_test)


# # Fine-Tuning Neural Network Hyperparameters
# 
# Seems like it is difficult to tell how many layers, how many neurons, and the learning rate, so you use GridSearch on it.  Here's how for the previous example
# 

# In[5]:


def build_model(n_hidden=1, n_neurons=30, learning_rate=3e-3, input_shape=[8]):
    "Create a model with paramters specified"
    model = keras.models.Sequential()
    model.add(keras.layers.InputLayer(input_shape = input_shape))
    for layer in range(n_hidden):
        model.add(keras.layers.Dense(n_neurons, activation="selu"))
    model.add(keras.layers.Dense(1))
    optimizer = keras.optimizers.SGD(lr=learning_rate)
    model.compile(loss="mse", optimizer=optimizer)
    return model

keras_reg = keras.wrappers.scikit_learn.KerasRegressor(build_model)


# In[22]:


keras_reg = keras.wrappers.scikit_learn.KerasRegressor(build_model)


# You can train a model using keras_reg as a model.

# In[23]:


keras_reg.fit(X_train, y_train, epochs=100,
             validation_data=(X_valid, y_valid),
              callbacks = [keras.callbacks.EarlyStopping(patience=10)])

mse_test = keras_reg.score(X_test, y_test)
y_pred = keras_reg.predict(X_new)


# Or, you can train a very computationally intensive Randomized or GridSearch here. I don't fully understand why Randomized Search is better here, but let's listen to the book and try it anyway

# In[15]:


# Custom error handler for the entire notebook so stack traces are not lost
from IPython.core.ultratb import AutoFormattedTB

# initialize the formatter for making the tracebacks into strings
itb = AutoFormattedTB(mode = 'Plain', tb_offset = 1)

# Define a global with the stack trace that we can append to in the handler.
viki_stack_trace = ''

# this function will be called on exceptions in any cell
def custom_exc(shell, etype, evalue, tb, tb_offset=None):
    global viki_stack_trace

    # still show the error within the notebook, don't just swallow it
    shell.showtraceback((etype, evalue, tb), tb_offset=tb_offset)

    # grab the traceback and make it into a list of strings
    stb = itb.structured_traceback(etype, evalue, tb)
    sstb = itb.stb2text(stb)

    print (sstb) # <--- this is the variable with the traceback string
    viki_stack_trace = viki_stack_trace + sstb

# this registers a custom exception handler for the whole current notebook
get_ipython().set_custom_exc((Exception,), custom_exc)


# In[ ]:


from scipy.stats import reciprocal
from sklearn.model_selection import RandomizedSearchCV
from datetime import datetime

param_distribs = {
    "n_hidden": [0, 1, 2, 3, 4, 5, 6, 7, 8],
    "n_neurons": np.arange(1, 200),  
    "learning_rate": reciprocal(3e-4, 3e-2),
}

# This is going to break.
viki_stack_trace = ''

start_time = datetime.now()
# The best estimator is only available if we 'refit=True'
rnd_cv = RandomizedSearchCV(keras_reg, param_distribs,
                            n_iter=10, cv=3, refit=True)

# Frustration upon frustration!
# Read cell below
# sklearn and Tensorflow/Keras don't play well with each other.
# You can get a Keras estimator, but cannot clone it. This does not allow
# RandomizedSearchCV to clone the best estimator and preserve it.
# And so we cannot specify refit=True.
# This has got to be one of the more frustrating parts of this 'software stack'
# Scikit-Learn makes specific assumptions, has expectations around clone.
# Tensorflow, and specifically Keras on Tensorflow doesn't work with the clone interface
# because it doesn't support copying its parameters.
#
# This whole stack is a ball of glue. When it works, you should celebrate because it
# can break at any moment.
# 
# Reference:
# https://github.com/keras-team/keras/issues/13586
# https://github.com/keras-team/keras/pull/13598


# verbose=0 removes all the noisy output from training.
rnd_cv.fit(X_train, y_train, epochs=200,
           validation_data=(X_valid, y_valid),
           callbacks=[keras.callbacks.EarlyStopping(patience=10)],
           verbose=0)

end_time = datetime.now()


# The failure disussed above
# 
# sklearn and Tensorflow/Keras don't play well with each other.
# You can get a Keras estimator, but cannot clone it. This does not allow
# RandomizedSearchCV to clone the best estimator and preserve it.
# And so we cannot specify refit=True.
# This has got to be one of the more frustrating parts of this 'software stack'
# Scikit-Learn makes specific assumptions, has expectations around clone.
# Tensorflow, and specifically Keras on Tensorflow doesn't work with the clone interface
# because it doesn't support copying its parameters.
# This whole stack is a ball of glue. When it works, you should celebrate because it
# can break at any moment.
# 
# Reference:
# [Issue on Keras](https://github.com/keras-team/keras/issues/13586) and [pull request to Keras to fix this discusing the issue](https://github.com/keras-team/keras/pull/13598) and another [pull request to Tensorflow to fix this issue](https://github.com/tensorflow/tensorflow/pull/41341)
# 
# A full illustration of the problem below.

# In[43]:


from sklearn.base import clone

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
 
estimator = keras.wrappers.scikit_learn.KerasClassifier(
    build_fn=create_keras_classifier_model, n_classes=2, class_weight={0: 1, 1:3})

viki_stack_trace = ''
clone(estimator)


# And this is the saved error from earlier.

# In[39]:


# keras_clone_error = viki_stack_trace
print (keras_clone_error)


# In[30]:


viki_stack_trace = ''
keras_clone_error = viki_stack_trace
print (keras_clone_error)
print(viki_stack_trace)


# In[38]:


keras_clone_error = viki_stack_trace
print (keras_clone_error)


# After the random search, you can get the best params, and the model, save it and use it for predictions.

# In[36]:


# The best paramters after search
# saved_best_params = rnd_cv.best_params_
# saved_best_score = rnd_cv.best_score_
#
print(saved_best_params)
print(saved_best_score)

# Time taken
end_time - start_time


# In[92]:


# best_model only exists when 'refit=True' is specified to RandomizedSearchCV, I think.
best_model = rnd_cv.best_estimator_


# In[93]:


dir(rnd_cv)


# In[94]:


rnd_cv.estimator


# In[ ]:





# Longer run for later

# In[ ]:


from scipy.stats import reciprocal
from sklearn.model_selection import RandomizedSearchCV
from datetime import datetime

param_distribs = {
    "n_hidden": (0, 1, 2, 3, 4, 5, 6, 7, 8),
    "n_neurons": np.arange(1, 200),  
    "learning_rate": reciprocal(3e-4, 3e-2),
}

start_time = datetime.now()
# The best estimator is only available if we 'refit=True'
rnd_cv = RandomizedSearchCV(keras_reg, param_distribs,
                            n_iter=500, cv=3, refit=False)

# verbose=0 removes all the noisy output from training.
rnd_cv.fit(X_train, y_train, epochs=200,
           validation_data=(X_valid, y_valid),
           callbacks=[keras.callbacks.EarlyStopping(patience=10)],
           verbose=0)

end_time = datetime.now()


# In[ ]:


end_time - start_time


# In[53]:


rnd_cv.best_score_


# In[54]:


rnd_cv.best_params_


# That finally worked, after close to two days of computation!

# # Exercises.
# 
# 1. Playing with [Tensorflow playground](https://playground.tensorflow.org/#activation=tanh&batchSize=10&dataset=circle&regDataset=reg-plane&learningRate=0.03&regularizationRate=0&noise=0&networkShape=4,2&seed=0.99375&showTestData=false&discretize=false&percTrainData=50&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false&hideText=false)
#    Ist gud. Limited, but good. I wish that kind of visualization was available for each layer on general data. As I remember, a friend was working on it. I should ask KD if he was still doing this.
#    
#    
# Observations:
# * Depth without hidden nodes is not helpful. 5 hidden layers, with 2 neurons per layer odn't converge with $x_1$ and $x_2$ as inputs.
# * Breadth without depth is actually ok. Four hidden neurons in just two layers still converges the 2d-donut data.
# * In the donut data, adding kernel functions $x_1^2$ and $x_2^2$ allows convergence. So Statistical fundamentals still matter! If you choose your kernel functions well, you can train simpler nets (shallow and narrow) and still et great performance.
# * Large amount of noise makes batch-size selection tricky. Too small a batch size, and your output doesn't generalize well.
# * Fun playground, and perhaps the first order of business on a problem should be to get a solid understanding of the properties of the data.
# 
# 
# 2. Draw A -> X, B -> NOT -> X.  X -> NOT -> Y, B -> Y. Then X two-connection-> OUTPUT, Y two-connection->OUTPUT. Then OUTPUT is A XOR B
# 
# 3. Logistic regression is inspectable. You can see what the intercept (bias) and scale terms mean. The scaling terms give some estimate of value of an attribute. A perceptron might be able to get the job done, though the complexity is not inspectable or understandable.
# 
# 4. Logistic activation allowed the training of models using back propagation. Previously, folks were using a step function which is not diffentiable everywhere. Logistic activation is differentiable everywhere and allows the errors to propagate backward to modify the weights of the node connections.
# 
# 5. Activation functions: Logistic, Rectified linear units, Sigmoid function. Step function.
# 
# 6.
# Answers below:
# 
# * Input matrix $X$ has shape $(m, 10)$ where $m$ is the number of observations.
# * Hidden layer weight vector $W_h$ has shape (10, 50) and bias vector $b_h$ has shape (1, 50)
# * Output layer weight vector $W_o$ has shape (50, 3) and bias vector $b_o$ has shape (1, 3)
# * Output matrix $Y$ has shape $(m,3)$ where $m$ is the number of observations.
# 
# Hidden layer matrix $L$ has shape $(m,50)$:
# $$L(i, k) = \sum_{j=1}^{10} \phi_1(X(i,j) \times W_h (j,k) + b_h(k))$$
# Then, the output Y is given by:
# $$Y(i, j) = \sum_{k=1}^{50} \phi_2(L(i,k) \times W_o (k,j) + b_o(j))$$
# 
# This should be the same as:
# $$ Y = \phi_2(\phi_1(X . W_h + b_h) . W_o + b_o)$$
# 
# 7. You need two neurons in the output layer for ham/spam. They will provide probabilities of a single observation being ham or spam. Pick the output with higher probability to classify.
#  For MNIST digits, you need 10 values: one for each digit. Again, you classify by picking the highest probability output.
# 
# 8. Backpropagation is taking the errors from a higher layer (or the output layer) and computing the gradients of the specific node, and using gradient descent to update the node weights to reduce the errors at this layer. You do this going back from output (calculating errors from the observed training data), to the highest layer, then the next lower layers, going back to the lowest layer of the network.
# 
# Reverse mode autodiff is the specific technique used to calculate gradients of the nodes in the network. It is used by back-propagation to translate errors into magnitude of node weight changes.
# 
# 9. Hyperparameters you can tune: Activation function: Linear/selu/relu/sigmoid, number of layers, number of neurons in each layer, Amount of training/test data, size of minibatch, number of epochs, stopping criteria, learning rate. You could try using RandomizedSearchCV to get through this blistering complexity. Scientific Dart-throwing, basically.
# 
# 10. Still waiting for the previous run to finish on housing dataset. Will run on MNIST once this notebook's kernel is freed up.

# In[4]:


(X_train_full, y_train_full), (X_test, y_test) = keras.datasets.mnist.load_data()


# In[5]:


X_train_full.shape


# In[6]:


y_train_full.shape


# In[8]:


X_train_full[0]


# Split into training and validation. Divide by 255.0 because the input data is in range \[0, 255\]. We want it in the range \(0, 1\)

# In[9]:


X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:] / 255.0
y_valid, y_train = y_train_full[:5000]        , y_train_full[5000:]
X_test = X_test / 255.0


# In[10]:


# model = keras.models.Sequential()
# model.add(keras.layers.Flatten(input_shape=[28, 28]))
# model.add(keras.layers.Dense(300, activation="relu"))
# model.add(keras.layers.Dense(100, activation="relu"))
# model.add(keras.layers.Dense(10, activation="softmax"))

# This is another way of creating it:
mnist_model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(300, activation="relu"),
    keras.layers.Dense(100, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])


# In[84]:


mnist_model.compile(loss="sparse_categorical_crossentropy",
             optimizer="sgd",
             metrics=["accuracy"])

history = mnist_model.fit(X_train, y_train, epochs=30,
                   validation_data=(X_valid, y_valid))


# In[85]:


import pandas as pd

pd.DataFrame(history.history).plot(figsize=(8,5))
plt.grid(True)
plt.gca().set_ylim(0,1) # Y axis set to [0,1]
plt.show()


# Validation accuracy is 97.96%, close enough to the 98% the book was yearning for.  Let's check this on the test set.

# In[86]:


y_pred = mnist_model.predict(X_test)


# In[89]:


mnist_model.evaluate(X_test, y_test)


# Test loss is 0.07, and test accuracy is 97.79% Can experiment with a few other combinations to find the ideal model.
# 
# Using SELU rather than RELU

# In[11]:


# model = keras.models.Sequential()
# model.add(keras.layers.Flatten(input_shape=[28, 28]))
# model.add(keras.layers.Dense(300, activation="relu"))
# model.add(keras.layers.Dense(100, activation="relu"))
# model.add(keras.layers.Dense(10, activation="softmax"))

# This is another way of creating it:
mnist_model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(300, activation="selu"),
    keras.layers.Dense(100, activation="selu"),
    keras.layers.Dense(10, activation="softmax")
])


# In[12]:


mnist_model.compile(loss="sparse_categorical_crossentropy",
             optimizer="sgd",
             metrics=["accuracy"])

history = mnist_model.fit(X_train, y_train, epochs=30,
                          validation_data=(X_valid, y_valid),
                          verbose=0)


# In[13]:


mnist_model.evaluate(X_test, y_test)


# 97.48\% accuracy, not bad.

# In[14]:


import pandas as pd

pd.DataFrame(history.history).plot(figsize=(8,5))
plt.grid(True)
plt.gca().set_ylim(0,1) # Y axis set to [0,1]
plt.show()


# Hmm, that was slightly worse. The graphs indicate that validation accuracy was still improving, and perhaps we need to iterate for longer.

# In[15]:


# model = keras.models.Sequential()
# model.add(keras.layers.Flatten(input_shape=[28, 28]))
# model.add(keras.layers.Dense(300, activation="relu"))
# model.add(keras.layers.Dense(100, activation="relu"))
# model.add(keras.layers.Dense(10, activation="softmax"))

# This is another way of creating it:
mnist_model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(300, activation="selu"),
    keras.layers.Dense(300, activation="selu"),
    keras.layers.Dense(100, activation="selu"),
    keras.layers.Dense(10, activation="softmax")
])


# In[16]:


mnist_model.compile(loss="sparse_categorical_crossentropy",
             optimizer="sgd",
             metrics=["accuracy"])

history = mnist_model.fit(X_train, y_train, epochs=100,
                          validation_data=(X_valid, y_valid),
                          verbose=0)


# In[17]:


mnist_model.evaluate(X_test, y_test)


# 97.97\%. I was hoping for higher, but this is fine too.

# # Done with all exercises

# In[ ]:




