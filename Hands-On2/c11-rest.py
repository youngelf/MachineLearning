
from sklearn.base import clone

def create_keras_classifier_model(n_classes=100):
    """Keras multinomial logistic regression creation model

    Args:
        n_classes(int): Number of classes to be classified

    Returns:
        Compiled keras model

    """
    # create model
    model = keras.models.Sequential()

    # The input: we get 32x32 pixels, each with 3 colors (rgb). StandardScalar wants the dimensions flattened,
    # so now this gets the input directly.
    model.add(keras.layers.Flatten(input_shape=[3072]))
    # Batch normalization after the input output.
    model.add(keras.layers.BatchNormalization())

    # Then the hidden layers, fully connected (100 by default)
    for i in range(20):
        model.add(keras.layers.Dense(
            n_classes,
            kernel_initializer="lecun_normal",
        ))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Activation("selu"))

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

# Got to remember them. mm_bn is the model with Batch normalization
mm_selu = create_keras_classifier_model(100)
print ("Model built: ", mm_selu)

# Now we must standard scale the data.
scaler = StandardScaler()
X_train_reshape = X_train.reshape(40000, 3072)
X_valid_reshape = X_valid.reshape(10000, 3072)
X_test_reshape = testX.reshape(10000, 3072)

X_train_ss = scaler.fit_transform(X_train_reshape)
# Always use the same scaler for the X_validation and X_test!
X_valid_ss = scaler.transform(X_valid_reshape)
X_test_ss = scaler.transform(X_test_reshape)

history_selu = mm_selu.fit(X_train_ss, y_train, epochs=100, verbose=0,
                 batch_size=32,
                 validation_data=(X_valid_ss, y_valid))


# In[14]:


import pandas as pd

pd.DataFrame(history_selu.history).plot(figsize=(8,5))
plt.grid(True)
plt.gca().set_ylim(0,1) # Y axis set to [0,1]
plt.show()


# Slightly better than before! Next up is
#
# e. Try regularization with the model using alpha dropout. Then, without retraining your model, see if you can achieve better accuracy using MC Dropout

# In[ ]:


from sklearn.base import clone

# Alpha dropout work.
def normalized_dropout(n_classes=100, dropout_rate=0.2):
    """Keras multinomial logistic regression creation model

    Args:
        n_classes(int): Number of classes to be classified

    Returns:
        Compiled keras model

    """
    # create model
    model = keras.models.Sequential()

    # The input: we get 32x32 pixels, each with 3 colors (rgb). StandardScalar wants the dimensions flattened,
    # so now this gets the input directly.
    model.add(keras.layers.Flatten(input_shape=[3072]))

    # Apply dropout and then normalize
    keras.layers.Dropout(rate=dropout_rate)
    # Batch normalization after the input output.
    model.add(keras.layers.BatchNormalization())

    # Then the hidden layers, fully connected (100 by default)
    for i in range(20):
        model.add(keras.layers.Dense(
            n_classes,
            kernel_initializer="lecun_normal",
        ))
        # Apply dropout and then normalize
        keras.layers.Dropout(rate=dropout_rate)
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Activation("selu"))

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

# Got to remember them. mm_bn is the model with Batch normalization
mm_drop = normalized_dropout(100)
print ("Model built: ", mm_drop)

# Now we must standard scale the data.
scaler = StandardScaler()
X_train_reshape = X_train.reshape(40000, 3072)
X_valid_reshape = X_valid.reshape(10000, 3072)
X_test_reshape = testX.reshape(10000, 3072)

X_train_ss = scaler.fit_transform(X_train_reshape)
# Always use the same scaler for the X_validation and X_test!
X_valid_ss = scaler.transform(X_valid_reshape)
X_test_ss = scaler.transform(X_test_reshape)

history_drop = mm_drop.fit(X_train_ss, y_train, epochs=100, verbose=0,
                 batch_size=32,
                 validation_data=(X_valid_ss, y_valid))


# In[17]:


import pandas as pd

pd.DataFrame(history_drop.history).plot(figsize=(8,5))
plt.grid(True)
plt.gca().set_ylim(0,1) # Y axis set to [0,1]
plt.show()


# In[28]:


from sklearn.metrics import accuracy_score

y_pred_nomc = mm_drop.predict(X_test_ss)


# In[30]:


y_test_nomc = np.argmax(y_pred_nomc, axis=1)


# In[31]:


accuracy_score(y_test_nomc, testy)


# In[19]:


y_probas = np.stack([mm_drop(X_test_ss, training=True)
                     for sameple in range(100)])

y_proba = y_probas.mean(axis=0)


# That was better, but only slightly, let's try MC dropout without training to see if it improves the performance

# In[21]:


np.round(y_proba[1], 2)


# In[23]:


y_pred = np.argmax(y_proba, axis=1)


# In[25]:


from sklearn.metrics import accuracy_score

accuracy_score(y_pred, testy)


# So there is an improvement of accuracy from 47.24% to 51.59%. Let's see if it improves further if we run it for 300 MC runs.

# In[32]:


y_probas = np.stack([mm_drop(X_test_ss, training=True)
                     for sameple in range(300)])

y_proba = y_probas.mean(axis=0)
y_pred = np.argmax(y_proba, axis=1)


# In[33]:


from sklearn.metrics import accuracy_score

accuracy_score(y_pred, testy)


# In[42]:


y_pred[1:10]


# 100 was as good as 300, it asymptotes out at some point, that point was at or before 100 MC runs. Good to know.

# f. Retrain your model using 1cycle scheduling and see if it improves speed and model accuracy.

# In[39]:


def linear_ramp_down(n0=0.0001, n1=0.00001, steps = 100.0):
    def decay_fn(epoch):
        "Return a linear ramp up from n1 to n0"
        rate = n1 + ((s * (n0 - n1)) / epoch)
        print (rate)
        return rate
    return decay_fn

lr_scheduler = keras.callbacks.LearningRateScheduler(linear_ramp_down)


# This didn't work. Using lr_scheduler in  callbacks=\[lr_scheduler\] failed and gave an error about the wrong datatype. Reading the [ipython notebook for the textbook](https://github.com/ageron/handson-ml2/blob/master/11_training_deep_neural_networks.ipynb) showed that this was not the correct method. Instead, you have to extend
# keras.callbacks.Callback and then update the lr in keras.backend.set_value(self.model.optimizer.lr, value)

# In[46]:


class OneCycleScheduler(keras.callbacks.Callback):
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


# In[47]:


from sklearn.base import clone


n_epochs = 100
batch_size=32
onecycle = OneCycleScheduler(len(X_train) // batch_size * n_epochs, max_rate=0.05)


# Alpha dropout work.
def normalized_dropout(n_classes=100, dropout_rate=0.2):
    """Keras multinomial logistic regression creation model

    Args:
        n_classes(int): Number of classes to be classified

    Returns:
        Compiled keras model

    """
    # create model
    model = keras.models.Sequential()

    # The input: we get 32x32 pixels, each with 3 colors (rgb). StandardScalar wants the dimensions flattened,
    # so now this gets the input directly.
    model.add(keras.layers.Flatten(input_shape=[3072]))

    # Apply dropout and then normalize
    keras.layers.Dropout(rate=dropout_rate)
    # Batch normalization after the input output.
    model.add(keras.layers.BatchNormalization())

    # Then the hidden layers, fully connected (100 by default)
    for i in range(20):
        model.add(keras.layers.Dense(
            n_classes,
            kernel_initializer="lecun_normal",
        ))
        # Apply dropout and then normalize
        keras.layers.Dropout(rate=dropout_rate)
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Activation("selu"))

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

# Got to remember them. mm_bn is the model with Batch normalization
mm_ramp = normalized_dropout(100)
print ("Model built: ", mm_drop)

# Now we must standard scale the data.
scaler = StandardScaler()
X_train_reshape = X_train.reshape(40000, 3072)
X_valid_reshape = X_valid.reshape(10000, 3072)
X_test_reshape = testX.reshape(10000, 3072)

X_train_ss = scaler.fit_transform(X_train_reshape)
# Always use the same scaler for the X_validation and X_test!
X_valid_ss = scaler.transform(X_valid_reshape)
X_test_ss = scaler.transform(X_test_reshape)

history_ramp = mm_ramp.fit(X_train_ss, y_train, epochs=100, verbose=0,
                 batch_size=32,
                 validation_data=(X_valid_ss, y_valid),
                 callbacks=[onecycle]) # Modify the learning-rate with a linear ramp-up just for testing



# In[48]:


import pandas as pd

pd.DataFrame(history_ramp.history).plot(figsize=(8,5))
plt.grid(True)
plt.gca().set_ylim(0,1) # Y axis set to [0,1]
plt.show()


# Wow, 1cycle actually performed much better.

# # Done with all exercises

# In[ ]:
