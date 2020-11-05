
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
