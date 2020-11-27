import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_datasets as tfds


def load_flowers_data():
    """Loads the flowers database from TFDS.

    Call with:
    dataset, info = load_flowers_data()
    """
    dataset, info = tfds.load("tf_flowers", as_supervised=True, with_info=True)
    dataset_size = info.splits["train"].num_examples
    print ("Size of dataset: ", dataset_size)

    class_names = info.features["label"].names
    n_classes = info.features["label"].num_classes
    print ("Number of classes: ", n_classes)

    return dataset, info

def train_test_split_flowers():
    """Split the database into training and testing

    Call with:
    train, valid, test = train_test_split_flowers()
    """

    # This is what the book asked for, but it doesn't work.
    # test_split, valid_split, train_split = tfds.Split.TRAIN.subsplit([10, 15, 75])

    # The first 10% as test
    test_set = tfds.load("tf_flowers", split='train[:10%]', as_supervised=True)
    # The next 15% as test
    valid_set = tfds.load("tf_flowers", split='train[10%:25%]', as_supervised=True)
    # The last 75% as test
    train_set = tfds.load("tf_flowers", split='train[-75%:]', as_supervised=True)

    return train_set, valid_set, test_set


def preprocess_xception(image, label):
    """Preprocess the image in a way that the XCeption model expects.

    Call with:
    train_set = train_set.map(preprocess_xception).batch(32).prefetch(2)
    """
    resized_image = tf.image.resize(image, [224, 224])
    final_image = keras.applications.xception.preprocess_input(resized_image)
    return final_image, label

def evaluate(model, test_set):
    """ Evaluate the model on the test data.

    Call with:
    loss, accuracy = evaluate(simplest, test_set)

    """

    (loss, accuracy) = model.evaluate(test_set)

    print ("Loss is %f" % loss)
    print ("Accuracy is %f" % accuracy)

    return loss, accuracy

def run_all_14():
    dataset_all, info_all = load_flowers_data()
    n_classes = info_all.features["label"].num_classes
    train_set, valid_set, test_set = train_test_split_flowers()

    # Preprocess the data
    batch_size = 32
    train_set = train_set.shuffle(1000)
    train_set = train_set.map(preprocess_xception).batch(batch_size).prefetch(2)
    valid_set = valid_set.map(preprocess_xception).batch(batch_size).prefetch(2)
    test_set = test_set.map(preprocess_xception).batch(batch_size).prefetch(2)

    # Load the Xcpetion model, pretrained on ImageNet
    base_model = keras.applications.Xception(weights="imagenet",
                                             include_top=False)
    avg = keras.layers.GlobalAveragePooling2D()(base_model.output)
    output = keras.layers.Dense(n_classes, activation="softmax")(avg)
    model = keras.Model(inputs=base_model.input, outputs=output)

    # Freeze the base (pretrained) layers, at least initially
    for layer in base_model.layers:
        layer.trainable = False

    # Now compile the model
    optimizer = keras.optimizers.SGD(lr=0.2, momentum=0.9, decay=0.01)
    model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer,
                  metrics=["accuracy"])
    history = model.fit(train_set, epochs=5, validation_data=valid_set, verbose=0)

    # Evaluate the error half-way (before unfreezing) to see what
    # impact it has on the model quality.
    loss, accuracy = evaluate(model, test_set)
    print ("First pass: Loss is %f" % loss)
    print ("First pass: Accuracy is %f" % accuracy)

    # Now you can unfreeze the base layers.
    for layer in base_model.layers:
        layer.trainable = True

    # Lower learning rate, and a lower decay this time, to avoid
    # damaging the base layers too much.
    optimizer = keras.optimizers.SGD(lr=0.01, momentum=0.9, decay=0.001)
    model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer,
                  metrics=["accuracy"])
    history = model.fit(train_set, epochs=5, validation_data=valid_set, verbose=0)

    # Evaluate the error at the end (after unfreezing) to see what
    # impact it has on the model quality.
    loss, accuracy = evaluate(model, test_set)
    print ("End: Loss is %f" % loss)
    print ("End: Accuracy is %f" % accuracy)
    # I should try training it differently, more epochs, less epochs.

    # I should also try evaluating the error half-way (before
    # unfreezing) to see what impact it has on the model quality.

    # Finally, I should try unfreezing just the top layers, and see
    # what works better.



# Exercises here:

# Exercise 9: Create a CNN for digits mnist
import Chapter_10 as c10


def prepare_digits_mnist_data(X_train, X_valid, X_test, debug=False):
    if (debug):
        print ("X_train shape = ", X_train.shape)
        print ("X_valid shape = ", X_valid.shape)
        print ("X_test shape = ", X_test.shape)

    X_train = X_train.reshape(55000,28,28,1)
    X_valid = X_valid.reshape(5000,28,28,1)
    X_test = X_test.reshape(10000,28,28,1)
    return X_train, X_valid, X_test


# Test out creating a model. This will need some iterations.
def create_e9_model(optimizer="sgd"):
    deep_model = keras.models.Sequential([
        keras.layers.Conv2D(32, 4, activation="relu", padding="same",
                            input_shape=(28, 28, 1), name="input"),
        keras.layers.MaxPooling2D(1,name="firstPool"),
        keras.layers.Conv2D(128, 3, activation="relu", padding="same",
                            name="first_conv_1"),
        keras.layers.Conv2D(128, 3, activation="relu", padding="same",
                            name="first_conv_2"),

        keras.layers.MaxPooling2D(1, name="secondPool"),
        keras.layers.Conv2D(256, 3, activation="relu", padding="same",
                            name="second_conv_1"),
        keras.layers.Conv2D(256, 3, activation="relu", padding="same",
                            name="second_conv_2"),

        keras.layers.MaxPooling2D(1, name="thirdPool"),

        keras.layers.Flatten(name="flatten"),
        keras.layers.Dense(128, activation="relu", name="pre-bottneck"),

        keras.layers.Dropout(0.5, name="bottleneckDropout"),
        keras.layers.Dense(64, activation="relu", name="bottleneck"),

        keras.layers.Dropout(0.5, name="outputDropout"),
        keras.layers.Dense(10, activation="softmax", name="output"),
    ])

    deep_model.compile(loss="sparse_categorical_crossentropy",
                      optimizer=optimizer,
                      metrics=["accuracy"])

    return deep_model

def fit_e9_model(model, X_train, y_train, X_valid, y_valid, epochs):
    history_conv = model.fit(x=X_train, y=y_train, batch_size=32, validation_data=[X_valid, y_valid],
                             epochs=epochs, verbose=0)
    return history_conv

def plot_e9_history(history, name):
    c10.plot_training(history, name, show=False)


def run_c14_e9():
    """ Run Chapter 14, Exercise 9 """
    X_train, X_valid, X_test, y_train, y_valid, y_test = c10.load_digits_mnist(debug=True)
    X_train, X_valid, X_test = prepare_digits_mnist_data(X_train, X_valid, X_test)

    # Create the model and train it
    model = create_e9_model()
    history = fit_e9_model(model, X_train, y_train, X_valid, y_valid, epochs=10)
    plot_e9_history(history, "naive_deep_mnist")

