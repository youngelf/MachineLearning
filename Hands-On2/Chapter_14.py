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
    train, test, valid = train_test_split_flowers()
    """

    # This is what the book asked for, but it doesn't work.
    # test_split, valid_split, train_split = tfds.Split.TRAIN.subsplit([10, 15, 75])

    # The first 10% as test
    test_set = tfds.load("tf_flowers", split='train[:10%]', as_supervised=True)
    # The next 15% as test
    valid_set = tfds.load("tf_flowers", split='train[10%:25%]', as_supervised=True)
    # The last 75% as test
    train_set = tfds.load("tf_flowers", split='train[-75%:]', as_supervised=True)

    return test_set, valid_set, train_set


def preprocess_xception(image, label):
    """Preprocess the image in a way that the XCeption model expects.

    Call with:
    train_set = train_set.map(preprocess_xception).batch(32).prefetch(2)
    """
    resized_image = tf.image.resize(image, [224, 224])
    final_image = keras.applications.xception.preprocess_input(resized_image)
    return final_image, label

def run_all_14():
    dataset_all, info_all = load_flowers_data()
    n_classes = info_all.features["label"].num_classes
    train_set, test_set, valid_set = train_test_split_flowers()

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
    history = model.fit(train_set, epochs=5, validation_data=valid_set)

    # Now you can unfreeze the base layers.
    for layer in base_model.layers:
        layer.trainable = True

    # Lower learning rate, and a lower decay this time, to avoid
    # damaging the base layers too much.
    optimizer = keras.optimizers.SGD(lr=0.01, momentum=0.9, decay=0.001)
    model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer,
                  metrics=["accuracy"])
    history = model.fit(train_set, epochs=5, validation_data=valid_set)
    


    
    
