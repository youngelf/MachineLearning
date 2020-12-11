import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_datasets as tfds

import numpy as np

# todo: Should pull out the common NN code in a library rather than chapter10.
import Chapter_10 as c10

# Chapter 16: Natural language processing with RNNs and Attention.

def get_shakespeare():
    """Download and return the training and test data

    Call with:
    tokenizer, encoded, max_id, dataset_size = get_shakespeare()
    """
    shakespeare_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    filepath = keras.utils.get_file("shakespeare.txt", shakespeare_url)
    with open(filepath) as f:
        shakespeare_text = f.read()

    # Break into characters
    tokenizer = keras.preprocessing.text.Tokenizer(char_level=True)
    tokenizer.fit_on_texts(shakespeare_text)

    # Total number of unique characters
    max_id = len(tokenizer.word_index)
    # Total number of characters
    dataset_size = tokenizer.document_count

    # Turn from chars to numbers
    [encoded] = np.array(tokenizer.texts_to_sequences([shakespeare_text])) - 1

    # 90% of the data is for training
    train_size = (dataset_size * 90) // 100
    dataset = tf.data.Dataset.from_tensor_slices(encoded[:train_size])

    # Input 100 characters at a time
    n_steps = 100
    # Predict a single character. TODO(viki): Change this and try
    window_length = n_steps + 1
    # Create overlapping windows, with input data shifted by 1
    # character at a time. Discard anything smaller than window_length
    # (drop the last few characters since they're smaller)
    dataset = dataset.window(window_length, shift=1, drop_remainder=True)

    # The previous method returns a nested list, we would like
    # flattened lists so that everytime we ask for a dataset, we get a
    # single instance.
    dataset = dataset.flat_map(lambda window: window.batch(window_length))

    # Shuffle the dataset, and separate out the features (first 100
    # values) and the prediction (last value)
    batch_size = 32
    random_seed = 10000
    dataset = dataset.shuffle(random_seed).batch(batch_size)
    dataset = dataset.map(lambda window: (window[:, :-1], window[:, 1:]))

    # Since we only have roughly 40 distinct characters, return a
    # one-hot encoding rather than integers.
    dataset = dataset.map(lambda X_batch, y_batch: (tf.one_hot(X_batch, depth=max_id), y_batch))

    # Warm up the memory, so the next item is ready when fetching one.
    dataset = dataset.prefetch(1)
    
    return tokenizer, encoded, max_id, dataset


def make_character_model(max_id):
    """Creates a character-based RNN model to generate a single character
    when fed a window of prior characters.

    """
    model = keras.models.Sequential([
        keras.layers.GRU(128, return_sequences=True, input_shape=[None, max_id],
                         dropout=0.2, recurrent_dropout=0.2),
        keras.layers.GRU(128, return_sequences=True,
                         dropout=0.2, recurrent_dropout=0.2),
        keras.layers.TimeDistributed(
            keras.layers.Dense(max_id, activation='softmax')),
    ])
    model.compile(loss="sparse_categorical_crossentropy", optimizer='adam')

    return model


def preprocess(in_text):
    """ Convert input text into a one-hot array for use in the shakespeare model"""
    X = np.array(tokenizer.texts_to_sequences(in_text)) - 1
    return tf.one_hot(X, max_id)


def next_char(text, tokenizer, temperature=0.1):
    """Generate a string of text, given just a character to start with"""
    X_new = preprocess([text])
    y_proba = model.predict(X_new)[0, -1:, :]
    rescaled_logits = tf.math.log(y_prob) / temperature
    char_id = tf.random.categorical(rescaled_logits, num_samples=1) + 1
    return tokenizer.sequences_to_texts(char_id.numpy())[0]

def complete_text(text, tokenizer, n_chars=50, temperature=0.1):
    for _ in range(n_chars):
        text += next_char(text, temperature)
    return text

def run_all_16():
    tokenizer, encoded, max_id, dataset = get_shakespeare()
    model = make_character_model(max_id)

    history = model.fit(dataset, epochs=20)

    # Can it complete: How are you
    X_new = preprocess(["How are yo"])
    y_pred = model.predict_classes(X_new)
    # Get the vector converted to the characters, take the first element, and the final character
    sentence = tokenizer.sequences_to_texts(y_pred + 1)[0]
    print ("Predicted sentence: ", sentence)
    predicted_character = sentence[-1]
    print ("Predicted character: ", predicted_character)

    print(complete_text('t', temperature=0.2))

print ("All done, starting out with Python directly now")
