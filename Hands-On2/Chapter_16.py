import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_datasets as tfds

import numpy as np

# todo: Should pull out the common NN code in a library rather than chapter10.
import Chapter_10 as c10

# Chapter 16: Natural language processing with RNNs and Attention.

def get_shakespeare():
    """Download and return the training and test data
    """
    shakespeare_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    filepath = keras.utils.get_file("shakespeare.txt", shakespeare_url)
    with open(filepath) as f:
        shakespeare_text = f.read()

    tokenizer = keras.preprocessing.text.Tokenizer(char_level=True)
    tokenizer.fit_on_texts(shakespeare_text)

    return tokenizer

def run_all_16():
    tokenizer = get_shakespeare()


print ("All done, starting out with Python directly now")
