from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
from collections import Counter
from itertools import izip

from unidecode import unidecode
from nltk import word_tokenize

import numpy as np
import tensorflow as tf
from tensorflow.python.util import nest

import utils as shake_utils

DEFAULT_DATA_DIR = "data/shakespeare/processed"
default_modern_train_path = os.path.join(DEFAULT_DATA_DIR, "all_modern.snt.aligned_train")
default_modern_dev_path = os.path.join(DEFAULT_DATA_DIR, "all_modern.snt.aligned_dev")
default_original_train_path = os.path.join(DEFAULT_DATA_DIR, "all_original.snt.aligned_train")
default_original_dev_path = os.path.join(DEFAULT_DATA_DIR, "all_original.snt.aligned_dev")

DEFAULT_EMBEDDINGS_DIR = "embeddings"
default_glove_6B_50d_path = os.path.join(DEFAULT_EMBEDDINGS_DIR, "glove.6B.50d.txt")
default_glove_6B_100d_path = os.path.join(DEFAULT_EMBEDDINGS_DIR, "glove.6B.100d.txt")
default_glove_6B_200d_path = os.path.join(DEFAULT_EMBEDDINGS_DIR, "glove.6B.200d.txt")
default_glove_6B_300d_path = os.path.join(DEFAULT_EMBEDDINGS_DIR, "glove.6B.300d.txt")

SPECIAL_TOKENS = {"_PAD": 0, "_OOV": 1, "_START": 2, "_END": 3}

MAX_VOCAB = 10000
MIN_COUNTS = 5
MAXLEN = 100


def _read_data(path):
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            yield line


def tokenize(text):
    # TODO: spacy tokenizer
    text = text.lower()
    text = text.decode('utf8', 'ignore')
    text = unidecode(text)
    text = text.replace("' ", "'")
    tokens = word_tokenize(text)
    return tokens


def vectorize(tokens, vocab):
    vector = [vocab.get(token, SPECIAL_TOKENS["_OOV"]) for token in tokens]
    vector = vector[:(MAXLEN - 2)]
    vector = [SPECIAL_TOKENS["_START"]] + vector + [SPECIAL_TOKENS["_END"]]
    return vector
 

@shake_utils.cache
def build_vocab(path, max_vocab=MAX_VOCAB, min_counts=MIN_COUNTS):
    counts = Counter()
    for sentence in _read_data(path):
        for token in tokenize(sentence):
            counts[token] += 1

    word2idx = {}
    for idx, (word, count) in enumerate(counts.most_common(max_vocab)):
        if count > min_counts:
            word2idx[word] = idx + len(SPECIAL_TOKENS)
    word2idx.update(SPECIAL_TOKENS)

    idx2word = [i[0] for i in sorted(word2idx.iteritems(), key=lambda x: x[1])]

    assert max(v for v in word2idx.itervalues()) == (len(word2idx) - 1), \
           "vocab size doesn't match with the indices."
    return word2idx, idx2word


@shake_utils.cache
def build_vocab_with_embeddings(path, max_vocab=10000, min_counts=None):
    embeddings = []

    with open(path, 'r') as f:
        for idx, line in enumerate(f):
            if idx >= max_vocab:
                break
            elements = line.split()
            word = elements[0]
            neurons = map(float, elements[1:])
            embeddings.append((word, neurons))

    hidden_dim = len(neurons)
    bump = len(SPECIAL_TOKENS)

    word2idx = {
        word: idx + bump for idx, (word, neurons) in enumerate(embeddings)
    }
    word2idx.update(SPECIAL_TOKENS)
    idx2word = [i[0] for i in sorted(word2idx.iteritems(), key=lambda x: x[1])]

    vocab_size = len(word2idx)

    embedding_matrix = np.zeros(shape=(vocab_size, hidden_dim), 
        dtype=np.float32)

    vocab_size = len(word2idx)

    for i in range(bump):
        if i == 0:
            embedding_matrix[i, :] = np.zeros(hidden_dim, dtype=np.float32)
        else:
            span = np.sqrt(6. / (hidden_dim + vocab_size))  # xavier uniform
            embedding_matrix[i, :] = np.random.uniform(-span, span, hidden_dim)

    for idx, (word, neurons) in enumerate(embeddings):
        embedding_matrix[idx + bump, :] = np.array(neurons, dtype=np.float32)

    assert len(word2idx) == embedding_matrix.shape[0]
    return word2idx, idx2word, embedding_matrix


def build_dataset(path, vocab,
                  batch_size=5, epoch_size=2):
    filenames = [path]
    dataset = tf.contrib.data.TextLineDataset(filenames)

    def _featurize_py_func(text):
        tokens = tokenize(text)
        vector = vectorize(tokens, vocab)
        return np.array(vector, dtype=np.int32)

    dataset = (dataset.map(lambda text: tf.py_func(
                          _featurize_py_func, [text], [tf.int32]))
                      .skip(0)
                      .padded_batch(batch_size, padded_shapes=[MAXLEN])
                      .repeat(epoch_size))

    return dataset


class ParallelDataset:
    def __init__(self, source_path, target_path,
                 batch_size=5, epoch_size=2):
        self._source_path = source_path
        self._target_path = target_path

        self._batch_size = batch_size
        self._epoch_size = epoch_size

        self._source_word2idx, self._source_idx2word = \
            build_vocab(source_path)
        self._target_word2idx, self._target_idx2word = \
            build_vocab(target_path)

        self._source_dataset = build_dataset(
            source_path, self._source_word2idx,
            self._batch_size, self._epoch_size)
        self._target_dataset = build_dataset(
            target_path, self._target_word2idx,
            self._batch_size, self._epoch_size)

        self._parallel_dataset = tf.contrib.data.Dataset.zip(
            [self._source_dataset, self._target_dataset]).shuffle(
            buffer_size=10000)

        self._iterator = self._parallel_dataset.make_one_shot_iterator()
        self._next_element = self._iterator.get_next()
        self._source_inputs, self._target_inputs = nest.flatten(
            self._next_element)

    @property
    def source_word2idx(self):
        return self._source_word2idx

    @property
    def source_idx2word(self):
        return self._source_idx2word

    @property
    def source_vocab_size(self):
        return len(self._source_word2idx)

    @property
    def target_word2idx(self):
        return self._target_word2idx
    
    @property
    def target_idx2word(self):
        return self._target_idx2word

    @property
    def target_vocab_size(self):
        return len(self._target_word2idx)

    @property
    def source_inputs(self):
        return self._source_inputs
    
    @property
    def target_inputs(self):
        return self._target_inputs


def _test():
    dataset = ParallelDataset(
        default_modern_dev_path, default_original_dev_path,
        32, 2)

    with tf.Session() as sess:
        while True:
            try:
                out = sess.run([dataset.source_inputs, dataset.target_inputs])
                shapes = (out[0].shape, out[1].shape)
                assert shapes[0] == shapes[1], shapes
            except tf.errors.OutOfRangeError:
                print("end of data")
                break


if __name__ == "__main__":
    _test()
