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
MAXLEN = 100


def _read_data(path):
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            yield line


def _check_parallel_data(source_path, target_path):
    source_line_counts = 0
    for source_text in _read_data(source_path):
        if source_line_counts == 0:
            print(source_line_counts, "\t", source_text)
        source_line_counts += 1

    target_line_counts = 0
    for target_text in _read_data(target_path):
        if target_line_counts == 0:
            print(target_line_counts, "\t", target_text)
        target_line_counts += 1
        
    print(source_line_counts, "\t", source_text)
    print(target_line_counts, "\t", target_text)
    assert source_line_counts == target_line_counts, "source and target should have same number of sentences."


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


def batch_and_pad(list_of_vectors, maxlen=None):
    maxlen = max([len(v) for v in list_of_vectors])
    padded_sequences = np.zeros((len(list_of_vectors), maxlen), dtype=np.int32)
    for i, v in enumerate(list_of_vectors):
        l = min(len(v), maxlen)
        padded_sequences[i, :l] = v[:l]
    return padded_sequences


# TODO: cleaner iterator with batch and pad
def data_iterator(source_path, source_vocab, target_path, target_vocab, batch_size=32, maxlen=100):
    batch_source, batch_target, batch_counter = [], [], 0
    # put in memory, randomize the data (so important!)

    source_data = list(_read_data(source_path))
    target_data = list(_read_data(target_path))
    line_ids = range(len(source_data))
    random.shuffle(line_ids)

    # TODO: memory issue? ideally do it online
    source_data_rando = [source_data[i] for i in line_ids]
    target_data_rando = [target_data[i] for i in line_ids]

    # for source_text, target_text in izip(_read_data(source_path), _read_data(target_path)):
    for source_text, target_text in izip(source_data_rando, target_data_rando):
        source_tokens = tokenize(source_text)
        source_vector = vectorize(source_tokens, source_vocab)
        batch_source.append(source_vector)

        target_tokens = tokenize(target_text)
        target_vector = vectorize(target_tokens, target_vocab)
        batch_target.append(target_vector)

        batch_counter += 1
        if batch_counter >= batch_size:
            yield batch_and_pad(batch_source, maxlen=maxlen), batch_and_pad(batch_target, maxlen=maxlen)
            batch_source, batch_target, batch_counter = [], [], 0

    # not sure if this helps
    del source_data, target_data, source_data_copy, target_data_copy, line_ids


@shake_utils.cache
def build_vocab(path, max_vocab=10000, min_counts=5):
    counts = Counter()
    for sentence in _read_data(path):
        for token in tokenize(sentence):
            counts[token] += 1

    word2idx = {word: idx + len(SPECIAL_TOKENS) for idx, (word, count) in enumerate(counts.most_common(max_vocab)) if count > min_counts}
    word2idx.update(SPECIAL_TOKENS)
    idx2word = [i[0] for i in sorted(word2idx.iteritems(), key=lambda x: x[1])]

    assert max(v for v in word2idx.itervalues()) == (len(word2idx) - 1), "vocab size doesn't match with the indices."
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

    word2idx = {word: idx + bump for idx, (word, neurons) in enumerate(embeddings)}
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


def _test():
    _check_parallel_data(default_modern_train_path, default_original_train_path)
    _check_parallel_data(default_modern_dev_path, default_original_dev_path)    

    source_vocab = build_vocab(default_modern_train_path)
    target_vocab = build_vocab(default_original_train_path)    

    source_vocab_with_embeddings = build_vocab_with_embeddings(default_glove_6B_50d_path)

    text = "how are you?"
    tokens = tokenize(text)
    vector = vectorize(tokens, source_vocab[0])

    for i_, j_ in data_iterator(default_modern_train_path, source_vocab[0], default_modern_train_path, target_vocab[0]):
        print(i_.shape, j_.shape)
        if sum(i_.shape) != sum(j_.shape):
            import pdb
            pdb.set_trace()
            a = None


if __name__ == "__main__":
    _test()