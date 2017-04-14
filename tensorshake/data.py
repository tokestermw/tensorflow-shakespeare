from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
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

SPECIAL_TOKENS = {"_PAD": 0, "_OOV": 1, "_START": 2, "_END": 3}


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

    for source_text, target_text in izip(_read_data(source_path), _read_data(target_path)):
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


def _test():
    _check_parallel_data(default_modern_train_path, default_original_train_path)
    _check_parallel_data(default_modern_dev_path, default_original_dev_path)    

    source_vocab = build_vocab(default_modern_train_path)
    target_vocab = build_vocab(default_original_train_path)    

    text = "how are you?"
    tokens = tokenize(text)
    vector = vectorize(tokens, source_vocab[0])

    for i_, j_ in data_iterator(default_modern_train_path, source_vocab[0], default_modern_train_path, target_vocab[0]):
        print(i_)


if __name__ == "__main__":
    _test()