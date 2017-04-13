from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from collections import Counter

from unidecode import unidecode
from nltk import word_tokenize

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
    tokens = ["_START"] + tokens + ["_END"]
    return tokens


def vectorize(tokens, vocab):
    return [vocab.get(token, SPECIAL_TOKENS["_OOV"]) for token in tokens]


# TODO: cache
@shake_utils.cache
def build_vocab(path, max_vocab=10000, min_counts=5):
    counts = Counter()
    for sentence in _read_data(path):
        for token in tokenize(sentence):
            counts[token] += 1

    word2idx = {word: idx + len(SPECIAL_TOKENS) for idx, (word, count) in enumerate(counts.most_common(max_vocab)) if count > min_counts}
    word2idx.update(SPECIAL_TOKENS)
    idx2word = [i[0] for i in sorted(word2idx.iteritems(), key=lambda x: x[1])]
    return word2idx, idx2word


def _test():
    _check_parallel_data(default_modern_train_path, default_original_train_path)
    _check_parallel_data(default_modern_dev_path, default_original_dev_path)    

    source_vocab = build_vocab(default_modern_train_path)
    target_vocab = build_vocab(default_original_train_path)    

    text = "how are you?"
    tokens = tokenize(text)
    vector = vectorize(tokens, source_vocab[0])


if __name__ == "__main__":
    _test()