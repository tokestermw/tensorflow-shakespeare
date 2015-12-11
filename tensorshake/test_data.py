"""
Test if the data is inputted correctly.
"""
from itertools import izip

from . import CACHE_DIR
from .get_data import MODERN_TRAIN_PATH, ORIGINAL_TRAIN_PATH
from .prepare_corpus import MODERN_TRAIN_IDS_PATH, MODERN_DEV_IDS_PATH, ORIGINAL_TRAIN_IDS_PATH, ORIGINAL_DEV_IDS_PATH
from .prepare_corpus import MODERN_VOCAB_PATH, ORIGINAL_VOCAB_PATH
from .prepare_corpus import MODERN_VOCAB_MAX, ORIGINAL_VOCAB_MAX
from .prepare_corpus import tokenizer


# -- prepare cache first
def get_vocab(filename):
    vocab = {}
    with open(filename, 'r') as f:
        for idx, line in enumerate(f):
            vocab[int(idx)] = line.strip()
    return vocab

modern_vocab = get_vocab(MODERN_VOCAB_PATH)
original_vocab = get_vocab(ORIGINAL_VOCAB_PATH)


def test_token2id2token(first_n_lines=5):

    with open(MODERN_TRAIN_PATH, 'r') as raw_file, open(MODERN_TRAIN_IDS_PATH, 'r') as id_file:
        for idx, (raw_line, id_line) in enumerate(izip(raw_file, id_file)):
            constructed_sentence = ' '.join([modern_vocab[int(id)] for id in id_line.split()])
            assert constructed_sentence.replace(' ', '').lower() == raw_line.strip().replace(' ', '').lower()
            if idx >= first_n_lines:
                break

    with open(ORIGINAL_TRAIN_PATH, 'r') as raw_file, open(ORIGINAL_TRAIN_IDS_PATH, 'r') as id_file:
        for idx, (raw_line, id_line) in enumerate(izip(raw_file, id_file)):
            constructed_sentence = ' '.join([original_vocab[int(id)] for id in id_line.split()])
            assert constructed_sentence.replace(' ', '').lower() == raw_line.strip().replace(' ', '').lower()
            if idx >= first_n_lines:
                break


if __name__ == '__main__':
    test_token2id2token()