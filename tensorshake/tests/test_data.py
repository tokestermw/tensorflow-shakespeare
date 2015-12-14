"""
Test if the data is inputted correctly.
"""
from itertools import izip

from tensorshake.get_data import MODERN_TRAIN_PATH, ORIGINAL_TRAIN_PATH, MODERN_DEV_PATH, ORIGINAL_DEV_PATH
from tensorshake.prepare_corpus import MODERN_TRAIN_IDS_PATH, MODERN_DEV_IDS_PATH, ORIGINAL_TRAIN_IDS_PATH, ORIGINAL_DEV_IDS_PATH
from tensorshake.prepare_corpus import MODERN_VOCAB_PATH, ORIGINAL_VOCAB_PATH


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
            print 'constructed:', constructed_sentence
            print 'raw:', raw_line
            # assert constructed_sentence.replace(' ', '').lower() == raw_line.strip().replace(' ', '').lower()
            if idx >= first_n_lines:
                break

    with open(MODERN_DEV_PATH, 'r') as raw_file, open(MODERN_DEV_IDS_PATH, 'r') as id_file:
        for idx, (raw_line, id_line) in enumerate(izip(raw_file, id_file)):
            constructed_sentence = ' '.join([modern_vocab[int(id)] for id in id_line.split()])
            print 'constructed:', constructed_sentence
            print 'raw:', raw_line
            # assert constructed_sentence.replace(' ', '').lower() == raw_line.strip().replace(' ', '').lower()
            if idx >= first_n_lines:
                break

    with open(ORIGINAL_TRAIN_PATH, 'r') as raw_file, open(ORIGINAL_TRAIN_IDS_PATH, 'r') as id_file:
        for idx, (raw_line, id_line) in enumerate(izip(raw_file, id_file)):
            constructed_sentence = ' '.join([original_vocab[int(id)] for id in id_line.split()])
            print 'constructed:', constructed_sentence
            print 'raw:', raw_line
            # assert constructed_sentence.replace(' ', '').lower() == raw_line.strip().replace(' ', '').lower()
            if idx >= first_n_lines:
                break

    with open(ORIGINAL_DEV_PATH, 'r') as raw_file, open(ORIGINAL_DEV_IDS_PATH, 'r') as id_file:
        for idx, (raw_line, id_line) in enumerate(izip(raw_file, id_file)):
            constructed_sentence = ' '.join([original_vocab[int(id)] for id in id_line.split()])
            print 'constructed:', constructed_sentence
            print 'raw:', raw_line
            # assert constructed_sentence.replace(' ', '').lower() == raw_line.strip().replace(' ', '').lower()
            if idx >= first_n_lines:
                break


if __name__ == '__main__':
    test_token2id2token()