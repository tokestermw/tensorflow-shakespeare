"""
Build vocab with a set max vocab size.
Build token ids given the vocab.
Do get_data.py first.
"""
from __future__ import unicode_literals, print_function, division

import os
import subprocess
import re

from data_utils import create_vocabulary, data_to_token_ids
from get_data import CACHE_DIR, MODERN_FILENAME, ORIGINAL_FILENAME, TRAIN_SUFFIX, DEV_SUFFIX
from get_data import MODERN_PATH, ORIGINAL_PATH, MODERN_TRAIN_PATH, MODERN_DEV_PATH, ORIGINAL_TRAIN_PATH, ORIGINAL_DEV_PATH

MODERN_VOCAB_FILENAME = "all_modern.vocab"
ORIGINAL_VOCAB_FILENAME = "all_original.vocab"

MODERN_VOCAB_MAX = 20000
ORIGINAL_VOCAB_MAX = 20000

IDS_SUFFIX = ".ids"

MODERN_VOCAB_PATH = os.path.join(CACHE_DIR, MODERN_VOCAB_FILENAME)
ORIGINAL_VOCAB_PATH = os.path.join(CACHE_DIR, ORIGINAL_VOCAB_FILENAME)

MODERN_TRAIN_IDS_PATH = os.path.join(CACHE_DIR, "all_modern" + TRAIN_SUFFIX + ".ids")
MODERN_DEV_IDS_PATH = os.path.join(CACHE_DIR, "all_modern" + DEV_SUFFIX + ".ids")
ORIGINAL_TRAIN_IDS_PATH = os.path.join(CACHE_DIR, "all_original" + TRAIN_SUFFIX + ".ids")
ORIGINAL_DEV_IDS_PATH = os.path.join(CACHE_DIR, "all_original" + DEV_SUFFIX + ".ids")

_WORD_SPLIT = re.compile("([.,!?\"':;)(])")


def tokenizer(sentence):
    """Very basic tokenizer: split the sentence into a list of tokens + lower()."""
    words = []
    for space_separated_fragment in sentence.lower().strip().split():
        words.extend(re.split(_WORD_SPLIT, space_separated_fragment))
    return [w for w in words if w]


def build_vocab():
    create_vocabulary(MODERN_VOCAB_PATH, MODERN_PATH, MODERN_VOCAB_MAX, tokenizer=tokenizer)
    create_vocabulary(ORIGINAL_VOCAB_PATH, ORIGINAL_PATH, ORIGINAL_VOCAB_MAX, tokenizer=tokenizer)

    print( subprocess.check_output(['wc', '-l', MODERN_VOCAB_PATH]) )
    print( subprocess.check_output(['wc', '-l', ORIGINAL_VOCAB_PATH]) )


def build_ids():
    data_to_token_ids(MODERN_TRAIN_PATH, MODERN_TRAIN_IDS_PATH, MODERN_VOCAB_PATH, tokenizer=tokenizer)
    data_to_token_ids(MODERN_DEV_PATH, MODERN_DEV_IDS_PATH, MODERN_VOCAB_PATH, tokenizer=tokenizer)
    data_to_token_ids(ORIGINAL_TRAIN_PATH, ORIGINAL_TRAIN_IDS_PATH, ORIGINAL_VOCAB_PATH, tokenizer=tokenizer)
    data_to_token_ids(ORIGINAL_DEV_PATH, ORIGINAL_DEV_IDS_PATH, ORIGINAL_VOCAB_PATH, tokenizer=tokenizer)

    print( subprocess.check_output(['wc', '-l', MODERN_TRAIN_IDS_PATH]) )
    print( subprocess.check_output(['wc', '-l', MODERN_DEV_IDS_PATH]) )
    print( subprocess.check_output(['wc', '-l', ORIGINAL_TRAIN_IDS_PATH]) )
    print( subprocess.check_output(['wc', '-l', ORIGINAL_DEV_IDS_PATH]) )


if __name__ == '__main__':
    build_vocab()
    build_ids()