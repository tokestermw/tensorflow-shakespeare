# coding=utf-8
"""
Test tokenizer.
Malformed apostrophes, quotes are the most pernicious.
Proper unicode needed for nltk tokenizer.
"""
from __future__ import unicode_literals

from tensorshake.get_data import ORIGINAL_DEV_PATH, ORIGINAL_TRAIN_PATH
from tensorshake.prepare_corpus import tokenizer


TEXT_NTOKENS = {
    "I'll go you.": 5,
    "I 'll go with you.": 6,
    "i ’ ll go with you.": 6,
    "i’ ll go with you.": 6,
    '"hello there."': 5,
    "Fie upon “But yet.” “But yet”": 11,
    "I am paid for ’t now.": 7,
    "We’ll speak with thee at sea.": 8,
    "Show ’s the way, sir.": 7,
    "More light and light—more dark and dark our woes!": 12,
}


def test_ntokens():
    for text, ntokens in TEXT_NTOKENS.iteritems():
        tokens = tokenizer(text)
        assert ntokens == len(tokens)


TEXT_MALFORMED = {
    "I 'll go with you.": "I'll go with you.",
    "i ’ ll go with you.": "I'll go with you.",
    "Fie upon “But yet.” “But yet”": 'Fie upon "But yet." "But yet"',
    "More light and light—more dark and dark our woes!": "More light and light -- more dark and dark our woes!"
}


def test_malformed():
    for text1, text2 in TEXT_MALFORMED.iteritems():
        tokens1 = tokenizer(text1)
        tokens2 = tokenizer(text2)
        assert tokens1 == tokens2


def test_on_data():
    with open(ORIGINAL_TRAIN_PATH, 'r') as raw_file:
        weird_chars = {'thatas', 'thereas'}
        counter = 0
        for idx, raw_line in enumerate(raw_file):
            text = ' '.join(tokenizer(raw_line))
            for char in weird_chars:
                if char in text:
                    print text
                    counter += 1
    assert counter == 0


if __name__ == '__main__':
    test_ntokens()
    test_malformed()
    test_on_data()
