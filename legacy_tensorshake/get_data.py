"""
Simple code:
1. get parallel Shakespeare data
2. store in ./cache folder
"""
# from __future__ import unicode_literals, print_function, division

import os
import subprocess

from tensorshake import get_dir, CACHE_DIR

DATA_LINKS = {
    "data/shakespeare/sparknotes/merged": "https://github.com/cocoxu/Shakespeare/tree/master/data/align/plays/merged",
    # "data/shakespeare/sparknotes/merged_except_romeo": "https://github.com/cocoxu/Shakespeare/tree/master/data/align/plays/merged_except_romeo",
    "data/shakespeare/enotes/merged": "https://github.com/cocoxu/Shakespeare/tree/master/data/align/plays2/merged",
    # "data/shakespeare/enotes/merged_except_romeo": "https://github.com/cocoxu/Shakespeare/tree/master/data/align/plays2/merged_except_romeo"
}

MODERN_FILENAME = "all_modern.snt.aligned"
ORIGINAL_FILENAME = "all_original.snt.aligned"

TRAIN_SUFFIX = "_train"
DEV_SUFFIX = "_dev"

MODERN_PATH = os.path.join(CACHE_DIR, MODERN_FILENAME)
ORIGINAL_PATH = os.path.join(CACHE_DIR, ORIGINAL_FILENAME)

MODERN_TRAIN_PATH = MODERN_PATH + TRAIN_SUFFIX
MODERN_DEV_PATH = MODERN_PATH + DEV_SUFFIX
ORIGINAL_TRAIN_PATH = ORIGINAL_PATH + TRAIN_SUFFIX
ORIGINAL_DEV_PATH = ORIGINAL_PATH + DEV_SUFFIX


def get_shakespeare_parallel_set():    
    # combine aligned data and put into ./cache
    
    modern_file = open(os.path.join(CACHE_DIR, MODERN_FILENAME), 'w')
    original_file = open(os.path.join(CACHE_DIR, ORIGINAL_FILENAME), 'w')

    for aligned_data in DATA_LINKS:
        for root, dirs, filenames in os.walk(get_dir(aligned_data)):
            for filename in sorted(filenames):
                with open(os.path.join(get_dir(aligned_data), filename), 'r') as f:
                    for line in f:
                        if '_modern.snt.aligned' in filename:
                            modern_file.write(line.strip())
                            modern_file.write('\n')
                        elif '_original.snt.aligned' in filename:
                            original_file.write(line.strip())
                            original_file.write('\n')
                        else:
                            pass

    modern_file.close()
    original_file.close()

    print( subprocess.check_output(['wc', '-l', MODERN_PATH]) )
    print( subprocess.check_output(['wc', '-l', ORIGINAL_PATH]) )


def split_shakespeare_parallel_set(split_size=30000):
    # split the dataset into train and dev sets (i'm using shell scripts)

    subprocess.call(['split', '-l', str(split_size), MODERN_PATH])
    subprocess.call(['mv', 'xaa', MODERN_TRAIN_PATH])
    subprocess.call(['mv', 'xab', MODERN_DEV_PATH])

    subprocess.call(['split', '-l', str(split_size), ORIGINAL_PATH])
    subprocess.call(['mv', 'xaa', ORIGINAL_TRAIN_PATH])
    subprocess.call(['mv', 'xab', ORIGINAL_DEV_PATH])

    print( subprocess.check_output(['wc', '-l', MODERN_TRAIN_PATH]) )
    print( subprocess.check_output(['wc', '-l', MODERN_DEV_PATH]) )
    print( subprocess.check_output(['wc', '-l', ORIGINAL_TRAIN_PATH]) )
    print( subprocess.check_output(['wc', '-l', ORIGINAL_DEV_PATH]) )


if __name__ == '__main__':
    get_shakespeare_parallel_set()
    split_shakespeare_parallel_set()
