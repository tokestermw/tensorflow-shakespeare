from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import hashlib
import functools
import cPickle as pickle

CACHE_DIR = "cache"
if not os.path.isdir(CACHE_DIR):
    os.mkdir(CACHE_DIR)


def cache(f):
    @functools.wraps(f)
    def _cache(*args, **kwargs):
        str_args = ''.join(map(str, args))
        str_keys = ''.join(map(str, kwargs.keys()))
        str_values = ''.join(map(str, kwargs.values()))

        m = hashlib.md5()
        m.update(str_args)
        m.update(str_keys)
        m.update(str_values)

        hash_filename = "{}_{}.pkl".format(f.__name__, m.hexdigest())
        hash_path = os.path.join(CACHE_DIR, hash_filename)

        if os.path.isfile(hash_path):
            with open(hash_path, 'rb') as file_:
                obj = pickle.load(file_)
            out = obj

        else:
            out = f(*args, **kwargs)
            with open(hash_path, 'wb') as file_:
                pickle.dump(out, file_)

        return out
    return _cache


def save_config(opts, model_dir, filename="config.json"):
    path = os.path.join(model_dir, filename)

    with open(path, 'w') as f:
        json.dump(opts, f, indent=4)


def load_config(model_dir, filename="config.json"):
    path = os.path.join(model_dir, filename)
    with open(path, 'r') as f:
        opts = json.load(f)
    return opts


if __name__ == "__main__":
    @cache
    def add(a=5):
        return a

    print(add(5))
