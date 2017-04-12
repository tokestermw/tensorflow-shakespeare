import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# print 'ROOT_DIR', ROOT_DIR

def get_dir(relative_path=''):
    if ROOT_DIR == '':
        raise Exception("dafaq")
    return os.path.join(ROOT_DIR, relative_path)

CACHE_DIR = get_dir('cache')

# print 'CACHE_DIR', CACHE_DIR

if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

def delete_cache():
    os.remove(CACHE_DIR)
