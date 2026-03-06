import pickle
import os

def get_cached_valid_combinations(combination_cache, combos):
    """
    Given a cache and a list of combos, return a tuple:
    (list of valid combos from cache, list of combos to compute)
    """
    cached = set()
    for key, valids in combination_cache.items():
        cached.update(valids)
    cached = set(cached)
    combos_set = set(combos)
    already_valid = list(cached & combos_set)
    to_compute = list(combos_set - cached)
    return already_valid, to_compute

def load_cache(filename):
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
    return {}

def save_cache(filename, cache):
    with open(filename, 'wb') as f:
        pickle.dump(cache, f)
