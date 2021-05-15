import os
import json
import logging

_path = os.path.dirname(__file__)


def save_load_results_to_json(func, print=True):
    """Saves and loads results of function to json."""
    def wrapper(filename, *args, **kwargs):
        full_path = os.path.join(_path, filename)

        if os.path.exists(full_path):
            if print:
                logging.info("Loading results for %s from %s." % (func.__name__, filename))
            with open(full_path, "r") as f:
                return json.load(f)

        obj = func(*args, **kwargs)

        with open(full_path, "w") as f:
            if print:
                logging.info("Saving results for %s from %s." % (func.__name__, filename))
            json.dump(obj, f)

        return obj

    return wrapper
