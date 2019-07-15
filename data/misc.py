import os
import json
import logging

_path = os.path.dirname(__file__)


def save_load_results_to_json(func, print=True):
    """Saves and loads results of function to json."""
    def wrapper(filename, *args, **kwargs):
        if filename in os.listdir(_path):
            if print:
                logging.info("Loading results for %s from %s." % (func.__name__, filename))
            with open(_path + "\\" + filename, "r") as f:
                return json.load(f)

        obj = func(*args, **kwargs)

        with open(_path + "\\" + filename, "w") as f:
            if print:
                logging.info("Saving results for %s from %s." % (func.__name__, filename))
            json.dump(obj, f)

        return obj

    return wrapper
