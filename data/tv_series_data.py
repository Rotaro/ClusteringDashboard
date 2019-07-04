import logging
import time
import os
import pickle

import pandas as pd

import data.wikipedia as wikipedia
import data.imdb as imdb

_local_cache = {}
_columns = ["id", "title", "org_title", "year", "imdb_tags", "summary", "wikipedia_summary"]


def _get_set_local_cache(key, value_func, set_value=True, pickle_value=True):
    pickle_key = key.replace("\\", ".").replace("/", ".") + ".pickle"

    if pickle_value and pickle_key in os.listdir():
        with open(pickle_key, "rb") as f:
            value = pickle.load(f)

        if set_value and key not in _local_cache:
            _local_cache[key] = value

    if key not in _local_cache:
        value = value_func()
        if set_value:
            _local_cache[key] = value
        if pickle_value:
            with open(pickle_key, "wb") as f:
                pickle.dump(value, f)

    return _local_cache[key]


def _get_tv_series_data(tv_series):
    logging.info("Starting retrieval of %d tv series.", len(tv_series))
    t_start = time.time()

    tv_series_data = []
    for link, (title,) in tv_series.items():
        try:
            series_data = imdb.get_single_tv_series_data(link)

            # Get wikipedia summary
            title, year = series_data.get("org_title", title), series_data["year"]
            wikipedia_summary = wikipedia.get_wikipedia_summary("%s %s tv series" % (title, year))

            d = {"id": link, "title": title, **series_data, "wikipedia_summary": wikipedia_summary}
            tv_series_data.append([*[d[col] for col in _columns]])
        except (KeyError, TypeError):
            print("Problem with link %s, skipping." % link)

        time.sleep(0.1)

    logging.info("TV series retrieved (%d / %d) - %.2f sec duration.",
                 len(tv_series_data), len(tv_series), time.time() - t_start)

    return tv_series_data


def _load_tv_series_csv(filename):
    try:
        return pd.read_csv(filename, sep="\t", encoding="utf-8")
    except FileNotFoundError:
        return pd.DataFrame(columns=_columns)


def _save_tv_series_csv(df, filename):
    df.to_csv(filename, sep="\t", encoding="utf-8", index=False)


def get(filename=None):
    """Gets data for top 250 tv series on imdb.

    :param filename: str, optional filename for saving results.
    """
    tv_series_data = _get_set_local_cache("tv_series_data_%s" % filename, lambda: _load_tv_series_csv(filename))

    top_tv_series = _get_set_local_cache("top_tv_series", imdb.get_top_tv_series)
    top_tv_series_list = [(url, v) for url, v in top_tv_series.items() if tv_series_data.id.isin([url]).sum() == 0]

    if len(top_tv_series_list) == 0:
        return tv_series_data

    chunk_size = 50
    for i in range(0, (len(top_tv_series_list) // chunk_size + 1) * chunk_size, chunk_size):
        data = _get_tv_series_data(dict(top_tv_series_list[i: i + chunk_size]))
        tv_series_data = tv_series_data.append(pd.DataFrame(data, columns=_columns))

    if filename is not None:
        _save_tv_series_csv(tv_series_data, filename)
    tv_series_data = _get_set_local_cache("tv_series_data_%s" % filename, lambda: _load_tv_series_csv(filename))

    return tv_series_data


if __name__ == "__main__":
    filename = "tv_series_data.csv"
    df = get(filename)
