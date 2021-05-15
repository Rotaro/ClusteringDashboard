import logging
import time
import os
from collections import namedtuple

import data.wikipedia as wikipedia
import data.imdb as imdb

import pandas as pd


_columns = ["title", "url", "summary"]
TVSeries = namedtuple("TVSeries", _columns)

_path = os.path.dirname(__file__)


def get_all_wikipedia_summaries(filename="wikipedia_tv_series_summary.csv"):
    if filename in os.listdir(_path):
        df = pd.read_csv(_path + "/" + filename)
    else:
        df = pd.DataFrame([], columns=_columns)

    title_url = wikipedia.get_wikipedia_tv_series_urls("wikipedia_tv_series_urls.json")
    title_url = [(title, url) for title, url in title_url if title not in set(df.title.values)]
    if len(title_url) == 0:
        return df

    t_start = time.time()
    chunk_size = 250
    for i in range(0, (len(title_url) // chunk_size + 1) * chunk_size, chunk_size):
        t_start_chunk = time.time()
        logging.info("Retrieving %d wikipedia summaries for tv series (%d / %d).", chunk_size, i, len(title_url))
        name_url_summary = []
        for name, url in title_url[i:i + chunk_size]:
            name_url_summary.append([
                name, url, wikipedia.get_wikipedia_summary(url)
            ])
        df = pd.concat([df, pd.DataFrame(name_url_summary, columns=_columns)], axis=0)
        df.to_csv(_path + "/" + filename, index=False)

        logging.info("Retrieved %d summaries in %.1f sec.", chunk_size, time.time() - t_start_chunk)

        time.sleep(0.5)

    logging.info("Retrieval of summaries complete - %d summaries in %.1f sec.", len(title_url), time.time() - t_start)

    return df


def get_imdb_top_250_wikipedia_summaries(top_tv_series_filename="top_tv_series.json",
                                         wikipedia_summaries_filename="wikipedia_tv_series_summary.csv"):
    """Gets data for top 250 tv series on imdb.

    :param top_tv_series_filename: str, filename for saving top imdb tv series.
    :param wikipedia_summaries_filename: str, filename for saving wikipedia summmaries.
    """
    wikipedia_summaries = get_all_wikipedia_summaries(wikipedia_summaries_filename)

    top_tv_series_titles = {title[0] for url, title in imdb.get_top_tv_series(top_tv_series_filename).items()}
    return wikipedia_summaries[wikipedia_summaries.title.isin(top_tv_series_titles)]

