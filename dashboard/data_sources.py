from data.tv_series_data import get_imdb_top_250_wikipedia_summaries
from data.tv_series_data import get_all_wikipedia_summaries

from dashboard.cache import cache

data_sources = {
    "Top 250 IMDB TV Series": get_imdb_top_250_wikipedia_summaries,
    "Wikipedia TV Series Summaries": get_all_wikipedia_summaries
}


@cache.memoize()
def get_data_source(data_name, sample_percent=100):
    df = data_sources[data_name]() if data_name is not None else None

    if df is not None and "org_title" not in df.columns:
        df["org_title"] = df["title"]

    if df is not None and sample_percent < 100:
        df = df.sample(frac=sample_percent / 100)

    return df
