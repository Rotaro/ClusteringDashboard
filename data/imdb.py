import re
import logging
import time
import os

import json

from bs4 import BeautifulSoup
import requests

logging.getLogger().setLevel(logging.INFO)
_path = os.path.dirname(__file__)


def get_top_tv_series(filename="top_tv_series.json"):
    if filename in os.listdir(_path):
        with open(_path + "\\" + filename, "r") as f:
            return json.load(f)

    t_start = time.time()
    start_url = "http://www.imdb.com/chart/toptv/"
    logging.info("%s - Retrieving top 250 tv series.", start_url)
    parsed = BeautifulSoup(requests.get(start_url).text, 'html.parser')

    link_container = parsed.find("tbody", attrs={"class": "lister-list"})
    tv_series = {}
    for tr in link_container.find_all("td", attrs={"class": "titleColumn"}):
        try:
            a = tr.find("a")
            url = re.sub(r"(/title/tt\d+/).+", r"\1", a["href"])
            title = a.text

            tv_series[url] = (title, )

        except KeyError:
            continue

    logging.info("%s - Tv series retrieved - %d entries - %.2f sec duration.", start_url, len(tv_series),
                 time.time() - t_start)

    with open(_path + "\\" + filename, "w") as f:
        json.dump(tv_series, f)

    return tv_series


def get_single_tv_series_data(link):
    # Data from IMDB
    url = "https://www.imdb.com/{link}".format(link=link)
    page_text = requests.get(url).text
    parsed = BeautifulSoup(page_text, 'html.parser')

    # Titles
    org_title_element = parsed.find("div", {"class": "originalTitle"})
    org_title = re.sub(r"(\s)\s+", r"\1", org_title_element.text).strip().replace(" (original title)", "") \
        if org_title_element else None
    title = parsed.find("div", {"class": "title_wrapper"}).h1.text.strip()

    # Addtional imdb information
    year = re.sub(r".*?\((\d+).*?\)\s*", r"\1",
                  parsed.find("a", attrs={"title": "See more release dates"}).text)
    summary = re.sub(r"(\s)\s+", r"\1", parsed.find("div", attrs={"class": "summary_text"}).text).strip()
    imdb_tags = ",".join([
        a.text.strip()
        for div_element in parsed.find_all("div", {"class": "see-more inline canwrap"})
        if div_element.h4.text.strip() == "Genres:"
        for a in div_element.find_all("a")
    ])

    replacements = (
        ("\r\n", " "),
        ("\r", " "),
        ("\n", " "),
        ("\"", ""),
        (" ... See full summary\xa0»", ""),
        (" (original title)", ""),
    )

    for m, repl in replacements:
        summary = summary.replace(m, repl)
        title = title.replace(m, repl)

    return {"summary": summary, "imdb_tags": imdb_tags, "year": year,
            "org_title": org_title or title}
