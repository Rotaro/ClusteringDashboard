import time
import re
import logging

from bs4 import BeautifulSoup
import requests
from urllib.parse import unquote

from data.misc import save_load_results_to_json


def _get_google_wikipedia_link(name):
    """Get wikipedia link using google search..."""
    if name is None or len(name) == 0:
        return
    url = "https://www.google.com/search?q={search_term}+wikipedia".format(search_term=name.replace(" ", "+"))
    page_text = requests.get(url).text
    links = [link for link in re.findall(r"a href=\"/url\?q=([^\"]*?)&amp", page_text) if "en.wikipedia" in link]

    return unquote(unquote(links[0])) if links else None


def get_wikipedia_summary(wikipedia_url):
    """Get wikipedia article summary. Summary = text until table of content."""
    wikipedia_page = BeautifulSoup(requests.get(wikipedia_url).text, 'html.parser')
    intro_ele = wikipedia_page.find("div", {"class": "mw-parser-output"})
    if intro_ele is None:
        logging.info("Parsing wikipedia page of %s failed.", wikipedia_url)
        return

    summary_text = ""
    for child in intro_ele.findChildren():
        # All relevant text is inside p-elements
        if child.name == "p":
            summary_text += child.text

        # Stop once table of contents is reached
        if child.name == "div" and child.attrs.get("id") == "toc":
            break

    return summary_text.replace("\n", "").replace("—", " ") if len(summary_text) > 0 else None


@save_load_results_to_json
def get_wikipedia_tv_series_urls():
    """"Retrieves all television urls found in wikipedia list of television programs."""
    next_list_url = "https://en.wikipedia.org/wiki/List_of_television_programs:_numbers"
    black_listed = ["/wiki/Help", "/wiki/Special", "/wiki/Main_Page", "/wiki/Category", "/wiki/List",
                    "/w/index.php", "/wiki/Wikipedia", "/wiki/Portal"]

    urls = []
    while next_list_url:
        list_page = BeautifulSoup(requests.get(next_list_url).text, 'html.parser')

        has_next = list_page.find("b", text="Next:")
        next_list_url = "https://en.wikipedia.org/%s" % has_next.find_next("a")["href"] if has_next else None

        next_link = list_page.find("span", attrs={"class": "mw-headline"}).find_next("a")
        i = 0
        while next_link:
            url = next_link["href"]
            if url and url.startswith("/wiki/") and not any([url.startswith(e) for e in black_listed]):
                urls.append((next_link.text, "https://en.wikipedia.org/%s" % url))

            next_link = next_link.find_next("a")
            i += 1

        logging.info("Found %d wikipedia tv series listed in %s." % (i, next_list_url))
        time.sleep(2.0)

    return urls
