import re
import logging

from bs4 import BeautifulSoup
import requests
from urllib.parse import unquote


def _get_wikipedia_link(name):
    """Get wikipedia link using google search..."""
    url = "https://www.google.com/search?q={search_term}+wikipedia".format(search_term=name.replace(" ", "+"))
    page_text = requests.get(url).text
    links = [link for link in re.findall("a href=\"/url\?q=([^\"]*?)\&amp", page_text) if "en.wikipedia" in link]

    return unquote(unquote(links[0])) if links else None


def get_wikipedia_summary(name):
    """Use google to find wikipedia article summary."""
    if name is None or len(name) == 0:
        return

    wikipedia_link = _get_wikipedia_link(name)
    if wikipedia_link is None:
        logging.info("No wikipedia for %s", name)
        return

    logging.info("%s - %s", name, wikipedia_link)

    wikipedia_page = BeautifulSoup(requests.get(wikipedia_link).text, 'html.parser')
    intro_ele = wikipedia_page.find("div", {"class": "mw-parser-output"})
    if intro_ele is None:
        logging.info("Parsing wikipedia page of %s failed.", name)
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
