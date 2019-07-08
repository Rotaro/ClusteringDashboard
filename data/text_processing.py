import re
import logging

import pandas as pd

from nltk.stem.snowball import EnglishStemmer
from nltk.stem.wordnet import WordNetLemmatizer
import nltk
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer, CountVectorizer

import dash_core_components as dcc

nltk.download('wordnet')


class TextAction:
    _default_options = {}
    include_checkbox = True

    def apply(self, df):
        raise NotImplemented

    @classmethod
    def get_options(cls):
        return cls._default_options

    @classmethod
    def parse_dash_elements(cls, elements):
        name, active, options = None, True, {}
        for e in elements:
            if not isinstance(e, dict):
                continue
            id, value = e["props"]["id"], e["props"]["value"]

            if "|" not in id:
                name = id
                active = value
            else:
                options[id.split("|")[1]] = eval(value)
                name = name or id.split("|")[0]

        return name, active, options

    def get_dash_element_ids(self):
        name = self.__class__.__name__
        return [
            inp
            for inp in [*([name] if self.include_checkbox else []),
                        *['%s|%s' % (name, opt) for opt in self.get_options().keys()]]
        ]

    def to_dash_elements(self):
        name = self.__class__.__name__

        options = [
            ("%s: " % option_name,
             dcc.Input(id="%s|%s" % (name, option_name), type="text", value=str(default_value)))
            for option_name, default_value in self.get_options().items()
        ]

        if self.include_checkbox:
            checkbox = dcc.Checklist(id=name, options=[{'label': name, 'value': name}], value=[],
                                     style={'padding': '5px', 'margin': '5px'})

            logging.info([checkbox, *[e for a_b in options for e in a_b]])
            return [checkbox, *[e for a_b in options for e in a_b]]

        return [e for a_b in options for e in a_b]


class Stem(TextAction):
    def apply(self, df):
        stemmer = EnglishStemmer()
        return pd.DataFrame([stemmer.stem(text) for text in df.values.ravel()], columns=df.columns)


class Lemmatize(TextAction):
    def apply(self, df):
        lemm = WordNetLemmatizer()

        def _lemmatize(text):
            return " ".join(lemm.lemmatize(word) for word in re.split(r"[\s,.:;$]", text) if len(word) > 0)

        return pd.DataFrame([_lemmatize(text) for text in df.values.ravel()], columns=df.columns)


class TFIDF(TextAction):
    _default_options = {
        "max_df": 0.5, "min_df": 10, "ngram_range": (1, 3)
    }
    include_checkbox = False

    def __init__(self, max_df=0.5, min_df=10, ngram_range=(1, 3)):
        self.options = {"max_df": float(max_df), "min_df": int(min_df),
                        "ngram_range": tuple(int(x) for x in ngram_range)}

    def get_vec_arr(self, df):
        # Tokenize with inverse term frequency
        countvec = TfidfVectorizer(stop_words="english", **self.options)
        countarr = countvec.fit_transform(df.values.ravel())

        return countvec, countarr

    def apply(self, df):
        vec, arr = self.get_vec_arr(df)
        return pd.DataFrame(arr.toarray(), columns=vec.get_feature_names())


class BOW(TextAction):
    _default_options = {
        "max_df": 0.5, "min_df": 10, "ngram_range": (1, 3)
    }
    include_checkbox = False

    def __init__(self, max_df=0.5, min_df=10, ngram_range=(1, 3)):
        self.options = {"max_df": float(max_df), "min_df": int(min_df),
                        "ngram_range": tuple(int(x) for x in ngram_range)}

    def get_vec_arr(self, df):
        # Tokenize with inverse term frequency
        countvec = CountVectorizer(stop_words="english", **self.options)
        countarr = countvec.fit_transform(df.values.ravel())

        return countvec, countarr

    def apply(self, df):
        vec, arr = self.get_vec_arr(df)
        return pd.DataFrame(arr.toarray(), columns=vec.get_feature_names())
