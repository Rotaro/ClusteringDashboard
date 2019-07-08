import re
import pandas as pd

from nltk.stem.snowball import EnglishStemmer
from nltk.stem.wordnet import WordNetLemmatizer
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

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
