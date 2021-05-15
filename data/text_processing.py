import re
import pandas as pd
from tempfile import NamedTemporaryFile
import os

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

import nltk
from nltk.stem.snowball import EnglishStemmer
from nltk.stem.wordnet import WordNetLemmatizer

try:
    import fasttext
except ImportError as e:
    fasttext = None


_path = os.path.dirname(__file__)
nltk.download('wordnet')


class TextAction:
    _default_options = {}
    info_link = None

    def apply(self, df):
        raise NotImplemented

    @classmethod
    def get_options(cls):
        return cls._default_options


class WikipediaTextCleanup(TextAction):
    def _clean_wikipedia_text(self, text):
        # Reference links
        text = re.sub(r"\[\d+\]", "", text)
        # Double whitespaces
        text = re.sub(r"(\s)\s", r"\1", text)

        return text

    def apply(self, df):
        return pd.DataFrame([self._clean_wikipedia_text(text) for text in df.values.ravel()], columns=df.columns)


class Stem(TextAction):
    info_link = "https://www.nltk.org/_modules/nltk/stem/snowball.html"

    def apply(self, df):
        stemmer = EnglishStemmer()
        return pd.DataFrame([stemmer.stem(text) for text in df.values.ravel()], columns=df.columns)


class Lemmatize(TextAction):
    info_link = "https://www.nltk.org/_modules/nltk/stem/wordnet.html"

    def apply(self, df):
        lemm = WordNetLemmatizer()

        def _lemmatize(text):
            return " ".join(lemm.lemmatize(word) for word in re.split(r"[\s,.:;$]", text) if len(word) > 0)

        return pd.DataFrame([_lemmatize(text) for text in df.values.ravel()], columns=df.columns)


class TFIDF(TextAction):
    _default_options = {
        "max_df": 0.8, "min_df": 50, "ngram_range": (1, 2)
    }
    info_link = "https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html"

    def __init__(self, max_df=0.8, min_df=50, ngram_range=(1, 2)):
        self.options = {"max_df": float(max_df), "min_df": int(min_df),
                        "ngram_range": tuple(int(x) for x in ngram_range)}

    def get_vec_arr(self, df):
        min_df = min(self.options["min_df"], int(self.options["max_df"] * len(df)) - 1)
        countvec = TfidfVectorizer(stop_words="english",  **{**self.options, "min_df": min_df})
        countarr = countvec.fit_transform(df.values.ravel())

        return countvec, countarr

    def apply(self, df):
        vec, arr = self.get_vec_arr(df)
        return pd.DataFrame(arr.toarray(), columns=vec.get_feature_names())


class BOW(TextAction):
    _default_options = {
        "max_df": 0.8, "min_df": 50, "ngram_range": (1, 2)
    }
    info_link = "https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html"

    def __init__(self, max_df=0.8, min_df=50, ngram_range=(1, 2)):
        self.options = {"max_df": float(max_df), "min_df": int(min_df),
                        "ngram_range": tuple(int(x) for x in ngram_range)}

    def get_vec_arr(self, df):
        # Tokenize with inverse term frequency
        min_df = min(self.options["min_df"], int(self.options["max_df"] * len(df)) -1)
        countvec = CountVectorizer(stop_words="english", **{**self.options, "min_df": min_df})
        countarr = countvec.fit_transform(df.values.ravel())

        return countvec, countarr

    def apply(self, df):
        vec, arr = self.get_vec_arr(df)
        return pd.DataFrame(arr.toarray(), columns=vec.get_feature_names())


class FastText(TextAction):
    _default_options = {
        "model": "skipgram", "dim": 25, "epoch": 5
    }
    info_link = "https://fasttext.cc/docs/en/python-module.html#api"

    def __init__(self, model, dim, epoch):
        self.options = {"model": model, "dim": int(dim), "epoch": int(epoch)}

    def _train_model(self, df):
        model, dim, epoch = (self.options[opt] for opt in ["model", "dim", "epoch"])
        text = "\n".join(map(lambda x: x[0], df.values.tolist()))

        with NamedTemporaryFile("w", encoding="utf-8", delete=False) as f:
            f.write(text)

        m = fasttext.train_unsupervised(f.name, model=model, dim=dim, epoch=epoch)
        os.remove(f.name)

        return m

    def apply(self, df):
        model = self._train_model(df)
        dim = self.options["dim"]
        arr = np.array([model.get_sentence_vector(row[0]) for row in df.values])

        return pd.DataFrame(arr, columns=["Dim%d" % i for i in range(dim)])


class FastTextPretrained(TextAction):
    """This requires manually constructed word: vector dictionary saved with pickle."""

    _default_options = {
        "max_df": 0.5, "min_df": 4, "ngram_range": (1, 1)
    }
    info_link = "https://fasttext.cc/docs/en/python-module.html#api"

    def __init__(self, max_df=0.5, min_df=4, ngram_range=(1, 3)):
        self.options = {"max_df": float(max_df), "min_df": int(min_df),
                        "ngram_range": tuple(int(x) for x in ngram_range)}

    @classmethod
    def has_pretrained(cls):
        return "fasttext_pretrained.pickle" in os.listdir(_path)

    @classmethod
    def _get_pretrained(cls):
        import pickle
        full_path = os.path.join(_path, "fasttext_pretrained.pickle")
        with open(full_path, "rb") as f:
            return pickle.loads(f.read())

    def apply(self, df):
        bow_df = BOW(**self.options).apply(df)
        word_to_vec = self._get_pretrained()

        avg_vectors = []
        for i, row in bow_df.iterrows():
            words = bow_df.columns[row > 0]
            series_vector = [word_to_vec[word] for word in words if word in word_to_vec]

            if len(series_vector) == 0:
                print("0 series vector...")
                series_vector = np.zeros((1, 300))

            avg_vectors.append(np.mean(series_vector, axis=0))

        return pd.DataFrame(avg_vectors, columns=["Dim%d" % i for i in range(len(avg_vectors[0]))])


def join_columns(df, chosen_cols):
    series = df[chosen_cols[0]].astype(str)
    for col in chosen_cols[1:]:
        series = series + ". " + df[col].astype(str)

    df = pd.DataFrame(series.values, columns=["text_to_cluster"])

    return df
