import re

from nltk.stem.snowball import EnglishStemmer
from nltk.stem.wordnet import WordNetLemmatizer
import nltk
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer, CountVectorizer

nltk.download('wordnet')


def text_prep(texts, stem=False, lemmatize=True, max_df=0.5, min_df=10, ngram_range=(1, 3)):
    # Stem
    if stem:
        stemmer = EnglishStemmer()
        texts = [stemmer.stem(text) for text in texts]
    if lemmatize:
        lemm = WordNetLemmatizer()
        texts = [" ".join(lemm.lemmatize(word)
                          for word in re.split(r"[\s,.:;$]", text) if len(word) > 0)
                 for text in texts]

    # Tokenize with inverse term frequency
    tfidvec = TfidfVectorizer(max_df=max_df, min_df=min_df, stop_words="english", ngram_range=ngram_range)
    tfidarr = tfidvec.fit_transform(texts)

    # Tokenize with counts
    countvec = CountVectorizer(max_df=max_df, min_df=min_df, stop_words="english", ngram_range=ngram_range)
    countarr = countvec.fit_transform(texts)

    return tfidvec, tfidarr, countvec, countarr
