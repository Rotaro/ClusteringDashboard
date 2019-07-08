import sklearn.decomposition as decomposition
import sklearn.manifold as manifold
from sklearn.preprocessing import Normalizer


class DimReduction:
    _default_options = {}

    def apply(self, df):
        raise NotImplemented

    @classmethod
    def get_options(cls):
        return cls._default_options


class PCA(DimReduction):
    _default_options = {'n_components': 3}

    def __init__(self, n_components):
        self.options = {"n_components": int(n_components)}

    def apply(self, df):
        arr = Normalizer().fit_transform(df.values)
        return decomposition.PCA(**self.options).fit_transform(arr)


class NMF(DimReduction):
    _default_options = {'n_components': 3, 'max_iter': 200}

    def __init__(self, n_components=3, max_iter=200):
        self.options = {'n_components': int(n_components), 'max_iter': int(max_iter)}

    def apply(self, df):
        return decomposition.NMF(**self.options).fit_transform(df.values)


class SVD(DimReduction):
    _default_options = {'n_components': 3, 'n_iter': 5}

    def __init__(self, n_components=3, n_iter=5):
        self.options = {'n_components': int(n_components), 'n_iter': int(n_iter)}

    def apply(self, df):
        arr = Normalizer().fit_transform(df.values)
        return decomposition.TruncatedSVD(**self.options).fit_transform(arr)


class TSNE(DimReduction):
    _default_options = {'n_components': 3, 'perplexity': 30, 'learning_rate': 200, 'n_iter': 1000}

    def __init__(self, n_components=3, perplexity=30, learning_rate=200, n_iter=1000):
        self.options = {'n_components': int(n_components), 'perplexity': int(perplexity),
                        'learning_rate': float(learning_rate), 'n_iter': int(n_iter)}

    def apply(self, df):
        arr = Normalizer().fit_transform(df.values)
        return manifold.TSNE(**self.options).fit_transform(arr)
