import sklearn.decomposition as decomposition
import sklearn.manifold as manifold
from sklearn.preprocessing import Normalizer, StandardScaler


def _str_to_bool(inp):
    return inp == "True"


class DimReduction:
    _default_options = {}
    info_link = None

    def apply(self, df):
        raise NotImplemented

    @classmethod
    def get_options(cls):
        return cls._default_options


class PCA(DimReduction):
    _default_options = {'n_components': 3, 'scale_before': False, 'scale_after': False}
    info_link = "https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html"

    def __init__(self, n_components, scale_before=False, scale_after=False):
        self.options = {"n_components": int(n_components),
                        'scale_before': _str_to_bool(scale_before), 'scale_after': _str_to_bool(scale_after)}

    def apply(self, df):
        arr = StandardScaler().fit_transform(df.values) if self.options["scale_before"] else df.values
        arr = decomposition.PCA(**{k: v for k, v in self.options.items() if not k.startswith("scale_")}
                                ).fit_transform(arr)
        arr = StandardScaler().fit_transform(arr) if self.options["scale_after"] else arr
        return arr


class NMF(DimReduction):
    _default_options = {'n_components': 3, 'max_iter': 200, 'scale_before': False, 'scale_after': False}
    info_link = "https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html"

    def __init__(self, n_components=3, max_iter=200, scale_before=False, scale_after=False):
        self.options = {'n_components': int(n_components), 'max_iter': int(max_iter),
                        'scale_before': _str_to_bool(scale_before), 'scale_after': _str_to_bool(scale_after)}

    def apply(self, df):
        arr = StandardScaler().fit_transform(df.values) if self.options["scale_before"] else df.values
        arr = decomposition.NMF(**{k: v for k, v in self.options.items() if not k.startswith("scale_")}
                                ).fit_transform(arr)
        arr = StandardScaler().fit_transform(arr) if self.options["scale_after"] else arr
        return arr


class SVD(DimReduction):
    _default_options = {'n_components': 3, 'n_iter': 5, 'scale_before': False, 'scale_after': False}
    info_link = "https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html"

    def __init__(self, n_components=3, n_iter=5, scale_before=False, scale_after=False):
        self.options = {'n_components': int(n_components), 'n_iter': int(n_iter),
                        'scale_before': _str_to_bool(scale_before), 'scale_after': _str_to_bool(scale_after)}

    def apply(self, df):
        arr = StandardScaler().fit_transform(df.values) if self.options["scale_before"] else df.values
        arr = decomposition.TruncatedSVD(**{k: v for k, v in self.options.items() if not k.startswith("scale_")}
                                         ).fit_transform(arr)
        arr = StandardScaler().fit_transform(arr) if self.options["scale_after"] else arr
        return arr


class FactorAnalysis(DimReduction):
    _default_options = {'n_components': 3, 'max_iter': 1000, 'scale_before': False, 'scale_after': False}
    info_link = "https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.FactorAnalysis.html"

    def __init__(self, n_components=3, max_iter=5, scale_before=False, scale_after=False):
        self.options = {'n_components': int(n_components), 'max_iter': int(max_iter),
                        'scale_before': _str_to_bool(scale_before), 'scale_after': _str_to_bool(scale_after)}

    def apply(self, df):
        arr = StandardScaler().fit_transform(df.values) if self.options["scale_before"] else df.values
        arr = decomposition.FactorAnalysis(**{k: v for k, v in self.options.items() if not k.startswith("scale_")}
                                           ).fit_transform(arr)
        arr = StandardScaler().fit_transform(arr) if self.options["scale_after"] else arr
        return arr


class TSNE(DimReduction):
    _default_options = {'n_components': 3, 'perplexity': 30, 'learning_rate': 200, 'n_iter': 1000,
                        'scale_before': False, 'scale_after': False}
    info_link = "https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html"

    def __init__(self, n_components=3, perplexity=30, learning_rate=200, n_iter=1000,
                 scale_before=False, scale_after=False):
        self.options = {'n_components': int(n_components), 'perplexity': int(perplexity),
                        'learning_rate': float(learning_rate), 'n_iter': int(n_iter),
                        'scale_before': _str_to_bool(scale_before), 'scale_after': _str_to_bool(scale_after)}

    def apply(self, df):
        arr = StandardScaler().fit_transform(df.values) if self.options["scale_before"] else df.values
        arr = manifold.TSNE(**{k: v for k, v in self.options.items() if not k.startswith("scale_")}).fit_transform(arr)
        arr = StandardScaler().fit_transform(arr) if self.options["scale_after"] else arr
        return arr

