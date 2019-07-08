import numpy as np

import sklearn.cluster
import sklearn.mixture

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.preprocessing import Normalizer


class Clustering:
    _default_options = {}

    def apply(self, df):
        raise NotImplemented

    @classmethod
    def get_options(cls):
        return cls._default_options


class KMeans(Clustering):
    _default_options = {'n_clusters': 8, 'n_init': 10}

    def __init__(self, n_clusters, n_init):
        self.options = {"n_clusters": int(n_clusters), 'n_init': int(n_init)}

    def apply(self, df):
        arr = Normalizer().fit_transform(df.values)
        return sklearn.cluster.KMeans(**self.options).fit_predict(arr)


class DBSCAN(Clustering):
    _default_options = {'eps': 0.5, 'min_samples': 5}

    def __init__(self, eps, min_samples):
        self.options = {"eps": float(eps), 'min_samples': int(min_samples)}

    def apply(self, df):
        # arr = Normalizer().fit_transform(df.values)
        clusters = sklearn.cluster.DBSCAN(**self.options).fit_predict(df.values)
        print(clusters)
        return clusters


class AgglomerativeClustering(Clustering):
    _default_options = {'n_clusters': 8, 'affinity': 'euclidean', 'linkage': 'ward'}

    def __init__(self, n_clusters, affinity, linkage):
        self.options = {"n_clusters": int(n_clusters), 'affinity': affinity, 'linkage': linkage}

    def apply(self, df):
        arr = Normalizer().fit_transform(df.values)
        return sklearn.cluster.AgglomerativeClustering(**self.options).fit_predict(arr)


class GaussianMixture(Clustering):
    _default_options = {'n_components': 8, 'covariance_type': 'full'}

    def __init__(self, n_components, covariance_type):
        self.options = {"n_components": int(n_components), 'covariance_type': covariance_type}

    def apply(self, df):
        arr = Normalizer().fit_transform(df.values)
        return sklearn.mixture.GaussianMixture(**self.options).fit_predict(arr)


class LDA(Clustering):
    _default_options = {'n_components': 8}

    def __init__(self, n_components):
        self.options = {'n_components': int(n_components)}

    def apply(self, df):
        lda = LatentDirichletAllocation(**self.options)
        lda_res = lda.fit_transform(df.values)
        return np.argmax(lda_res, axis=1)
