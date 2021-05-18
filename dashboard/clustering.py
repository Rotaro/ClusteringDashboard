import dash_core_components as dcc
import dash_html_components as html

import numpy as np
import pandas as pd

import model

import dashboard.app_misc as misc
from dashboard.cache import cache


clusterings = misc.DropdownWithOptions(
    header="Choose clustering algorithm:", dropdown_id="clustering", dropdown_objects={
        "KMeans": model.clustering.KMeans,
        "DBSCAN": model.clustering.DBSCAN,
        "AgglomerativeClustering": model.clustering.AgglomerativeClustering,
        "SpectralClustering": model.clustering.SpectralClustering,
        "GaussianMixture": model.clustering.GaussianMixture,
        "LDA": model.clustering.LDA,
    }, include_refresh_button=True
)


@cache.memoize()
def get_clusters(to_cluster, clustering, clustering_options):
    if clustering_options:
        clusters = clusterings.apply(clustering, clustering_options, to_cluster)
    else:
        clusters = np.zeros(to_cluster.shape[0])

    return clusters


def get_cluster_info_df(n_cluster_info, clusters, titles, bow_data_df):
    cluster_info = []

    for i, cluster in enumerate(np.unique(clusters)):
        idx = clusters == cluster
        if idx.sum() == 0:
            continue
        # Collect cluster information
        cluster_info.append([
            int(cluster), idx.sum(),
            *np.pad(titles.loc[idx].sample(min(n_cluster_info, idx.sum()), replace=False).values,
                    (0, max(0, n_cluster_info - idx.sum())), 'constant'),
            *bow_data_df.columns[bow_data_df.loc[idx].sum(0).argsort()[::-1][:n_cluster_info]]
        ])

    n_cluster_info = int(max([len(row) for row in cluster_info]) - 2) // 2

    cluster_info_df = pd.DataFrame(cluster_info, columns=[
        "Cluster", "Size",
        *["Sample%d" % i for i in range(1, n_cluster_info + 1)],
        *["Top Word %d" % i for i in range(1, n_cluster_info + 1)],
    ])

    return cluster_info_df


clustering_tab = dcc.Tab(
    label="Clustering", children=[
        html.Div(id="clustering_area", children=[
            clusterings.generate_dash_element(),
        ]),
        html.P(children=None, id="cluster_info_text", style={"padding": "5px", "margin": "5px"})
    ], className="custom-tab", selected_className="custom-tab--selected"
)

clusters_tab = dcc.Tab(
    label="Clusters", children=[html.Div(id="cluster_info_table")],
    className="custom-tab", selected_className="custom-tab--selected"
)
