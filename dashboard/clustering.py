import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Output

import numpy as np
import pandas as pd
from sklearn.metrics.cluster import silhouette_score

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
    elif to_cluster is not None:
        clusters = np.zeros(to_cluster.shape[0])
    else:
        clusters = None

    return clusters


def get_cluster_info_df(n_cluster_info, clusters, titles, bow):
    """Returns dataframe with basic information regarding clusters.

    Will contain columns with cluster number, cluster size, random samples and top words.
    """
    cluster_info_rows = []

    for i, cluster in enumerate(np.unique(clusters)):
        idx = clusters == cluster
        if idx.sum() == 0:
            continue

        # Find n_cluster_info samples
        samples = titles.loc[idx].sample(min(n_cluster_info, idx.sum()), replace=False).values
        samples = np.pad(samples, (0, max(0, n_cluster_info - len(samples))))

        # Find n_cluster_info top words
        top_words = bow.columns[bow.loc[idx].sum(0).argsort()[::-1][:n_cluster_info]]
        top_words = np.pad(top_words, (0, max(0, n_cluster_info - len(top_words))))

        cluster_info_rows.append([int(cluster), idx.sum(), *samples, *top_words])

    cluster_info_df = pd.DataFrame(cluster_info_rows, columns=[
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


def get_clustering_output(df_arr, clustering_method, clustering_options, titles, bow):
    clusters = get_clusters(df_arr, clustering_method, clustering_options)

    cluster_info_df = None
    if clusters is not None and titles is not None and bow is not None:
        cluster_info_df = get_cluster_info_df(10, clusters, titles, bow)

    cluster_info_score = None
    if np.unique(clusters).size > 1:
        cluster_info_score = "Silhouette Score: %.2f" % silhouette_score(df_arr.values, clusters)

    return misc.generate_datatable(cluster_info_df, "cluster_info", 1000, "600px"), cluster_info_score


arguments = {
    "clustering_method": clusterings._dropdown_args,
    "clustering_options": clusterings._options_args,
    "clustering_refresh": clusterings._refresh_args,
    "cluster_info_table": ("cluster_info_table", "children"),
}
outputs = [Output("cluster_info_table", "children"), Output("cluster_info_text", "children")]
