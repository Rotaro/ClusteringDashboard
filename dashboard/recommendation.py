import dash_core_components as dcc
import dash_html_components as html

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import pairwise_distances

import model

import dashboard.app_misc as misc
from dashboard.cache import cache

from dashboard.data_dim_reduction import get_dim_reduction


def get_recommendations(n, recommend_for, recommendation_metric, data_df, clusters, titles):
    recommend_for_idx = np.argwhere(titles.values == recommend_for).ravel()[0]
    recommend_for_cluster = clusters[recommend_for_idx]
    dists = pairwise_distances(data_df.values[recommend_for_idx, :][None, :], data_df.values,
                               metric=recommendation_metric)
    dists_in_cluster = pairwise_distances(data_df.values[recommend_for_idx, :][None, :],
                                          data_df.values[clusters == recommend_for_cluster],
                                          metric=recommendation_metric)

    top = np.argsort(dists).ravel()[:n]
    top_cluster = np.argsort(dists_in_cluster).ravel()[:n]

    return pd.DataFrame(
        {"Top Recommendations": titles.values[top],
         "Top Recommendations Score": dists.ravel()[top],
         "Top Recommendations in Cluster":
             np.pad(titles.values[clusters == recommend_for_cluster][top_cluster],
                    (0, max(n - len(top_cluster), 0)), 'constant'),
         "Top Recommendations in Cluster Score":
             np.pad(dists_in_cluster.ravel()[top_cluster], (0, max(n - len(top_cluster), 0)), 'constant')
         }
    )


recommendation_tab = dcc.Tab(
    label="Recommendation", children=[
        html.Div(id="recommendation_area", children=[
            html.P("Pairwise Distance:", style={"padding": "5px"}),
            dcc.Dropdown(id="recommendation_metric", options=[
                {"label": name, "value": name} for name in ("cosine", "euclidean", "manhattan")
            ], value="cosine"),
            html.P("Recommend for:", style={"padding": "5px"}),
            dcc.Dropdown(id="recommendation_picker"),
            html.P("Recommendations:", style={"padding": "5px"}),
            html.Div(id="recommendations")
        ]),
    ], className="custom-tab", selected_className="custom-tab--selected"
)