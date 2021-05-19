import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Output

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import pairwise_distances

import dashboard.app_misc as misc


recommendation_tab = dcc.Tab(
    label="Recommendation", children=[
        html.Div(id="recommendation_area", children=[
            html.P("Pairwise Distance:", style={"padding": "5px"}),
            dcc.Dropdown(id="recommendation_metric", options=[
                {"label": name, "value": name} for name in ("cosine", "euclidean", "manhattan")
            ], value="cosine"),
            html.P("Recommendations for:", style={"padding": "5px"}),
            dcc.Dropdown(id="recommendation_picker"),
            html.P("Recommendations:", style={"padding": "5px"}),
            html.Div(id="recommendations")
        ]),
    ], className="custom-tab", selected_className="custom-tab--selected"
)

arguments = {
    "recommendation_title": misc.HtmlElement("recommendation_picker", "value"),
    "recommendation_metric": misc.HtmlElement("recommendation_metric", "value"),
}
outputs = [Output("recommendations", "children"), Output("recommendation_picker", "options")]


def get_recommendations(n, title, metric, data_df, clusters, titles):
    title_idx = np.argwhere(titles.values == title).ravel()[0]
    title_cluster = clusters[title_idx]
    dists = pairwise_distances(data_df.values[title_idx, :][None, :], data_df.values, metric=metric)
    dists_in_cluster = pairwise_distances(data_df.values[title_idx, :][None, :],
                                          data_df.values[clusters == title_cluster],
                                          metric=metric)

    top = np.argsort(dists).ravel()[:n]
    top_cluster = np.argsort(dists_in_cluster).ravel()[:n]

    return pd.DataFrame(
        {"Top Recommendations": titles.values[top],
         "Top Recommendations Score": dists.ravel()[top],
         "Top Recommendations in Cluster":
             np.pad(titles.values[clusters == title_cluster][top_cluster],
                    (0, max(n - len(top_cluster), 0)), 'constant'),
         "Top Recommendations in Cluster Score":
             np.pad(dists_in_cluster.ravel()[top_cluster], (0, max(n - len(top_cluster), 0)), 'constant')
         }
    )


def get_recommendation_output(
        recommendation_title, recommendation_metric, n_recommendations,
        df_arr, clusters, titles
):
    recommendations = None
    titles_for_recommendation = []

    if titles is not None:
        titles_for_recommendation = [{"label": title, "value": title} for title in sorted(titles)]

    if recommendation_title and df_arr is not None:
        recommendation_df = get_recommendations(n_recommendations, recommendation_title, recommendation_metric,
                                                df_arr, clusters, titles)
        recommendations = misc.generate_datatable(recommendation_df.round(2), "recommendations_table")

    return recommendations, titles_for_recommendation
