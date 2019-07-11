# -*- coding: utf-8 -*-
import logging

import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go

import pandas as pd
import numpy as np

import ClusteringDashboard.data.text_processing as text_processing
import ClusteringDashboard.model as model
import ClusteringDashboard.dashboard.app_misc as misc
from ClusteringDashboard.data.tv_series_data import get as get_tv_series_data
from flask_caching import Cache


external_stylesheets = [
    # Dash CSS
    "https://codepen.io/chriddyp/pen/bWLwgP.css",
    # Loading screen CSS
    "https://codepen.io/chriddyp/pen/brPBPO.css"]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

CACHE_CONFIG = {
    "CACHE_TYPE": "simple"
}
cache = Cache()
cache.init_app(app.server, config=CACHE_CONFIG)

data_sources = {
    "TV Series Data": lambda: get_tv_series_data("tv_series_data.csv")
}

avail_preprocess = {
    "WikipediaTextCleanup": text_processing.WikipediaTextCleanup,
    "Stem": text_processing.Stem,
    "Lemmatize": text_processing.Lemmatize,
}

to_array = misc.DropdownWithOptions(
    header="Choose text to array method:", dropdown_id="to_array", dropdown_objects={
        "TFIDF": text_processing.TFIDF,
        "BOW": text_processing.BOW,
    }, include_refresh_button=True
)
if text_processing.fasttext is not None:
    to_array.dropdown_objects["FastText"] = text_processing.FastText
if text_processing.fasttext is not None and text_processing.FastTextPretrained.has_pretrained():
    to_array.dropdown_objects["FastTextPretrained"] = text_processing.FastTextPretrained

dim_reductions = misc.DropdownWithOptions(
    header="Choose dimensionality reduction method for plotting:", dropdown_id="dim_reduction", dropdown_objects={
        "NMF": model.dim_reduction.NMF,
        "PCA": model.dim_reduction.PCA,
        "SVD": model.dim_reduction.SVD,
        "TSNE": model.dim_reduction.TSNE,
    }, include_refresh_button=True
)

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
def get_data_source(data_name):
    return data_sources[data_name]() if data_name is not None else None


@cache.memoize()
def get_chosen_cols(data_name, chosen_cols):
    df = get_data_source(data_name)
    if df is not None and chosen_cols is not None and len(chosen_cols) > 0:
        return text_processing.join_columns(df, chosen_cols)


@cache.memoize()
def get_preprocessed(data_name, chosen_cols, chosen_preprocess):
    df = get_chosen_cols(data_name, chosen_cols)
    if df is not None and chosen_preprocess:
        for preprocess_div in chosen_preprocess:
            name, active = preprocess_div["props"]["id"], preprocess_div["props"]["value"]
            if not active:
                continue
            df = avail_preprocess[name]().apply(df)

        return df


@cache.memoize()
def get_cluster_data(data_name, chosen_cols, chosen_preprocess, chosen_to_array, to_array_options):
    df = get_preprocessed(data_name, chosen_cols, chosen_preprocess)
    data_df = None
    if df is not None and to_array_options:
        data_df = to_array.apply(chosen_to_array, to_array_options, df)

    return df, data_df


@cache.memoize()
def get_dim_reduction(data_name, chosen_cols, chosen_preprocess, chosen_to_array, to_array_options,
                      dim_reduction, dim_reduction_options):
    df, data_df = get_cluster_data(data_name, chosen_cols, chosen_preprocess, chosen_to_array, to_array_options)
    return df, data_df, dim_reductions.apply(dim_reduction, dim_reduction_options, data_df)


@cache.memoize()
def get_clusters(data_name, chosen_cols, chosen_preprocess, chosen_to_array, to_array_options,
                 dim_reduction, dim_reduction_options,
                 clustering, clustering_options, cluster_on_dim_reduction):
    df, data_df, coords = get_dim_reduction(
        data_name, chosen_cols, chosen_preprocess, chosen_to_array, to_array_options,
        dim_reduction, dim_reduction_options
    )

    if clustering_options:
        clusters = clusterings.apply(clustering, clustering_options,
                                     pd.DataFrame(coords) if cluster_on_dim_reduction else data_df)
    else:
        clusters = np.zeros(coords.shape[0])

    return df, data_df, coords, clusters


app.layout = html.Div([
    html.Div([
        dcc.Tabs(id="tabs_1", children=[
            dcc.Tab(label="Data Selection", children=[
                html.Div(id="data_area", children=[
                    # Choose data
                    html.H5(children="Choose data:"),
                    dcc.Dropdown(
                        id="data",
                        options=[{"label": name, "value": name} for name, func in data_sources.items()]
                    ),
                    # Choose columns
                    html.H5("Choose columns to use:"),
                    html.Div(dcc.Dropdown(id="col_picker"), id="col_picker_div"),
                    # Display top rows
                    html.H5("Top rows:"),
                    html.Div(dash_table.DataTable(id="data_head_table"), id="data_head_div")
                ]),
            ], className="custom-tab", selected_className="custom-tab--selected"),
            dcc.Tab(label="Text Preprocessing", children=[
                html.Div(id="text_preprocessing_area", children=[
                    # Choose preprocessing
                    html.Div([
                        html.H5("Text preprocessing:"),
                        html.Div([dcc.Checklist(id=name, options=[{"label": name, "value": name}], value=[],
                                                style={"padding": "5px", "margin": "5px"})
                                  for name, cls in avail_preprocess.items()],
                                 id="preprocess_picker")
                    ], id="preprocess_picker_div"),
                    # Display preprocessed text
                    html.H5("Text used for clustering:"),
                    html.Div(dash_table.DataTable(id="text_to_cluster"), id="text_to_cluster_div"),
                ]),
            ], className="custom-tab", selected_className="custom-tab--selected"),
            dcc.Tab(label="Text to Array", children=[
                html.Div(id="text_to_array_area", children=[
                    # Choose text_to_array method
                    to_array.generate_dash_element(),
                    # Display array
                    html.H5("Cluster array:", id="cluster_array_header"),
                    html.Div(dash_table.DataTable(id="data_array"), id="data_to_cluster_div"),
                ]),
            ], className="custom-tab", selected_className="custom-tab--selected"),
            dcc.Tab(label="Hide", children=[], className="custom-tab", selected_className="custom-tab--selected"),
        ]),
    ], style={"border": "grey solid", "padding": "5px"}),

    html.Div([
        dcc.Tabs(id="tabs_2", children=[
            dcc.Tab(label="Plotting / Dimensionality Reduction", children=[
                html.Div(id="dim_red_area", children=[
                    dim_reductions.generate_dash_element(),
                ]),
            ], className="custom-tab", selected_className="custom-tab--selected"),
            dcc.Tab(label="Clustering", children=[
                html.Div(id="clustering_area", children=[
                    dcc.Checklist(id="cluster_on_dim_reduction",
                                  options=[{"label": "Cluster on dimensional reduction",
                                            "value": "true"}], value=[],
                                  style={"padding": "5px", "margin": "5px"}),
                    clusterings.generate_dash_element(),
                ]),
            ], className="custom-tab", selected_className="custom-tab--selected"),
            dcc.Tab(label="Hide", children=[], className="custom-tab", selected_className="custom-tab--selected"),
        ]),
    ], style={"marginTop": "10px", "padding": "5px", "border": "grey solid"}),

    html.Div(id="plot_area", children=[
        dcc.Tabs(id="tabs_3", children=[
            dcc.Tab(label="Plot", children=[dcc.Graph(id="scatter-plot")],
                    className="custom-tab", selected_className="custom-tab--selected"),
            dcc.Tab(label="Clusters", children=[html.Div(id="cluster_info_table")],
                    className="custom-tab", selected_className="custom-tab--selected"),
            dcc.Tab(label="Recommendation", children=[
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
            ], className="custom-tab", selected_className="custom-tab--selected"),
        ], style={"border": "grey solid", "padding": "5px", "marginTop": "10px"})
    ], style={"marginTop": "10px", "padding": "5px", "border": "grey solid"})
], style={"background-color": "#f2f2f2", "margin": "20px"})


to_array.generate_update_options_callback(app)
dim_reductions.generate_update_options_callback(app)
clusterings.generate_update_options_callback(app)


@app.callback(
    [Output("data_head_div", "children"), Output("col_picker_div", "children")],
    [Input("data", "value")]
)
def update_data_area(input_value):
    df = get_data_source(input_value)
    return misc.generate_datatable(df, "data_head_table", 5),\
           misc.generate_column_picker(df, "col_picker")


@app.callback(
    [Output("text_to_cluster_div", "children"), Output("data_to_cluster_div", "children"),
     Output("cluster_array_header", "children")],
    [Input("col_picker", "value"),
     Input("preprocess_picker", "children"),
     to_array.get_input("options"),
     *[Input(name, "value") for name in avail_preprocess.keys()],
     Input("to_array_refresh", "n_clicks")],
    [State("data", "value"), to_array.get_state("dropdown")]
)
def update_text_to_cluster_area(chosen_cols, chosen_preprocess, to_array_options, *args):
    data_name, chosen_to_array = args[-2:]
    df, data_df = get_cluster_data(data_name, chosen_cols, chosen_preprocess, chosen_to_array, to_array_options)
    cluster_array_header = "Matrix to cluster (shape %dx%d):" % ((0, 0) if data_df is None else data_df.shape)

    return misc.generate_datatable(df, "text_to_cluster", 5, max_cell_width="2000px"), \
           misc.generate_datatable(data_df.sample(min(data_df.shape[1], 20), axis=1).round(2) if data_df is not None else None,
                                   "data_to_cluster", 5, max_cell_width="350px"), \
           cluster_array_header


@app.callback(
    [Output("scatter-plot", "figure"), Output("cluster_info_table", "children"),
     Output("recommendation_picker", "options")],
    [Input("cluster_array_header", "children"),
     dim_reductions.get_input("dropdown"),
     dim_reductions.get_input("options"),
     dim_reductions.get_input("refresh"),
     clusterings.get_input("dropdown"),
     clusterings.get_input("options"),
     clusterings.get_input("refresh"),
     Input("cluster_on_dim_reduction", "value")],
    [State("data", "value"),
     State("col_picker", "value"),
     State("preprocess_picker", "children"),
     to_array.get_state("dropdown"),
     to_array.get_state("options")]
)
def plot(cluster_array_header, dim_reduction, dim_reduction_options, dim_reduction_refresh,
         clustering, clustering_options, clustering_refresh, cluster_on_dim_reduction,
         data_name, chosen_cols, chosen_preprocess, chosen_to_array, to_array_options):
    if data_name is None or not chosen_cols or not dim_reduction_options:
        return go.Figure(layout=go.Layout(margin=dict(l=0, r=0, b=0, t=0), plot_bgcolor="#f2f2f2")), None, None

    df, data_df, coords, clusters = get_clusters(
        data_name, chosen_cols, chosen_preprocess, chosen_to_array, to_array_options,
        dim_reduction, dim_reduction_options,
        clustering, clustering_options, cluster_on_dim_reduction
    )
    titles = get_data_source(data_name).org_title

    # Plots
    scatter_plots = misc.get_scatter_plots(coords, clusters, titles)
    figure = go.Figure(data=scatter_plots, layout=go.Layout(margin=dict(l=0, r=0, b=0, t=0), plot_bgcolor="#f2f2f2",
                                                            legend={"bgcolor": "#f2f2f2"}, hovermode="closest"))

    # Cluster information
    bow_data_df = text_processing.TFIDF(ngram_range=(1, 1)).apply(
        get_preprocessed(data_name, chosen_cols, chosen_preprocess)
    )
    cluster_info_df = misc.get_cluster_info_df(10, clusters, titles, bow_data_df)

    recommendation_options = [{"label": org_title, "value": org_title} for org_title in sorted(titles.values)]

    return figure, misc.generate_datatable(cluster_info_df, "cluster_info", 1000, "600px"), recommendation_options


@app.callback(
    Output("recommendations", "children"),
    [Input("recommendation_picker", "value"), Input("recommendation_metric", "value")],
    [dim_reductions.get_state("dropdown"),
     dim_reductions.get_state("options"),
     clusterings.get_state("dropdown"),
     clusterings.get_state("options"),
     State("cluster_on_dim_reduction", "value"),
     State("data", "value"),
     State("col_picker", "value"),
     State("preprocess_picker", "children"),
     to_array.get_state("dropdown"),
     to_array.get_state("options")]
)
def recommend(recommend_for, recommendation_metric, dim_reduction, dim_reduction_options,
              clustering, clustering_options, cluster_on_dim_reduction,
              data_name, chosen_cols, chosen_preprocess, chosen_to_array, to_array_options):
    if not recommend_for or data_name is None or not chosen_cols or not dim_reduction_options or not clustering_options:
        return

    df, data_df, coords, clusters = get_clusters(
        data_name, chosen_cols, chosen_preprocess, chosen_to_array, to_array_options,
        dim_reduction, dim_reduction_options,
        clustering, clustering_options, cluster_on_dim_reduction
    )
    titles = get_data_source(data_name).org_title

    recommendation_df = misc.get_recommendations(20, recommend_for, recommendation_metric, data_df, clusters, titles)
    return misc.generate_datatable(recommendation_df, "recommendations_table")


if __name__ == "__main__":
    app.run_server(debug=True)
