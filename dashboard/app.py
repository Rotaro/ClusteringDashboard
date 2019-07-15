# -*- coding: utf-8 -*-
import logging

import dash
import dash_daq
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
from ClusteringDashboard.data.tv_series_data import get_imdb_top_250_wikipedia_summaries
from ClusteringDashboard.data.tv_series_data import get_all_wikipedia_summaries
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
    "Top 250 IMDB TV Series": get_imdb_top_250_wikipedia_summaries,
    "Wikipedia TV Series Summaries": get_all_wikipedia_summaries
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
        "None": None,
        "NMF": model.dim_reduction.NMF,
        "PCA": model.dim_reduction.PCA,
        "SVD": model.dim_reduction.SVD,
        "FactorAnalysis": model.dim_reduction.FactorAnalysis,
        "TSNE": model.dim_reduction.TSNE,
    }, include_refresh_button=True
)

plot_dim_reductions = misc.DropdownWithOptions(
    header="Choose dimensionality reduction method for plotting:", dropdown_id="plot_dim_reduction", dropdown_objects={
        "NMF": model.dim_reduction.NMF,
        "PCA": model.dim_reduction.PCA,
        "SVD": model.dim_reduction.SVD,
        "FactorAnalysis": model.dim_reduction.FactorAnalysis,
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
def get_data_source(data_name, sample_percent=100):
    df = data_sources[data_name]() if data_name is not None else None

    if df is not None and "org_title" not in df.columns:
        df["org_title"] = df["title"]

    if df is not None and sample_percent < 100:
        df = df.sample(frac=sample_percent / 100)

    return df


@cache.memoize()
def get_chosen_cols(data_name, use_sample_perc, chosen_cols):
    df = get_data_source(data_name, use_sample_perc)
    if df is not None and chosen_cols is not None and len(chosen_cols) > 0:
        return text_processing.join_columns(df, chosen_cols)


@cache.memoize()
def get_preprocessed(data_name, use_sample_perc, chosen_cols, chosen_preprocess):
    df = get_chosen_cols(data_name, use_sample_perc, chosen_cols)
    if df is not None and chosen_preprocess:
        for preprocess_div in chosen_preprocess:
            name, active = preprocess_div["props"]["id"], preprocess_div["props"]["value"]
            if not active:
                continue
            df = avail_preprocess[name]().apply(df)

        return df


@cache.memoize()
def get_cluster_data(data_name, use_sample_perc, chosen_cols, chosen_preprocess, chosen_to_array, to_array_options):
    df = get_preprocessed(data_name, use_sample_perc, chosen_cols, chosen_preprocess)
    data_df = None
    if df is not None and to_array_options:
        data_df = to_array.apply(chosen_to_array, to_array_options, df)

    return df, data_df


@cache.memoize()
def get_dim_reduction(data_name, use_sample_perc, chosen_cols, chosen_preprocess, chosen_to_array, to_array_options,
                      dim_reduction, dim_reduction_options):
    df, data_df = get_cluster_data(data_name, use_sample_perc, chosen_cols, chosen_preprocess,
                                   chosen_to_array, to_array_options)
    if df is None or data_df is None or not dim_reduction_options:
        return df, data_df, None

    return df, data_df, pd.DataFrame(dim_reductions.apply(dim_reduction, dim_reduction_options, data_df))


@cache.memoize()
def get_clusters(data_name, use_sample_perc, chosen_cols, chosen_preprocess, chosen_to_array, to_array_options,
                 dim_reduction, dim_reduction_options,
                 clustering, clustering_options):
    df, data_df, dim_red_df = get_dim_reduction(
        data_name, use_sample_perc, chosen_cols, chosen_preprocess, chosen_to_array, to_array_options,
        dim_reduction, dim_reduction_options
    )
    to_cluster = dim_red_df if dim_red_df is not None else data_df

    if clustering_options:
        clusters = clusterings.apply(clustering, clustering_options, to_cluster)
    else:
        clusters = np.zeros(to_cluster.shape[0])

    return df, data_df, dim_red_df, clusters


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
                    html.H5("Percentage of data to use:"),
                    html.Div(dash_daq.NumericInput("input_sample_perc", value=100, min=0, max=100)),
                    # Choose columns
                    html.H5("Choose columns to use:"),
                    html.Div(dcc.Dropdown(id="data_col_picker"), id="data_col_picker_div"),
                    # Display top rows
                    html.H5("Top rows:", id="data_top_rows"),
                    html.Div(dash_table.DataTable(id="data_head_table"), id="data_head_div")
                ]),
            ], className="custom-tab", selected_className="custom-tab--selected"),
            dcc.Tab(label="Text Preprocessing", children=[
                html.Div(id="text_preprocess_area", children=[
                    # Choose preprocessing
                    html.Div([
                        html.H5("Text preprocessing:"),
                        html.Div([dcc.Checklist(id=name, options=[{"label": name, "value": name}], value=[],
                                                style={"padding": "5px", "margin": "5px"})
                                  for name, cls in avail_preprocess.items()],
                                 id="text_preprocess_picker")
                    ], id="text_preprocess_picker_div"),
                    # Display preprocessed text
                    html.H5("Text used for clustering:"),
                    html.Div(dash_table.DataTable(id="text_preprocess"), id="text_preprocess_div"),
                ]),
            ], className="custom-tab", selected_className="custom-tab--selected"),
            dcc.Tab(label="Text to Array", children=[
                html.Div(id="text_to_array_area", children=[
                    # Choose text_to_array method
                    to_array.generate_dash_element(),
                    # Display array
                    html.H5("Cluster array:", id="text_to_array_header"),
                    html.Div(dash_table.DataTable(id="text_to_array"), id="text_to_array_div"),
                ]),
            ], className="custom-tab", selected_className="custom-tab--selected"),
            dcc.Tab(label="Dimensionality Reduction", children=[
                html.Div(id="dim_red_area", children=[
                    # Choose text_to_array method
                    dim_reductions.generate_dash_element(),
                    # Display array
                    html.H5("Dimensionality Reduced Array:", id="dim_red_array_header"),
                    html.Div(dash_table.DataTable(id="dim_red_array"), id="dim_red_array_div"),
                ]),
            ], className="custom-tab", selected_className="custom-tab--selected"),
            dcc.Tab(label="Hide", children=[], className="custom-tab", selected_className="custom-tab--selected"),
        ]),
    ], style={"border": "grey solid", "padding": "5px"}),

    html.Div([
        dcc.Tabs(id="tabs_2", children=[
            dcc.Tab(label="Plotting Options", children=[
                html.Div(id="plotting_dim_red_area", children=[
                    plot_dim_reductions.generate_dash_element(),
                ]),
            ], className="custom-tab", selected_className="custom-tab--selected"),
            dcc.Tab(label="Clustering", children=[
                html.Div(id="clustering_area", children=[
                    clusterings.generate_dash_element(),
                ]),
                html.P(children=None, id="cluster_info_text", style={"padding": "5px", "margin": "5px"})
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
plot_dim_reductions.generate_update_options_callback(app)
clusterings.generate_update_options_callback(app)


@app.callback(
    [Output("data_head_div", "children"), Output("data_col_picker_div", "children"),
     Output("recommendation_picker", "options"), Output("data_top_rows", "children")],
    [Input("data", "value"), Input("input_sample_perc", "value")]
)
def update_chosen_data(input_value, use_sample_perc):
    df = get_data_source(input_value, use_sample_perc)

    if df is not None:
        recommendation_options = [{"label": org_title, "value": org_title}
                                  for org_title in sorted(df.org_title.values)]
    else:
        recommendation_options = None

    top_rows_text = "Top Rows (%d rows total)" % (0 if df is None else len(df))

    return misc.generate_datatable(df, "data_head_table", 5),\
           misc.generate_column_picker(df, "data_col_picker"), \
           recommendation_options, top_rows_text


@app.callback(
    Output("text_preprocess_div", "children"),
    [Input("input_sample_perc", "value"),
     Input("data_col_picker", "value"),
     Input("text_preprocess_picker", "children"),
     # Additional triggers, but don't need input
     *[Input(name, "value") for name in avail_preprocess.keys()]],
    [State("data", "value")]
)
def update_text_preprocess_area(ues_sample_perc, chosen_cols, chosen_preprocess, *args):
    data_name = args[-1]
    df = get_preprocessed(data_name, ues_sample_perc, chosen_cols, chosen_preprocess)

    return misc.generate_datatable(df, "text_preprocess", 5, max_cell_width=None)


@app.callback(
    [Output("text_to_array_div", "children"), Output("text_to_array_header", "children")],
    [to_array.get_input("dropdown"), to_array.get_input("options"),
     # Additional triggers, but don't need input
     to_array.get_input("refresh"),
     Input("text_preprocess_div", "children")],
    [State("data", "value"), State("input_sample_perc", "value"),
     State("data_col_picker", "value"), State("text_preprocess_picker", "children")]
)
def update_text_to_array_area(chosen_to_array, to_array_options, *args):
    data_name, use_sample_perc, chosen_cols, chosen_preprocess = args[-4:]
    df, data_df = get_cluster_data(data_name, use_sample_perc, chosen_cols, chosen_preprocess,
                                   chosen_to_array, to_array_options)
    text_to_array_header = "Array to cluster (shape %dx%d):" % ((0, 0) if data_df is None else data_df.shape)
    sample_df = data_df.sample(min(data_df.shape[1], 20), axis=1).round(2) if data_df is not None else None

    return misc.generate_datatable(sample_df, "text_to_array", 5, max_cell_width=None), \
           text_to_array_header


@app.callback(
    Output("dim_red_array_div", "children"),
    [dim_reductions.get_input("dropdown"), dim_reductions.get_input("options"),
     # Additional triggers, but don't need inputs
     dim_reductions.get_input("refresh"),
     Input("text_to_array_div", "children")],
    [State("data", "value"), State("input_sample_perc", "value"),
     State("data_col_picker", "value"), State("text_preprocess_picker", "children"),
     to_array.get_state("dropdown"), to_array.get_state("options")]
)
def update_dim_red_area(chosen_dim_reduction, dim_reduction_options, *args):
    data_name, use_sample_perc, chosen_cols, chosen_preprocess, chosen_to_array, to_array_options = args[-6:]
    df, data_df, dim_red_df = get_dim_reduction(data_name, use_sample_perc, chosen_cols, chosen_preprocess,
                                                chosen_to_array, to_array_options,
                                                chosen_dim_reduction, dim_reduction_options)

    sample_df = dim_red_df.sample(min(dim_red_df.shape[1], 20), axis=1).round(2) if dim_red_df is not None else None

    return misc.generate_datatable(sample_df, "dim_red_array", 5, max_cell_width="350px")


@app.callback(
    [Output("scatter-plot", "figure"), Output("cluster_info_table", "children"),
     Output("cluster_info_text", "children")],
    [to_array.get_input("dropdown"),
     to_array.get_input("options"),
     dim_reductions.get_input("dropdown"),
     dim_reductions.get_input("options"),
     plot_dim_reductions.get_input("dropdown"),
     plot_dim_reductions.get_input("options"),
     clusterings.get_input("dropdown"),
     clusterings.get_input("options"),
     # Additional triggers, but don't need data
     dim_reductions.get_input("refresh"),
     plot_dim_reductions.get_input("refresh"),
     clusterings.get_input("refresh")],
    [State("data", "value"),  State("input_sample_perc", "value"),
     State("data_col_picker", "value"),
     State("text_preprocess_picker", "children")]
)
def plot(chosen_to_array, to_array_options,
         dim_reduction, dim_reduction_options,
         plot_dim_reduction, plot_dim_reduction_options,
         clustering, clustering_options,
         _, _1, _2,
         data_name, use_sample_perc, chosen_cols, chosen_preprocess):
    if data_name is None or not chosen_cols or not plot_dim_reduction_options:
        return go.Figure(layout=go.Layout(margin=dict(l=0, r=0, b=0, t=0), plot_bgcolor="#f2f2f2")), None, None

    # Cluster
    _, data_df, _, clusters = get_clusters(
        data_name, use_sample_perc, chosen_cols, chosen_preprocess, chosen_to_array, to_array_options,
        dim_reduction, dim_reduction_options,
        clustering, clustering_options
    )
    # Get original dataframe to make sure titles are included
    df = get_data_source(data_name, use_sample_perc)

    # Plots
    _, _, coords_df = get_dim_reduction(data_name, use_sample_perc, chosen_cols, chosen_preprocess,
                                        chosen_to_array, to_array_options,
                                        plot_dim_reduction, plot_dim_reduction_options)
    scatter_plots = misc.get_scatter_plots(coords_df.values, clusters, df.org_title)
    figure = go.Figure(data=scatter_plots, layout=go.Layout(margin=dict(l=0, r=0, b=0, t=0), plot_bgcolor="#f2f2f2",
                                                            legend={"bgcolor": "#f2f2f2"}, hovermode="closest"))
    # Cluster information
    bow_data_df = text_processing.TFIDF(ngram_range=(1, 1)).apply(
        get_preprocessed(data_name, use_sample_perc, chosen_cols, chosen_preprocess)
    )
    cluster_info_df = misc.get_cluster_info_df(10, clusters, df.org_title, bow_data_df)

    from sklearn.metrics.cluster import silhouette_score
    if np.unique(clusters).size > 1:
        cluster_info_score = "Silhouette Score: %.2f" % silhouette_score(data_df.values, clusters)
    else:
        cluster_info_score = None

    return figure, misc.generate_datatable(cluster_info_df, "cluster_info", 1000, "600px"), cluster_info_score


@app.callback(
    Output("recommendations", "children"),
    [Input("recommendation_picker", "value"),
     Input("recommendation_metric", "value"),
     clusterings.get_input("dropdown"),
     clusterings.get_input("options"),
     # Additional triggers, but don't need input
     clusterings.get_input("refresh"),
     Input("cluster_info_table", "children")],
    [State("data", "value"), State("input_sample_perc", "value"),
     State("data_col_picker", "value"),
     State("text_preprocess_picker", "children"),
     to_array.get_state("dropdown"),
     to_array.get_state("options"),
     dim_reductions.get_state("dropdown"),
     dim_reductions.get_state("options")]
)
def recommend(recommend_for, recommendation_metric, clustering, clustering_options, *args):
    data_name, use_sample_perc, chosen_cols, chosen_preprocess, chosen_to_array, to_array_options, \
    dim_reduction, dim_reduction_options = args[-8:]

    if not recommend_for or data_name is None or not chosen_cols \
            or not clustering_options:
        return

    df, data_df, dim_red_df, clusters = get_clusters(
        data_name, use_sample_perc, chosen_cols, chosen_preprocess, chosen_to_array, to_array_options,
        dim_reduction, dim_reduction_options,
        clustering, clustering_options
    )
    data_df = dim_red_df if dim_red_df is not None else data_df
    titles = get_data_source(data_name, use_sample_perc).org_title

    recommendation_df = misc.get_recommendations(20, recommend_for, recommendation_metric, data_df, clusters, titles)
    return misc.generate_datatable(recommendation_df.round(2), "recommendations_table")


if __name__ == "__main__":
    app.run_server(debug=True)
