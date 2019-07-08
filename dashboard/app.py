# -*- coding: utf-8 -*-
import logging

import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go

import os
import pandas as pd
import numpy as np

import dashboard.app_misc as misc
from data.tv_series_data import get as get_tv_series_data
import data.text_processing
import model.dim_reduction
import model.clustering
from flask_caching import Cache


external_stylesheets = [
    # Dash CSS
    'https://codepen.io/chriddyp/pen/bWLwgP.css',
    # Loading screen CSS
    'https://codepen.io/chriddyp/pen/brPBPO.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

CACHE_CONFIG = {
    'CACHE_TYPE': 'simple'
}
cache = Cache()
cache.init_app(app.server, config=CACHE_CONFIG)

data_sources = {
    "TV Series Data": lambda: get_tv_series_data("../data/tv_series_data.csv")
}

avail_preprocess = {
    "Stem": data.text_processing.Stem,
    "Lemmatize": data.text_processing.Lemmatize,
}

avail_to_array = {
    "TFIDF": data.text_processing.TFIDF,
    "BOW": data.text_processing.BOW
}

avail_dim_reductions = {
    "NMF": model.dim_reduction.NMF,
    "PCA": model.dim_reduction.PCA,
    "SVD": model.dim_reduction.SVD,
    "TSNE": model.dim_reduction.TSNE,
}

to_array = misc.DropdownWithOptions("Choose text to array method:", "to_array", avail_to_array, True)
dim_reductions = misc.DropdownWithOptions("Choose dimensionality reduction method for plotting:",
                                          "dim_reduction", avail_dim_reductions, True)


@cache.memoize()
def get_data_source(data_name):
    return data_sources[data_name]() if data_name is not None else None


@cache.memoize()
def get_chosen_cols(data_name, chosen_cols):
    df = get_data_source(data_name)
    if df is not None and chosen_cols is not None and len(chosen_cols) > 0:
        series = df[chosen_cols[0]].astype(str)
        for col in chosen_cols[1:]:
            series = series + ". " + df[col].astype(str)

        df = pd.DataFrame(series.values, columns=["text_to_cluster"])

        return df


@cache.memoize()
def get_preprocessed(data_name, chosen_cols, chosen_preprocess):
    df = get_chosen_cols(data_name, chosen_cols)
    if df is not None and chosen_preprocess:
        for preprocess_div in chosen_preprocess:
            name, active, options = data.text_processing.TextAction.parse_dash_elements(preprocess_div["props"]["children"])
            if not active:
                continue
            df = avail_preprocess[name](**options).apply(df)

        return df


@cache.memoize()
def get_cluster_data(data_name, chosen_cols, chosen_preprocess, chosen_to_array, to_array_options):
    df = get_preprocessed(data_name, chosen_cols, chosen_preprocess)
    data_df = None
    if df is not None and to_array_options:
        data_df = to_array.apply(chosen_to_array, to_array_options, df)

    return data_df


app.layout = html.Div([
    html.Div([
        dcc.Tabs(id="tabs_1", children=[
            dcc.Tab(label='Data Selection', children=[
                html.Div(id="data_area", children=[
                    # Choose data
                    html.H5(children='Choose data:'),
                    dcc.Dropdown(
                        id='data',
                        options=[{'label': name, 'value': name} for name, func in data_sources.items()]
                    ),
                    # Choose columns
                    html.H5('Choose columns to use:'),
                    html.Div(dcc.Dropdown(id='col_picker'), id='col_picker_div'),
                    # Display top rows
                    html.H5('Top rows:'),
                    html.Div(dash_table.DataTable(id='data_head_table'), id='data_head_div')
                ]),
            ], className='custom-tab', selected_className='custom-tab--selected'),
            dcc.Tab(label='Text Preprocessing', children=[
                html.Div(id="text_preprocessing_area", children=[
                    # Choose preprocessing
                    html.Div([
                        html.H5('Text preprocessing:'),
                        html.Div(
                            [html.Div(cls().to_dash_elements()) for name, cls in avail_preprocess.items()],
                            id='preprocess_picker')
                    ], id='preprocess_picker_div'),
                    # Display preprocessed text
                    html.H5('Text used for clustering:'),
                    html.Div(dash_table.DataTable(id='text_to_cluster'), id='text_to_cluster_div'),
                ]),
            ], className='custom-tab', selected_className='custom-tab--selected'),
            dcc.Tab(label='Text to Array', children=[
                html.Div(id="text_to_array_area", children=[
                    # Choose text_to_array method
                    to_array.generate_dash_element(),
                    # Display array
                    html.H5('Cluster array:', id='cluster_array_header'),
                    html.Div(dash_table.DataTable(id='data_array'), id='data_to_cluster_div'),
                ]),
            ], className='custom-tab', selected_className='custom-tab--selected'),
            dcc.Tab(label='Hide', children=[], className='custom-tab', selected_className='custom-tab--selected'),
        ]),
    ], style={'border': 'grey solid', 'padding': '5px'}),

    html.Div([
        dcc.Tabs(id="tabs_2", children=[
            dcc.Tab(label='Plotting / Dimensionality Reduction', children=[
                html.Div(id="dim_red_area", children=[
                    dim_reductions.generate_dash_element(),
                ]),
            ], className='custom-tab', selected_className='custom-tab--selected'),
            dcc.Tab(label='Clustering', children=[
                html.Div(id="clustering_area", children=[]),
            ], className='custom-tab', selected_className='custom-tab--selected'),
            dcc.Tab(label='Hide', children=[], className='custom-tab', selected_className='custom-tab--selected'),
        ]),
    ], style={'marginTop': '10px', 'padding': '5px', 'border': 'grey solid'}),

    html.Div(id="plot_area", children=[
        dcc.Graph(id="plot-3d")
    ], style={"border": 'grey solid', 'padding': '5px', 'marginTop': '20px'})
], style={'background-color': '#f2f2f2', 'margin': '20px'})


to_array.generate_options_callback(app)
dim_reductions.generate_options_callback(app)


@app.callback(
    [Output('data_head_div', 'children'), Output('col_picker_div', 'children')],
    [Input('data', 'value')]
)
def update_data_area(input_value):
    df = get_data_source(input_value)
    return misc.generate_datatable(df, 'data_head_table', 5),\
           misc.generate_column_picker(df, 'col_picker')


@app.callback(
    [Output('text_to_cluster_div', 'children'), Output('data_to_cluster_div', 'children'),
     Output('cluster_array_header', 'children')],
    [Input('col_picker', 'value'),
     Input('preprocess_picker', 'children'),
     to_array.get_input('options'),
     *misc.generate_text_processing_inputs([avail_preprocess]),
     Input('to_array_refresh', 'n_clicks')],
    [State('data', 'value'), to_array.get_state('dropdown')]
)
def update_text_to_cluster_area(chosen_cols, chosen_preprocess, to_array_options, *args):
    data_name, chosen_to_array = args[-2:]
    df = get_preprocessed(data_name, chosen_cols, chosen_preprocess)
    data_df = get_cluster_data(data_name, chosen_cols, chosen_preprocess, chosen_to_array, to_array_options)
    cluster_array_header = "Matrix to cluster (shape %dx%d):" % ((0, 0) if data_df is None else data_df.shape)

    return misc.generate_datatable(df, 'text_to_cluster', 5, max_cell_width="1000px"), \
           misc.generate_datatable(data_df.sample(20, axis=1).round(2) if data_df is not None else None,
                                   'data_to_cluster', 5, max_cell_width="350px"), \
           cluster_array_header


@app.callback(
    Output('plot-3d', 'figure'),
    [Input('cluster_array_header', 'children'),
     dim_reductions.get_input('dropdown'),
     dim_reductions.get_input('options'),
     dim_reductions.get_input('refresh')],
    [State('data', 'value'),
     State('col_picker', 'value'),
     State('preprocess_picker', 'children'),
     to_array.get_state('dropdown'),
     to_array.get_state('options')]
)
def plot(cluster_array, dim_reduction, dim_reduction_options, dim_reduction_refresh,
         data_name, chosen_cols, chosen_preprocess, chosen_to_array, to_array_options):
    df = get_data_source(data_name)
    data_df = get_cluster_data(data_name, chosen_cols, chosen_preprocess, chosen_to_array, to_array_options)
    if df is None or data_df is None or not dim_reduction_options:
        return go.Figure()

    arr = dim_reductions.apply(dim_reduction, dim_reduction_options, data_df)
    dims = list(zip(("x", "y", "z"), range(arr.shape[1])))
    scatter_class = go.Scatter3d if len(dims) == 3 else go.Scatter

    from sklearn.cluster import KMeans
    clusters = KMeans(n_clusters=8).fit_predict(data_df.values)

    data = []
    for cluster in np.unique(clusters):
        idx = clusters == cluster
        data.append(scatter_class(
            name="Cluster %d" % cluster,
            **{label: arr[idx, i] for label, i in dims},
            text=df.org_title.values[idx],
            textposition="top center",
            mode="markers",
            marker=dict(size=5, symbol="circle"),
        ))
    figure = go.Figure(data=data, layout=go.Layout(margin=dict(l=0, r=0, b=0, t=0)))

    return figure


if __name__ == '__main__':
    app.run_server(debug=True)
