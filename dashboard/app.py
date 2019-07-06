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
from data import text_processing
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

_data_sources = {
    "TV Series Data": lambda: get_tv_series_data("../data/tv_series_data.csv")
}

preprocess = {
    "Stem": text_processing.Stem,
    "Lemmatize": text_processing.Lemmatize,
}

to_array = {
    "TFID": text_processing.TFID,
    "Count": text_processing.Count
}


@cache.memoize()
def get_data_source(data_name):
    return _data_sources[data_name]() if data_name is not None else None


@cache.memoize()
def get_chosen_cols(data_name, chosen_cols, chosen_preprocess):
    df = get_data_source(data_name)
    if df is not None and chosen_cols is not None and len(chosen_cols) > 0:
        series = df[chosen_cols[0]].astype(str)
        for col in chosen_cols[1:]:
            series = series + ". " + df[col].astype(str)

        df = pd.DataFrame(series.values, columns=["text_to_cluster"])

        for preprocess_div in chosen_preprocess:
            name, active, options = text_processing.TextAction.parse_dash_elements(
                preprocess_div["props"]["children"])
            if not active:
                continue
            df = preprocess[name](**options).apply(df)

        return df


@cache.memoize()
def get_cluster_data(data_name, chosen_cols, chosen_preprocess, chosen_to_array):
    df = get_chosen_cols(data_name, chosen_cols, chosen_preprocess)
    data_df = None
    if df is not None:
        for to_array_div in chosen_to_array:
            name, active, options = text_processing.TextAction.parse_dash_elements(to_array_div["props"]["children"])
            if not active or data_df is not None:
                continue
            data_df = to_array[name](**options).apply(df)

    return data_df


app.layout = html.Div([
    html.Div(id="data_area", children=[
        html.H5(children='Choose data to cluster:'),
        dcc.Dropdown(
            id='data',
            options=[
                {'label': name, 'value': name} for name, func in _data_sources.items()
            ]
        ),
        html.H5(children='Top rows of data:'),
        html.Div(dash_table.DataTable(id='data_head_table'), id='data_head_div')
    ], style={"border": 'grey solid', 'padding': '5px'}),

    html.Div(id="text_to_cluster_area", children=[
        html.H5('Choose columns to cluster:'),
        html.Div(dcc.Dropdown(id='col_picker'), id='col_picker_div'),

        html.Div([
            html.H5('Text preprocessing:'),
            html.Div([html.Div(cls().to_dash_elements()) for name, cls in preprocess.items()], id='preprocess_picker')
        ], id='preprocess_picker_div'),
        html.Div([
            html.H5('Text to array:'),
            html.Div([html.Div(cls().to_dash_elements()) for name, cls in to_array.items()], id='to_array_picker')
        ], id='to_array_div'),

        html.H5('Text used for clustering:'),
        html.Div(dash_table.DataTable(id='text_to_cluster'), id='text_to_cluster_div'),

        html.H5('Cluster matrix:', id='cluster_matrix_header'),
        html.Div(dash_table.DataTable(id='data_to_cluster'), id='data_to_cluster_div')
    ], style={"border": 'grey solid', 'marginTop': '20px', 'padding': '5px'}),

    html.Div(id="plot_area", children=[dcc.Graph(id="plot-3d")])
])


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
     Output('cluster_matrix_header', 'children')],
    [Input('col_picker', 'value'),
     Input('preprocess_picker', 'children'),
     Input('to_array_picker', 'children'),
     *misc.generate_text_processing_inputs([preprocess, to_array])],
    [State('data', 'value')]
)
def update_text_to_cluster_area(chosen_cols, chosen_preprocess, chosen_to_array, *args):
    data_name = args[-1]
    df = get_chosen_cols(data_name, chosen_cols, chosen_preprocess)
    data_df = get_cluster_data(data_name, chosen_cols, chosen_preprocess, chosen_to_array)
    cluster_matrix_header = "Matrix to cluster (shape %dx%d):" % ((0, 0) if data_df is None else data_df.shape)

    return misc.generate_datatable(df, 'text_to_cluster', 5, max_cell_width="1000px"), \
           misc.generate_datatable(data_df.sample(10, axis=1) if data_df is not None else None,
                                   'data_to_cluster', 5, max_cell_width="350px"), \
           cluster_matrix_header


@app.callback(
    Output('plot-3d', 'figure'),
    [Input('cluster_matrix_header', 'children')],
    [State('data', 'value'),
     State('col_picker', 'value'),
     State('preprocess_picker', 'children'),
     State('to_array_picker', 'children')]
)
def plot3d(inp, data_name, chosen_cols, chosen_preprocess, chosen_to_array):
    df = get_data_source(data_name)
    data_df = get_cluster_data(data_name, chosen_cols, chosen_preprocess, chosen_to_array)
    if df is None or data_df is None:
        return go.Figure()

    # Find 2-D centroids for clusters using PCA
    from sklearn.decomposition import TruncatedSVD, PCA, LatentDirichletAllocation
    from sklearn.preprocessing import Normalizer

    arr = Normalizer().fit_transform(data_df.values)
    pca = PCA(n_components=3).fit_transform(arr)

    layout = go.Layout(
        margin=dict(l=0, r=0, b=0, t=0),
        # scene=dict(xaxis=axes, yaxis=axes, zaxis=axes),
    )
    scatter = go.Scatter3d(
        name="PCA - 3D",
        x=pca[:, 0],
        y=pca[:, 1],
        z=pca[:, 2],
        text=df.org_title,
        textposition="top center",
        mode="markers",
        marker=dict(size=5, symbol="circle"),
    )
    figure = go.Figure(data=[scatter], layout=layout)

    return figure


if __name__ == '__main__':
    app.run_server(debug=True)
