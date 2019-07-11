# -*- coding: utf-8 -*-
import logging

import numpy as np
import matplotlib
import pandas as pd
from sklearn.metrics.pairwise import pairwise_distances

import dash_table
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go


def flatten(obj):
    if isinstance(obj, (list, tuple)):
        return [e for l in obj for e in flatten(l)]
    else:
        return [obj]


class DropdownWithOptions:
    style = {"border": 'grey solid', 'padding': '5px', 'margin': '5px'}

    def __init__(self, header, dropdown_id, dropdown_objects, include_refresh_button):
        self.header = header
        self.dropdown_id = dropdown_id
        self.dropdown_objects = dropdown_objects
        self.include_refresh_button = include_refresh_button

    def generate_dash_element(self):
        return html.Div([
            html.H5(self.header, id="%s_heading" % self.dropdown_id),
            dcc.Dropdown(
                id=self.dropdown_id,
                options=[{'label': name, 'value': name} for name, _ in self.dropdown_objects.items()]
            ),
            html.P("Options:", style={'padding': '5px', 'margin': '5px'}),
            html.Div(id='%s_options' % self.dropdown_id, style=self.style),
            html.Div(html.Button("Refresh", id="%s_refresh" % self.dropdown_id, style=self.style))
        ], id='%s_div' % self.dropdown_id)

    @property
    def _dropdown_args(self):
        return '%s' % self.dropdown_id, 'value'

    @property
    def _refresh_args(self):
        return '%s_refresh' % self.dropdown_id, 'n_clicks'

    @property
    def _options_args(self):
        return '%s_options' % self.dropdown_id, 'children'

    def get_input(self, element='dropdown'):
        return Input(*getattr(self, f"_{element}_args"))

    def get_state(self, element='dropdown'):
        return State(*getattr(self, f"_{element}_args"))

    def generate_update_options_callback(self, app):
        @app.callback(
            Output(*self._options_args),
            [self.get_input('dropdown')]
        )
        def update_options(dropdown_choice):
            if dropdown_choice is None:
                return
            return self.generate_options_element(dropdown_choice)

    def generate_options_element(self, dropdown_choice):
        return [
            e
            for option_name, default_value in self.dropdown_objects[dropdown_choice].get_options().items()
            for e in ("%s: " % option_name,
                      dcc.Input(id="%s|%s" % (dropdown_choice, option_name), type="text", value=str(default_value)))
        ]

    def _parse_options_element(self, options_element):
        options = {}
        for e in options_element:
            if not isinstance(e, dict):
                continue

            id, value = e["props"]["id"], tuple(e["props"]["value"].strip("()").split(","))
            if len(value) == 1:
                value = value[0]
            options[id.split("|")[1]] = value

        return options

    def apply(self, dropdown_choice, options_element, df):
        options = self._parse_options_element(options_element)
        return self.dropdown_objects[dropdown_choice](**options).apply(df)


def generate_datatable(df, table_id, max_rows=10, max_cell_width="600px"):
    if df is None:
        return dash_table.DataTable(id=table_id)

    return dash_table.DataTable(
        id=table_id,
        columns=[{"name": i, "id": i} for i in df.columns],
        data=df[:max_rows].to_dict("rows"),
        css=[{
            'selector': '.dash-cell div.dash-cell-value',
            'rule': 'display: inline; white-space: inherit; overflow: inherit; text-overflow: inherit;'
        }],
        style_table={
            'maxHeight': '350px',
            'overflowY': 'auto',
            'border': 'thin lightgrey solid',
        },
        style_cell={
            'whiteSpace': 'no-wrap',
            'overflow': 'hidden',
            'textOverflow': 'ellipsis',
            'minWidth': '0px', 'maxWidth': max_cell_width,
        }
    )


def generate_column_picker(df, element_id):
    if df is None:
        return dcc.Dropdown(id=element_id)

    return dcc.Dropdown(
        id=element_id, value=[], multi=True,
        options=[{'label': col, 'value': col} for col in df.columns]
    )


def generate_text_processing_inputs(text_processing_dicts):
    L = [
        Input(name, 'value')
        for d in text_processing_dicts
        for cls_name, cls in d.items()
        for name in cls().get_dash_element_ids()
    ]
    return L


def get_scatter_plots(coords, clusters, titles):
    # Generate unique colors (https://stackoverflow.com/a/55828367)
    np.random.seed(42)
    colors = np.random.choice(list(matplotlib.colors.cnames.values()), size=np.unique(clusters).size, replace=False)

    dims = list(zip(("x", "y", "z"), range(coords.shape[1])))
    scatter_class = go.Scatter3d if len(dims) == 3 else go.Scatter
    scatter_plots = []
    for i, cluster in enumerate(np.unique(clusters)):
        idx = clusters == cluster
        if idx.sum() == 0:
            continue

        scatter_plots.append(scatter_class(
            name="Cluster %d" % cluster,
            **{label: coords[idx, i] for label, i in dims},
            text=titles.values[idx],
            textposition="top center",
            mode="markers",
            marker=dict(size=5 if len(dims) == 3 else 12, symbol="circle", color=colors[i]),
        ))

    return scatter_plots


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

    cluster_info_df = pd.DataFrame(cluster_info, columns=[
        "Cluster", "Size",
        *["Sample%d" % i for i in range(1, n_cluster_info + 1)],
        *["Top Word %d" % i for i in range(1, n_cluster_info + 1)],
    ])

    return cluster_info_df


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
         "Top Recommendations in Cluster":
             np.pad(titles.values[clusters == recommend_for_cluster][top_cluster],
                    (0, max(n - len(top_cluster), 0)), 'constant')
         }
    )
