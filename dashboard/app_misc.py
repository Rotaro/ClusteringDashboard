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

from data.tv_series_data import get as get_tv_series_data
from data import text_processing


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
    return [
        inp
        for d in text_processing_dicts
        for inp in [*[Input('%s' % name, 'value') for name, cls in d.items()],
                    *[Input('%s|%s' % (name, opt), 'value')
                      for name, cls in d.items()
                      for opt in cls.get_options().keys()]]
    ]
