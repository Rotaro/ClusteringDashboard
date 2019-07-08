# -*- coding: utf-8 -*-
import logging

import dash_table
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State


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