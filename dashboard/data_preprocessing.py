import dash_core_components as dcc
import dash_html_components as html
import dash_table
from dash.dependencies import Output

import data.text_processing as text_processing

import dashboard.app_misc as misc
from dashboard.cache import cache


data_preprocessing_options = {
    "WikipediaTextCleanup": text_processing.WikipediaTextCleanup,
    "Stem": text_processing.Stem,
    "Lemmatize": text_processing.Lemmatize,
}

data_preprocessing_tab = dcc.Tab(
    label="Text Preprocessing", children=[
        html.Div(id="text_preprocess_area", children=[
            # Choose preprocessing
            html.Div([
                html.H5("Text preprocessing:"),
                dcc.Checklist(id="text_preprocess_checklist",
                              options=[{"label": name, "value": name} for name, cls in data_preprocessing_options.items()], value=[],
                              style={"padding": "5px", "margin": "5px"})
            ], id="text_preprocess_checklist_div"),
            # Display preprocessed text
            html.H5("Text used for clustering:"),
            html.Div(dash_table.DataTable(id="text_preprocess"), id="text_preprocess_div"),
        ]),
    ], className="custom-tab", selected_className="custom-tab--selected"
)

arguments = {
    "preprocessing_method": misc.HtmlElement("text_preprocess_checklist", "value"),
    "preprocessing_output": misc.HtmlElement("text_preprocess_div", "children"),
}
outputs = Output("text_preprocess_div", "children")


@cache.memoize()
def get_preprocessed_data(df, preprocessing_method):
    if df is not None and preprocessing_method:
        for method in preprocessing_method:
            df = data_preprocessing_options[method]().apply(df)

    return df


def get_data_preprocessing_output(df, preprocessing_method):
    df = get_preprocessed_data(df, preprocessing_method)

    return misc.generate_datatable(df, "text_preprocess", 5, max_cell_width=None)
