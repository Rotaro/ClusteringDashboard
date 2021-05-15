import dash_core_components as dcc
import dash_html_components as html
import dash_table

import data.text_processing as text_processing

from dashboard.cache import cache

from dashboard.data_selection import get_selected_columns


preprocessing = {
    "WikipediaTextCleanup": text_processing.WikipediaTextCleanup,
    "Stem": text_processing.Stem,
    "Lemmatize": text_processing.Lemmatize,
}


@cache.memoize()
def get_preprocessed_data(data_name, use_sample_perc, selected_columns, selected_preprocessing):
    df = get_selected_columns(data_name, use_sample_perc, selected_columns)
    if df is not None and selected_preprocessing:
        for preprocess_div in selected_preprocessing:
            name, active = preprocess_div["props"]["id"], preprocess_div["props"]["value"]
            if not active:
                continue
            df = preprocessing[name]().apply(df)

        return df


data_preprocessing_tab = dcc.Tab(
    label="Text Preprocessing", children=[
        html.Div(id="text_preprocess_area", children=[
            # Choose preprocessing
            html.Div([
                html.H5("Text preprocessing:"),
                html.Div([dcc.Checklist(id=name, options=[{"label": name, "value": name}], value=[],
                                        style={"padding": "5px", "margin": "5px"})
                          for name, cls in preprocessing.items()],
                         id="text_preprocess_picker")
            ], id="text_preprocess_picker_div"),
            # Display preprocessed text
            html.H5("Text used for clustering:"),
            html.Div(dash_table.DataTable(id="text_preprocess"), id="text_preprocess_div"),
        ]),
    ], className="custom-tab", selected_className="custom-tab--selected"
)

