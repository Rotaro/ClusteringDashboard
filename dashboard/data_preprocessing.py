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
def get_preprocessed_data(data_name, use_sample_perc, selected_columns, preprocessing_method):
    df = get_selected_columns(data_name, use_sample_perc, selected_columns)

    if preprocessing_method:
        for method in preprocessing_method:
            df = preprocessing[method]().apply(df)

    return df


data_preprocessing_tab = dcc.Tab(
    label="Text Preprocessing", children=[
        html.Div(id="text_preprocess_area", children=[
            # Choose preprocessing
            html.Div([
                html.H5("Text preprocessing:"),
                dcc.Checklist(id="text_preprocess_picker",
                              options=[{"label": name, "value": name} for name, cls in preprocessing.items()], value=[],
                              style={"padding": "5px", "margin": "5px"})
            ], id="text_preprocess_picker_div"),
            # Display preprocessed text
            html.H5("Text used for clustering:"),
            html.Div(dash_table.DataTable(id="text_preprocess"), id="text_preprocess_div"),
        ]),
    ], className="custom-tab", selected_className="custom-tab--selected"
)

