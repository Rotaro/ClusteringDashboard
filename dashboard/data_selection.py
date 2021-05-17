import dash_core_components as dcc
import dash_html_components as html
import dash_daq
import dash_table

import data.text_processing as text_processing

from data.tv_series_data import get_imdb_top_250_wikipedia_summaries
from data.tv_series_data import get_all_wikipedia_summaries

from dashboard.cache import cache

data_sources = {
    "Top 250 IMDB TV Series": get_imdb_top_250_wikipedia_summaries,
    "Wikipedia TV Series Summaries": get_all_wikipedia_summaries
}


@cache.memoize()
def get_data(data_source, data_sample_percent=100):
    df = data_sources[data_source]() if data_source is not None else None

    if df is not None and "org_title" not in df.columns:
        df["org_title"] = df["title"]

    if df is not None and data_sample_percent < 100:
        df = df.sample(frac=data_sample_percent / 100)

    return df


@cache.memoize()
def get_selected_columns(data_source, data_sample_percent, selected_columns):
    df = get_data(data_source, data_sample_percent)
    if df is not None and selected_columns is not None and len(selected_columns) > 0:
        return text_processing.join_columns(df, selected_columns)


data_selection_tab = dcc.Tab(
    label="Data Selection", children=[
        html.Div(id="data_selection_area", children=[
            # Choose data
            html.H5(children="Select data:"),
            dcc.Dropdown(
                id="data",
                options=[{"label": name, "value": name} for name, func in data_sources.items()]
            ),
            # Choose columns
            html.H5("Percentage of data to use:"),
            html.Div(dash_daq.NumericInput("data_sample_percent", value=100, min=0, max=100)),
            # Choose columns
            html.H5("Select columns to use:"),
            html.Div(dcc.Dropdown(id="data_column_selector"), id="data_column_selector_div"),
            # Display top rows
            html.H5("Top rows:", id="data_top_rows"),
            html.Div(dash_table.DataTable(id="data_top_rows_table"), id="data_top_rows_div")
        ]),
    ], className="custom-tab", selected_className="custom-tab--selected"
)
