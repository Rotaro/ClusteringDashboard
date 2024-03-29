import dash_core_components as dcc
import dash_html_components as html
import dash_table
from dash.dependencies import Output

import data.text_processing as text_processing

import dashboard.app_misc as misc
from dashboard.cache import cache


data_to_array_dropdown = misc.DropdownWithOptions(
    header="Choose text to array method:", dropdown_id="to_array", dropdown_objects={
        "TFIDF": text_processing.TFIDF,
        "BOW": text_processing.BOW,
    }, include_refresh_button=True
)
if text_processing.fasttext is not None:
    data_to_array_dropdown.dropdown_objects["FastText"] = text_processing.FastText
if text_processing.fasttext is not None and text_processing.FastTextPretrained.has_pretrained():
    data_to_array_dropdown.dropdown_objects["FastTextPretrained"] = text_processing.FastTextPretrained

data_to_array_tab = dcc.Tab(
    label="Text to Array", children=[
        html.Div(id="data_to_array_area", children=[
            # Choose data_to_array_method
            data_to_array_dropdown.generate_dash_element(),
            # Display array
            html.H5("Data array:", id="data_to_array_header"),
            html.Div(dash_table.DataTable(id="data_to_array_table"), id="data_to_array_div"),
        ]),
    ], className="custom-tab", selected_className="custom-tab--selected"
)

arguments = {
    "data_to_array_method": misc.HtmlElement(*data_to_array_dropdown.dropdown_args),
    "data_to_array_options": misc.HtmlElement(*data_to_array_dropdown.options_args),
    "data_to_array_refresh": misc.HtmlElement(*data_to_array_dropdown.refresh_args),
    "data_to_array_table": misc.HtmlElement("data_to_array_div", "children"),
}
outputs = [Output("data_to_array_div", "children"), Output("data_to_array_header", "children")]


@cache.memoize()
def get_data_as_array(df, data_to_array_method, data_to_array_options):
    df_arr = None
    if df is not None and data_to_array_options:
        df_arr = data_to_array_dropdown.apply(data_to_array_method, data_to_array_options, df)

    return df_arr


def get_data_to_array_output(df, data_to_array_method, data_to_array_options):
    df_arr = get_data_as_array(df, data_to_array_method, data_to_array_options)

    data_to_array_header = "Array to cluster (shape %dx%d):" % ((0, 0) if df_arr is None else df_arr.shape)
    df_sample = df_arr.sample(min(df_arr.shape[1], 20), axis=1).round(2) if df_arr is not None else None

    return misc.generate_datatable(df_sample, "data_to_array", 5, max_cell_width=None), \
           data_to_array_header
