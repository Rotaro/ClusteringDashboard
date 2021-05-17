import dash_core_components as dcc
import dash_html_components as html
import dash_table

import data.text_processing as text_processing

import dashboard.app_misc as misc
from dashboard.cache import cache

from dashboard.data_preprocessing import get_preprocessed_data

processing = misc.DropdownWithOptions(
    header="Choose text to array method:", dropdown_id="to_array", dropdown_objects={
        "TFIDF": text_processing.TFIDF,
        "BOW": text_processing.BOW,
    }, include_refresh_button=True
)
if text_processing.fasttext is not None:
    processing.dropdown_objects["FastText"] = text_processing.FastText
if text_processing.fasttext is not None and text_processing.FastTextPretrained.has_pretrained():
    processing.dropdown_objects["FastTextPretrained"] = text_processing.FastTextPretrained


@cache.memoize()
def get_data_as_array(data_source, data_sample_percent, selected_columns,
                      selected_preprocessing,
                      data_to_array_method, data_to_array_options):
    df = get_preprocessed_data(data_source, data_sample_percent, selected_columns, selected_preprocessing)
    data_df = None
    if df is not None and data_to_array_options:
        data_df = processing.apply(data_to_array_method, data_to_array_options, df)

    return df, data_df


text_to_array_tab = dcc.Tab(
    label="Text to Array", children=[
        html.Div(id="data_to_array_area", children=[
            # Choose data_to_array_method
            processing.generate_dash_element(),
            # Display array
            html.H5("Data array:", id="data_to_array_header"),
            html.Div(dash_table.DataTable(id="data_to_array_table"), id="data_to_array_div"),
        ]),
    ], className="custom-tab", selected_className="custom-tab--selected"
)
