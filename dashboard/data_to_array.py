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
def get_cluster_data(data_name, use_sample_perc, selected_columns, selected_preprocessing, chosen_to_array, to_array_options):
    df = get_preprocessed_data(data_name, use_sample_perc, selected_columns, selected_preprocessing)
    data_df = None
    if df is not None and to_array_options:
        data_df = processing.apply(chosen_to_array, to_array_options, df)

    return df, data_df


text_to_array_tab = dcc.Tab(
    label="Text to Array", children=[
        html.Div(id="text_to_array_area", children=[
            # Choose text_to_array method
            processing.generate_dash_element(),
            # Display array
            html.H5("Cluster array:", id="text_to_array_header"),
            html.Div(dash_table.DataTable(id="text_to_array"), id="text_to_array_div"),
        ]),
    ], className="custom-tab", selected_className="custom-tab--selected"
)
