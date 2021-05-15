import dash_core_components as dcc
import dash_html_components as html
import dash_table

import pandas as pd

import model

import dashboard.app_misc as misc
from dashboard.cache import cache

from dashboard.data_to_array import get_cluster_data


dim_reductions = misc.DropdownWithOptions(
    header="Choose dimensionality reduction before clustering:", dropdown_id="dim_reduction", dropdown_objects={
        "None": None,
        "NMF": model.dim_reduction.NMF,
        "PCA": model.dim_reduction.PCA,
        "SVD": model.dim_reduction.SVD,
        "FactorAnalysis": model.dim_reduction.FactorAnalysis,
        "TSNE": model.dim_reduction.TSNE,
    }, include_refresh_button=True
)


@cache.memoize()
def get_dim_reduction(data_name, use_sample_perc, selected_columns, selected_preprocessing, chosen_to_array,
                      to_array_options, dim_reduction, dim_reduction_options):
    df, data_df = get_cluster_data(data_name, use_sample_perc, selected_columns, selected_preprocessing,
                                   chosen_to_array, to_array_options)
    if df is None or data_df is None or not dim_reduction_options:
        return df, data_df, None

    return df, data_df, pd.DataFrame(dim_reductions.apply(dim_reduction, dim_reduction_options, data_df))


dim_reduction_tab = dcc.Tab(
    label="Dimensionality Reduction", children=[
        html.Div(id="dim_red_area", children=[
            # Choose text_to_array method
            dim_reductions.generate_dash_element(),
            # Display array
            html.H5("Dimensionality Reduced Array:", id="dim_red_array_header"),
            html.Div(dash_table.DataTable(id="dim_red_array"), id="dim_red_array_div"),
        ]),
    ], className="custom-tab", selected_className="custom-tab--selected"
)
