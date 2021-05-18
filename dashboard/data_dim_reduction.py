import dash_core_components as dcc
import dash_html_components as html
import dash_table

import pandas as pd

import model

import dashboard.app_misc as misc
from dashboard.cache import cache

from dashboard.data_to_array import get_data_as_array


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
def get_dim_reduction(data_df, dim_reduction_method, dim_reduction_options):
    return pd.DataFrame(dim_reductions.apply(dim_reduction_method, dim_reduction_options, data_df))


dim_reduction_tab = dcc.Tab(
    label="Dimensionality Reduction", children=[
        html.Div(id="dim_red_area", children=[
            # Choose data_to_array method
            dim_reductions.generate_dash_element(),
            # Display array
            html.H5("Dimensionality Reduced Array:", id="dim_red_table_header"),
            html.Div(dash_table.DataTable(id="dim_red_table"), id="dim_red_table_div"),
        ]),
    ], className="custom-tab", selected_className="custom-tab--selected"
)
