import dash_core_components as dcc
import dash_html_components as html
import dash_table
from dash.dependencies import Output

import pandas as pd

import model

import dashboard.app_misc as misc
from dashboard.cache import cache


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
def get_dim_reduction(df_arr, dim_reduction_method, dim_reduction_options):
    if dim_reduction_method and dim_reduction_options:
        return pd.DataFrame(dim_reductions.apply(dim_reduction_method, dim_reduction_options, df_arr))

    return None


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


def get_dim_reduction_output(df_arr, dim_reduction_method, dim_reduction_options):
    df_arr_dim_red = get_dim_reduction(df_arr, dim_reduction_method, dim_reduction_options)

    sample_df = None
    if df_arr_dim_red is not None:
        sample_df = df_arr_dim_red.sample(min(df_arr_dim_red.shape[1], 20), axis=1).round(2)

    return misc.generate_datatable(sample_df, "dim_red_table", 5, max_cell_width="350px")


arguments = {
    "dim_reduction_method": dim_reductions._dropdown_args,
    "dim_reduction_options": dim_reductions._options_args,
    "dim_reduction_refresh": dim_reductions._refresh_args,
}
outputs = Output("dim_red_table_div", "children")
