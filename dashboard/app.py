import inspect
import sys
import os
sys.path.append(os.getcwd())  # This is here so that can start app from root of project..

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, State

import data.text_processing as text_processing

from dashboard.cache import cache

import dashboard.data_selection as data_selection
import dashboard.data_preprocessing as data_preprocessing
import dashboard.data_to_array as data_to_array
import dashboard.data_dim_reduction as data_dim_reduction
import dashboard.plotting as plotting
import dashboard.clustering as clustering
import dashboard.recommendation as recommendation

CACHE_CONFIG = {
    "CACHE_TYPE": "simple"
}

external_stylesheets = [
    # Dash CSS
    "https://codepen.io/chriddyp/pen/bWLwgP.css",
    # Loading screen CSS
    "https://codepen.io/chriddyp/pen/brPBPO.css"
]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

cache.init_app(app.server, config=CACHE_CONFIG)

# This dictionary explicitly maps names to html elements defined in UI components
name_to_html_element_pool = {
    **data_selection.arguments,
    **data_preprocessing.arguments,
    **data_to_array.arguments,
    **data_dim_reduction.arguments,
    **plotting.arguments,
    **clustering.arguments,
    **recommendation.arguments
}


def map_arguments(outputs):
    """Creates dash callback for function by mapping arguments to Inputs / States using name_to_html_element_pool."""
    def _map_arguments(func):
        inputs = []
        states = []
        for argument in inspect.getfullargspec(func).args:
            if argument.startswith("s_"):
                # s_ -> State
                states.append(State(*name_to_html_element_pool[argument.replace("s_", "")]))
            else:
                inputs.append(Input(*name_to_html_element_pool[argument]))

        return app.callback(outputs, inputs, states)(func)

    return _map_arguments


################################################################################################
# Layer / components of dashboard
app.layout = html.Div([
    html.Div([
        dcc.Tabs(id="tabs_1", children=[
            data_selection.data_selection_tab,
            data_preprocessing.data_preprocessing_tab,
            data_to_array.data_to_array_tab,
            data_dim_reduction.dim_reduction_tab,
            dcc.Tab(label="Hide", children=[], className="custom-tab", selected_className="custom-tab--selected"),
        ]),
    ], style={"border": "grey solid", "padding": "5px"}),

    html.Div([
        dcc.Tabs(id="tabs_2", children=[
            plotting.plotting_options_tab,
            clustering.clustering_tab,
            dcc.Tab(label="Hide", children=[], className="custom-tab", selected_className="custom-tab--selected"),
        ]),
    ], style={"marginTop": "10px", "padding": "5px", "border": "grey solid"}),

    html.Div(id="plot_area", children=[
        dcc.Tabs(id="tabs_3", children=[
            plotting.plot_tab,
            clustering.clusters_tab,
            recommendation.recommendation_tab,
        ], style={"border": "grey solid", "padding": "5px", "marginTop": "10px"})
    ], style={"marginTop": "10px", "padding": "5px", "border": "grey solid"})
], style={"background-color": "#f2f2f2", "margin": "20px"})


################################################################################################
# Helper functions which tie the UI components together
# Alternative would be to repeatedly call different UI components in update functions
def get_data(selected_data, data_sample_percent):
    return data_selection.get_data(selected_data, data_sample_percent)


def get_data_selected_columns(selected_data, data_sample_percent, selected_columns):
    df = get_data(selected_data, data_sample_percent)
    return data_selection.get_selected_columns(df, selected_columns)


def get_data_preprocessed(selected_data, data_sample_percent, selected_columns, preprocessing_method):
    df = get_data_selected_columns(selected_data, data_sample_percent, selected_columns)
    return data_preprocessing.get_preprocessed_data(df, preprocessing_method)


def get_data_as_array(selected_data, data_sample_percent, selected_columns, preprocessing_method,
                      data_to_array_method, data_to_array_options):
    df = get_data_preprocessed(selected_data, data_sample_percent, selected_columns, preprocessing_method)
    df_arr = data_to_array.get_data_as_array(df, data_to_array_method, data_to_array_options)

    return df, df_arr


def get_data_as_array_dim_red(selected_data, data_sample_percent, selected_columns,
                              preprocessing_method,
                              data_to_array_method, data_to_array_options,
                              dim_reduction, dim_reduction_options):
    df, df_arr = get_data_as_array(selected_data, data_sample_percent, selected_columns,
                                   preprocessing_method,
                                   data_to_array_method, data_to_array_options)
    df_arr_dim_red = df_arr
    if dim_reduction and dim_reduction_options:
        df_arr_dim_red = data_dim_reduction.get_dim_reduction(df_arr, dim_reduction, dim_reduction_options)

    return df, df_arr, df_arr_dim_red


def get_data_clustered(selected_data, data_sample_percent, selected_columns,
                       preprocessing_method,
                       data_to_array_method, data_to_array_options,
                       dim_reduction_method, dim_reduction_options,
                       clustering_method, clustering_options):
    df, df_arr, df_arr_dim_red = get_data_as_array_dim_red(
        selected_data, data_sample_percent, selected_columns,
        preprocessing_method, data_to_array_method, data_to_array_options,
        dim_reduction_method, dim_reduction_options
    )
    clusters = clustering.get_clusters(df_arr_dim_red, clustering_method, clustering_options)

    return df, df_arr, df_arr_dim_red, clusters


################################################################################################
# Generate callbacks for updating dropdowns (e.g. showing correct options when changing dimensionality reduction)
data_to_array.data_to_array_dropdown.generate_update_options_callback(app)
data_dim_reduction.dim_reduction_dropdown.generate_update_options_callback(app)
plotting.plotting_options_dropdown.generate_update_options_callback(app)
clustering.clustering_dropdown.generate_update_options_callback(app)


################################################################################################
# Functions with callbacks which update UI elements
@map_arguments(data_selection.outputs)
def update_data_selection(
        # Inputs
        selected_data,
        selected_data_percent
):
    return data_selection.get_data_selection_output(selected_data, selected_data_percent)


@map_arguments(data_preprocessing.outputs)
def update_data_preprocessing(
        # Inputs
        selected_data_percent, selected_columns, preprocessing_method,

        # States
        s_selected_data
):
    df = get_data_selected_columns(s_selected_data, selected_data_percent, selected_columns)

    return data_preprocessing.get_data_preprocessing_output(df, preprocessing_method)


@map_arguments(data_to_array.outputs)
def update_data_to_array(
        # Inputs
        data_to_array_method,
        data_to_array_options,

        # Additional triggers
        data_to_array_refresh,
        preprocessing_output,

        # States
        s_selected_data, s_selected_data_percent, s_selected_columns,
        s_preprocessing_method,
):
    df = get_data_preprocessed(s_selected_data, s_selected_data_percent, s_selected_columns,
                               s_preprocessing_method)
    return data_to_array.get_data_to_array_output(df, data_to_array_method, data_to_array_options)


@map_arguments(data_dim_reduction.outputs)
def update_dim_reduction(
        # Inputs
        dim_reduction_method,
        dim_reduction_options,

        # Additional triggers
        dim_reduction_refresh,
        data_to_array_table,

        # States
        s_selected_data, s_selected_data_percent, s_selected_columns,
        s_preprocessing_method,
        s_data_to_array_method, s_data_to_array_options,
):
    df, df_arr = get_data_as_array(s_selected_data, s_selected_data_percent, s_selected_columns,
                                   s_preprocessing_method,
                                   s_data_to_array_method, s_data_to_array_options)

    return data_dim_reduction.get_dim_reduction_output(df_arr, dim_reduction_method, dim_reduction_options)


@map_arguments(plotting.outputs)
def update_plot(
        # Inputs
        data_to_array_method, data_to_array_options,
        dim_reduction_method, dim_reduction_options,
        plot_dim_reduction_method, plot_dim_reduction_options,
        clustering_method, clustering_options,

        # Additional triggers
        dim_reduction_refresh,
        plot_dim_reduction_refresh, clustering_refresh,

        # States
        s_selected_data, s_selected_data_percent, s_selected_columns,
        s_preprocessing_method,
):
    _, df_arr, _, clusters = get_data_clustered(
        s_selected_data, s_selected_data_percent, s_selected_columns, s_preprocessing_method,
        data_to_array_method, data_to_array_options,
        dim_reduction_method, dim_reduction_options,
        clustering_method, clustering_options
    )

    titles = None
    if s_selected_data:
        titles = get_data(s_selected_data, s_selected_data_percent).org_title

    return plotting.get_plot_output(df_arr, plot_dim_reduction_method, plot_dim_reduction_options,
                                    clusters, titles)


@map_arguments(clustering.outputs)
def update_cluster_clustering(
        # Inputs
        data_to_array_method, data_to_array_options,
        dim_reduction_method, dim_reduction_options,
        plot_dim_reduction_method, plot_dim_reduction_options,
        clustering_method, clustering_options,

        # Additional triggers
        dim_reduction_refresh,
        plot_dim_reduction_refresh, clustering_refresh,

        # States
        s_selected_data, s_selected_data_percent, s_selected_columns,
        s_preprocessing_method,
):
    _, _, df_arr_dim_red = get_data_as_array_dim_red(
        s_selected_data, s_selected_data_percent, s_selected_columns, s_preprocessing_method,
        data_to_array_method, data_to_array_options,
        dim_reduction_method, dim_reduction_options
    )

    titles = None
    if s_selected_data:
        titles = get_data(s_selected_data, s_selected_data_percent).org_title

    bow = None
    if s_selected_data:
        bow = text_processing.BOW(ngram_range=(1, 1), min_df=1).apply(
            get_data_preprocessed(s_selected_data, s_selected_data_percent, s_selected_columns, s_preprocessing_method)
        )

    return clustering.get_clustering_cluster_output(df_arr_dim_red, clustering_method, clustering_options, titles, bow)


@map_arguments(recommendation.outputs)
def update_recommendation(
        # Inputs
        recommendation_title, recommendation_metric,
        clustering_method, clustering_options,

        # Additional triggers
        clustering_refresh, cluster_info_table,

        # States
        s_selected_data, s_selected_data_percent, s_selected_columns,
        s_preprocessing_method,
        s_data_to_array_method, s_data_to_array_options,
        s_dim_reduction_method, s_dim_reduction_options
):
    n_recommendations = 20

    df, df_arr, df_arr_dim_red, clusters, titles = None, None, None, None, None
    if s_selected_data and s_selected_columns and clustering_options:
        df, df_arr, df_arr_dim_red, clusters = get_data_clustered(
            s_selected_data, s_selected_data_percent, s_selected_columns,
            s_preprocessing_method, s_data_to_array_method, s_data_to_array_options,
            s_dim_reduction_method, s_dim_reduction_options,
            clustering_method, clustering_options
        )

        titles = get_data(s_selected_data, s_selected_data_percent).org_title

    return recommendation.get_recommendation_output(recommendation_title, recommendation_metric, n_recommendations,
                                                    df_arr_dim_red, clusters, titles)


if __name__ == "__main__":
    app.run_server(host='0.0.0.0', debug=True)
