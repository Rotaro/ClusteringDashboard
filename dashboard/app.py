import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go

import pandas as pd
import numpy as np

import data.text_processing as text_processing

import dashboard.app_misc as misc
from dashboard.cache import cache

from dashboard.data_selection import get_data_source, data_selection_tab
from dashboard.data_preprocessing import get_preprocessed_data, data_preprocessing_tab
from dashboard.data_to_array import get_cluster_data, processing as data_to_array_processing, text_to_array_tab
from dashboard.data_dim_reduction import get_dim_reduction, dim_reductions, dim_reduction_tab
from dashboard.plotting import get_scatter_plots, plot_dim_reductions, plotting_options_tab, plotting_tab
from dashboard.clustering import get_clusters, get_cluster_info_df, clusterings, clustering_tab, clusters_tab
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

# This dictionary explicitly maps argument names to html elements for callbacks
argument_pool = {
    "selected_data": ("data", "value"),
    "selected_data_percent": ("input_sample_perc", "value"),
    "selected_columns": ("data_column_selector", "value"),

    "preprocessing_method": ("text_preprocess_picker", "value"),
    "preprocessing_output": ("text_preprocess_div", "children"),

    "data_to_array_method": data_to_array_processing._dropdown_args,
    "data_to_array_options": data_to_array_processing._options_args,
    "data_to_array_refresh": data_to_array_processing._refresh_args,
    "data_to_array_table": ("text_to_array_div", "children"),

    "dim_reduction_method": dim_reductions._dropdown_args,
    "dim_reduction_options": dim_reductions._options_args,
    "dim_reduction_refresh": dim_reductions._refresh_args,

    "plot_dim_reduction_method": plot_dim_reductions._dropdown_args,
    "plot_dim_reduction_options": plot_dim_reductions._options_args,
    "plot_dim_reduction_refresh": dim_reductions._refresh_args,

    "clustering_method": clusterings._dropdown_args,
    "clustering_options": clusterings._options_args,
    "clustering_refresh": clusterings._refresh_args,
    "cluster_info_table": ("cluster_info_table", "children"),

    "recommendation_title": ("recommendation_picker", "value"),
    "recommendation_metric": ("recommendation_metric", "value"),
}


def map_arguments(outputs):
    """Maps function arguments to Input / State using parameter_pool."""
    def _map_arguments(func):
        import inspect

        inputs = []
        states = []
        for argument in inspect.getfullargspec(func).args:
            if argument.startswith("s_"):
                # s_ -> State
                states.append(State(*argument_pool[argument.replace("s_", "")]))
            else:
                inputs.append(Input(*argument_pool[argument]))

        return app.callback(outputs, inputs, states)(func)

    return _map_arguments


# Layer / components of dashboard
app.layout = html.Div([
    html.Div([
        dcc.Tabs(id="tabs_1", children=[
            data_selection_tab,
            data_preprocessing_tab,
            text_to_array_tab,
            dim_reduction_tab,
            dcc.Tab(label="Hide", children=[], className="custom-tab", selected_className="custom-tab--selected"),
        ]),
    ], style={"border": "grey solid", "padding": "5px"}),

    html.Div([
        dcc.Tabs(id="tabs_2", children=[
            plotting_options_tab,
            clustering_tab,
            dcc.Tab(label="Hide", children=[], className="custom-tab", selected_className="custom-tab--selected"),
        ]),
    ], style={"marginTop": "10px", "padding": "5px", "border": "grey solid"}),

    html.Div(id="plot_area", children=[
        dcc.Tabs(id="tabs_3", children=[
            plotting_tab,
            clusters_tab,
            recommendation.recommendation_tab,
        ], style={"border": "grey solid", "padding": "5px", "marginTop": "10px"})
    ], style={"marginTop": "10px", "padding": "5px", "border": "grey solid"})
], style={"background-color": "#f2f2f2", "margin": "20px"})


# Generate callbacks for updating dropdowns (e.g. showing correct options when changing dimensionality reduction)
data_to_array_processing.generate_update_options_callback(app)
dim_reductions.generate_update_options_callback(app)
plot_dim_reductions.generate_update_options_callback(app)
clusterings.generate_update_options_callback(app)


@map_arguments([Output("data_head_div", "children"), Output("data_column_selector_div", "children"),
                Output("data_top_rows", "children")])
def update_chosen_data(
        selected_data,
        selected_data_percent
):
    df = get_data_source(selected_data, selected_data_percent)
    top_rows_text = "Top Rows (%d rows total)" % (0 if df is None else len(df))

    return misc.generate_datatable(df, "data_head_table", 5),\
           misc.generate_column_picker(df, "data_column_selector"), \
           top_rows_text


@map_arguments(Output("text_preprocess_div", "children"))
def update_text_preprocess_area(
    # Inputs
    selected_data_percent, selected_columns, preprocessing_method,

    # States
    s_selected_data
):
    df = get_preprocessed_data(s_selected_data, selected_data_percent, selected_columns, preprocessing_method)

    return misc.generate_datatable(df, "text_preprocess", 5, max_cell_width=None)


@map_arguments(
    [Output("text_to_array_div", "children"), Output("text_to_array_header", "children")],
)
def update_text_to_array_area(
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
    df, data_df = get_cluster_data(s_selected_data, s_selected_data_percent, s_selected_columns,
                                   s_preprocessing_method,
                                   data_to_array_method, data_to_array_options)
    text_to_array_header = "Array to cluster (shape %dx%d):" % ((0, 0) if data_df is None else data_df.shape)
    sample_df = data_df.sample(min(data_df.shape[1], 20), axis=1).round(2) if data_df is not None else None

    return misc.generate_datatable(sample_df, "text_to_array", 5, max_cell_width=None), \
           text_to_array_header


@map_arguments(Output("dim_red_array_div", "children"))
def update_dim_red_area(
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
    df, data_df, dim_red_df = get_dim_reduction(s_selected_data, s_selected_data_percent, s_selected_columns,
                                                s_preprocessing_method,
                                                s_data_to_array_method, s_data_to_array_options,
                                                dim_reduction_method, dim_reduction_options)

    sample_df = dim_red_df.sample(min(dim_red_df.shape[1], 20), axis=1).round(2) if dim_red_df is not None else None

    return misc.generate_datatable(sample_df, "dim_red_array", 5, max_cell_width="350px")


@map_arguments(
    [Output("scatter-plot", "figure"), Output("cluster_info_table", "children"),
     Output("cluster_info_text", "children")]
)
def plot(
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

    if s_selected_data is None or not s_selected_columns or not plot_dim_reduction_options:
        return go.Figure(layout=go.Layout(margin=dict(l=0, r=0, b=0, t=0), plot_bgcolor="#f2f2f2")), None, None

    # Cluster
    _, data_df, _, clusters = get_clusters(
        s_selected_data, s_selected_data_percent, s_selected_columns, s_preprocessing_method,
        data_to_array_method, data_to_array_options,
        dim_reduction_method, dim_reduction_options,
        clustering_method, clustering_options
    )
    # Get original dataframe to make sure titles are included
    df = get_data_source(s_selected_data, s_selected_data_percent)

    # Plots
    _, _, coords_df = get_dim_reduction(s_selected_data, s_selected_data_percent, s_selected_columns,
                                        s_preprocessing_method,
                                        data_to_array_method, data_to_array_options,
                                        plot_dim_reduction_method, plot_dim_reduction_options)
    scatter_plots = get_scatter_plots(coords_df.values, clusters, df.org_title)
    figure = go.Figure(data=scatter_plots, layout=go.Layout(margin=dict(l=0, r=0, b=0, t=0), plot_bgcolor="#f2f2f2",
                                                            legend={"bgcolor": "#f2f2f2"}, hovermode="closest"))
    # Cluster information
    bow_data_df = text_processing.TFIDF(ngram_range=(1, 1)).apply(
        get_preprocessed_data(s_selected_data, s_selected_data_percent, s_selected_columns,  s_preprocessing_method)
    )
    cluster_info_df = get_cluster_info_df(10, clusters, df.org_title, bow_data_df)

    from sklearn.metrics.cluster import silhouette_score
    if np.unique(clusters).size > 1:
        cluster_info_score = "Silhouette Score: %.2f" % silhouette_score(data_df.values, clusters)
    else:
        cluster_info_score = None

    return figure, misc.generate_datatable(cluster_info_df, "cluster_info", 1000, "600px"), cluster_info_score


@map_arguments([Output("recommendations", "children"), Output("recommendation_picker", "options")])
def recommend(
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
    if s_selected_data is None or not s_selected_columns or not clustering_options:
        return None, []

    df, data_df, dim_red_df, clusters = get_clusters(
        s_selected_data, s_selected_data_percent, s_selected_columns,
        s_preprocessing_method, s_data_to_array_method, s_data_to_array_options,
        s_dim_reduction_method, s_dim_reduction_options,
        clustering_method, clustering_options
    )
    data_df = dim_red_df if dim_red_df is not None else data_df
    titles = get_data_source(s_selected_data, s_selected_data_percent).org_title

    recommendations = None
    titles_for_recommendation = [{"label": org_title, "value": org_title} for org_title in sorted(titles)]

    if recommendation_title:
        recommendation_df = recommendation.get_recommendations(20, recommendation_title, recommendation_metric,
                                                               data_df, clusters, titles)
        recommendations = misc.generate_datatable(recommendation_df.round(2), "recommendations_table")

    return recommendations, titles_for_recommendation


if __name__ == "__main__":
    app.run_server(host='0.0.0.0', debug=True)
