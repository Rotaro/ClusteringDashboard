import dash_core_components as dcc
import dash_html_components as html

import numpy as np
import matplotlib
import plotly.graph_objs as go

import model

import dashboard.app_misc as misc


plot_dim_reductions = misc.DropdownWithOptions(
    header="Choose dimensionality reduction method for plotting:", dropdown_id="plot_dim_reduction", dropdown_objects={
        "NMF": model.dim_reduction.NMF,
        "PCA": model.dim_reduction.PCA,
        "SVD": model.dim_reduction.SVD,
        "FactorAnalysis": model.dim_reduction.FactorAnalysis,
        "TSNE": model.dim_reduction.TSNE,
    }, include_refresh_button=True
)


def get_scatter_plots(coords, clusters, titles):
    # Generate unique colors (https://stackoverflow.com/a/55828367)
    np.random.seed(42)
    colors = np.random.choice(list(matplotlib.colors.cnames.values()), size=np.unique(clusters).size, replace=False)

    dims = list(zip(("x", "y", "z"), range(coords.shape[1])))
    scatter_class = go.Scatter3d if len(dims) == 3 else go.Scatter
    scatter_plots = []
    for i, cluster in enumerate(np.unique(clusters)):
        idx = clusters == cluster
        if idx.sum() == 0:
            continue

        scatter_plots.append(scatter_class(
            name="Cluster %d" % cluster,
            **{label: coords[idx, i] for label, i in dims},
            text=titles.values[idx],
            textposition="top center",
            mode="markers",
            marker=dict(size=5 if len(dims) == 3 else 12, symbol="circle", color=colors[i]),
        ))

    return scatter_plots


plotting_options_tab = dcc.Tab(
    label="Plotting Options", children=[
        html.Div(id="plotting_dim_red_area", children=[
            plot_dim_reductions.generate_dash_element(),
        ]),
    ], className="custom-tab", selected_className="custom-tab--selected"
)

plotting_tab = dcc.Tab(
    label="Plot", children=[dcc.Graph(id="scatter-plot")],
    className="custom-tab", selected_className="custom-tab--selected"
)
