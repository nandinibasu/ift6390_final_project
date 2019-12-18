# helpers.py
# Isabelle Bouchard
# Carolyne Pelletier
# 2019-12-14
# IFT-6390

from scikit_classifiers import DATASETS
from scikit_classifiers import CLASSIFIERS
import plotly.graph_objects as go


def print_models():
    _MODELS = list(CLASSIFIERS.keys())
    print(f"Models")
    for i, m in enumerate(_MODELS):
        print(f"{i}: {m}")

def print_datasets():
    _DATASETS = list(DATASETS.keys())
    print(f"Datasets")
    for i, m in enumerate(_DATASETS):
        print(f"{i}: {m}")


def plot_grid_search(cv_results, model_name, dataset_name, balance=True):
    """
    Helper to plot grid search using plotly.
    """
    params = cv_results["params"]
    param_names = list(params[0].keys())
    if len(param_names) == 0:
        print("No parameters to show!")
        return None
    # Special case for MLP hidden layers
    elif len(param_names) == 1:
        x_title = "1st hidden layer"
        y_title = "2nd hidden layer"
        x, y = [], []
        for p in params:
            x.append(
                p[param_names[0]][0]
                if isinstance(p[param_names[0]], tuple)
                else p[param_names[0]]
            )
            y.append(
                p[param_names[0]][0]
                if isinstance(p[param_names[0]], tuple)
                else 0
            )

    elif len(param_names) == 2:
        x_title = list(params[0].keys())[0]
        y_title = list(params[0].keys())[1]
        x = [p[x_title] for p in params]
        y = [p[y_title] for p in params]


    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        hovertext=cv_results["mean_test_score"],
        marker=dict(
            size=10,
            color=cv_results["mean_test_score"],
            colorbar=dict(
                title="mean_f1_score"
            ),
            colorscale="Magma"
        ),
        mode="markers")
    )
    fig.update_layout(
        title=f"Grid Search for {model_name} \
                on {dataset_name} dataset\
                {' balanced' if balance else ''}",
        xaxis_title=x_title,
        yaxis_title=repr(y_title),
    )

    return fig
