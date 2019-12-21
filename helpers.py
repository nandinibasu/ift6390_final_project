# helpers.py
# Isabelle Bouchard
# Carolyne Pelletier
# 2019-12-14
# IFT-6390

from scikit_classifiers import DATASETS
from scikit_classifiers import CLASSIFIERS
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix
import numpy as np


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

def separate_by_gender(x,y):
    """
    female 0
    male 1
    """
    male_x = []
    male_y = []
    female_x = []
    female_y = []
    print(len(x))
    print()

    for i,j in zip(x,y):
        # print(i)
        print(i[1])
        if i[1] == 1:
            male_x.append(i)
            male_y.append(j)
        elif i[1] == 0:
            female_x.append(i)
            female_y.append(j)
    print("len of male  : ", len(male_y))
    print("len of female: ", len(female_y))
    return male_x, male_y, female_x, female_y


def save_conf_matrix_gender_specific(cm_male, cm_female, model_type, SMOTE):

    sns.set_style("whitegrid")

    plt.figure(figsize=(12, 5))

    title = "Confusion Matrices for Heart Disease using " + model_type +" "+ SMOTE

    plt.suptitle(title, fontsize=24)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)

    plt.subplot(1, 2, 1)
    plt.title("Male")

    sns.heatmap(cm_male, annot=True, cmap="Blues", fmt="d", cbar=False, annot_kws={"size": 24})
    plt.xlabel("Pred")
    plt.ylabel("True")

    b, t = plt.ylim()  # discover the values for bottom and top
    b += 0.5  # Add 0.5 to the bottom
    t -= 0.5  # Subtract 0.5 from the top
    plt.ylim(b, t)  # update the ylim(bottom, top) values

    plt.subplot(1, 2, 2)
    plt.title("Female")

    sns.heatmap(cm_female, annot=True, cmap="Blues", fmt="d", cbar=False, annot_kws={"size": 24})
    plt.xlabel("Pred")
    plt.ylabel("True")

    b, t = plt.ylim()  # discover the values for bottom and top
    b += 0.5  # Add 0.5 to the bottom
    t -= 0.5  # Subtract 0.5 from the top
    plt.ylim(b, t)  # update the ylim(bottom, top) values

    plt.savefig(title+".png")
    print("saved CM plot")

def plot_target_distribution(target):
    sns.set(style="whitegrid")
    sns.set_palette(sns.color_palette("Set2", n_colors=5))

    plt.figure(figsize=(15, 12))
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    y_df = pd.DataFrame({'target': target})
    plt.subplot(2, 2, 1)
    plt.title("Distribution of diseased and not diseased patients after SMOTE.")
    sns.countplot(data=y_df, x='target')
    plt.xlabel("Target (0 = no cervical cancer, 1= cervical cancer)")
    # plt.savefig('after_smote_{}.png'.format(dataset_name))
    print("saved figure")

def plot_conf_matrices(X, y, model, model_name, balance):
    male_x, male_y, female_x, female_y = separate_by_gender(X, y)

    male_prediction = model.predict(male_x)
    male_accuracy = np.mean(male_prediction == male_y)
    print("Male Acc  : ", male_accuracy)
    cm_male = confusion_matrix(male_y, male_prediction)

    female_prediction = model.predict(female_x)
    female_accuracy = np.mean(female_prediction == female_y)
    print("Female Acc: ", female_accuracy)
    print(female_prediction)
    cm_female = confusion_matrix(female_y, female_prediction)

    save_conf_matrix_gender_specific(cm_male, cm_female, model_type=model_name, SMOTE=balance)