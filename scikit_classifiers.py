# scikit_classifier.py
# Isabelle Bouchard
# Carolyne Pelletier
# 2019-12-14
# IFT-6390

import csv
import argparse
import numpy as np
import pathlib
import math
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE
from sklearn.utils import shuffle

_TEST_SIZE = 0.2
_SEED = 544
_DATA_PATH = pathlib.Path("data/")

DATASETS = {
    "heart-statlog": "heart-statlog_csv.csv",
    "cervical-cancer": "clean_cervical-cancer_csv.csv"
}

CLASSIFIERS = {
    "MLP": (
        MLPClassifier(
            early_stopping=True,
            hidden_layer_sizes=(50, 50),
            validation_fraction=0.2,
            random_state=_SEED,
        ), {}
    ),
    "RF": (
        RandomForestClassifier(
            random_state=_SEED,
            max_depth=2,
            n_estimators=30,
            ), {}
    ),
    "Decision Tree": (
        DecisionTreeClassifier(
            max_depth=2,
            random_state=_SEED,
        ), {}
    ),
    "LR_L1": (
        LogisticRegression(
            solver="liblinear",
            multi_class="ovr",
            penalty="l1",
            C=0.1,
            class_weight="balanced",
            random_state=_SEED
        ), {}
    ),
    "LR_L2": (
        LogisticRegression(
            solver="liblinear",
            multi_class="ovr",
            penalty="l2",
            C=0.1,
            class_weight="balanced",
            random_state=_SEED
        ), {}
    ),

}

SELECTED_FEATURES = {
    "heart-statlog": [0, 1, 2, 8, 9, 11, 12, 13],
    "cervical-cancer": [3, 6, 8, 9, 12, 14, 16, 19, 20, 21, 22, 23, 24, 25, 26]
}

def run_grid_search(model_name, X_train, y_train):
    """
    Run grid search on a given classifier
    """
    clf, params = CLASSIFIERS[model_name]
    clf = GridSearchCV(clf, params, cv=5, n_jobs=-1)
    clf.fit(X_train, y_train)

    return clf

def _label_to_float(a_str):
    """
    Convert labels as str to float
    """
    if a_str == "present":
        label = 1.
    elif a_str == "absent":
        label = 0.
    else:
        label = float(a_str)

    return label

def _split_row(row, selected_indexes=None):
    """
    Split row into features, label.
    The last element of the row is always the label.
    """
    if selected_indexes is None:
        row_X = row[:-1]
    else:
        row_X = [
            el for i, el in enumerate(row[:-1]) if i in selected_indexes
        ]
    row_y = row[-1]

    return row_X, row[-1]

def read_dataset(dataset_name, select_features=False, head=False):
    """
    Read dataset from csv file and return it as a supervised X, y dataset.
    If select_features=True, returns only the selected indexes.
    If head=True, returns only the header names.
    If header=False, returns only the data.
    """
    data_path = _DATA_PATH / dataset_name / "data" / DATASETS[dataset_name]
    X, y = [], []
    selected_indexes = SELECTED_FEATURES[dataset_name] if select_features else None
    with open(data_path, "r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for i, row in enumerate(csv_reader):
            row_X, row_y = _split_row(row, selected_indexes)
            if i == 0:
                if head:
                    return row_X, row_y
                continue
            X.append([float(el) for el in row_X])
            y.append(_label_to_float(row_y))

    return X, y

def _get_dataset(dataset_name, balance=True, select_features=False):
    """
    Read and split dataset into train/test sets.
    If balance=True, applies SMOTHE algorithm to rebalance the data.
    """
    X, y = read_dataset(
                dataset_name,
                select_features=select_features
    )
    if balance:
        print("balancing the dataset")
        smote = SMOTE(random_state=42)
        X, y = smote.fit_resample(X, y)
    # Split train/test set
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                            test_size=_TEST_SIZE,
                                            shuffle=True,
                                            random_state=_SEED)

    return X_train, X_test, y_train, y_test

def _get_accuracy(model, X, y, z=1.96):
    """
    Return the accuracy of the predictions and the confidence.
    By default, z=1.96 for a 95% interval.
    """
    prediction = model.predict(X)
    accuracy = np.mean(prediction == y)
    confidence = z * math.sqrt( (accuracy * (1 - accuracy)) / len(y))

    return accuracy, confidence


def main(model_name, dataset_name, balance=True, select_features=False, verbose=True):
    """
    Main training function.

    Given a model and a dataset, run grid search to find the best estimator.
    """
    X_train, X_test, y_train, y_test = _get_dataset(
            dataset_name,
            balance,
            select_features=select_features)
    model = run_grid_search(model_name, X_train, y_train)
    accuracy, confidence = _get_accuracy(model, X_test, y_test)

    if verbose:
        print(f">>> {model_name}")
        print(f"valid score = {model.best_score_}")
        print(f"test score = {accuracy} +/- {confidence}")
        print(f"{model.best_params_}")

    return model


def _parse_args():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "-m",
        "--model-name",
        type=str,
        help=f"One of {list(CLASSIFIERS.keys())}",
    )
    parser.add_argument(
        "-d",
        "--dataset-name",
        type=str,
        help=f"One of {list(DATASETS.keys())}",
    )
    parser.add_argument(
        "-b",
        "--balance",
        action="store_true",
        help="balance dataset using SMOTE",
    )
    return vars(parser.parse_args())


if __name__ == "__main__":
    args = _parse_args()
    main(**args)
