# scikit_classifier.py
# Isabelle Bouchard
# Carolyne Pelletier
# 2019-12-14
# IFT-6390

import csv
import argparse
import numpy as np
import pathlib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

_TEST_SIZE = 0.2
_SEED = 544
_DATA_PATH = pathlib.Path("data/")
_VALID_DATASET_NAMES = ["heart-statlog", "cervical-cancer"]
_CLASSIFIERS = {
    "MLP": (
        MLPClassifier(early_stopping=True, random_state=_SEED),
        {
            "validation_fraction": [0.005, 0.01, 0.05, 0.1],
            "hidden_layer_sizes": [(10, 10), 25, (25, 25), 50, 100, (50, 50), 150, 250],
            "alpha": [0.0001, 0.001, 0.005, 0.00005, 0.]
        }
    ),
    "RF": (
        RandomForestClassifier(random_state=_SEED),
        {
            "n_estimators": [5, 10, 20],
            "max_depth": [5, 10, 20]
        }
    ),
    "LR": (
        LogisticRegression(solver="liblinear", multi_class="ovr", random_state=_SEED),
        {
            "penalty": ["l2", "l1"],
            # Smaller means more regularization
            "C": [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]
        }
    ),
}

def run_grid_search(model_name, X_train, y_train):
    clf, params = _CLASSIFIERS[model_name]

    clf = GridSearchCV(clf, params, cv=5, n_jobs=-1)
    clf.fit(X_train, y_train)
    return clf

def _to_float(a_str):
    if a_str == "present":
        return 1.
    elif a_str == "absent":
        return 0.

    try:
        a_float = float(a_str)
    except ValueError:
        # TODO handle missing data
        a_float = 0.
    return a_float


def _get_dataset(dataset_name):
    data_path = _DATA_PATH / dataset_name / "data" / f"{dataset_name}_csv.csv"
    X, y = [], []
    with open(data_path, "r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for i, row in enumerate(csv_reader):
            if i == 0:
                continue
            X.append([_to_float(el) for el in row[:-1]])
            y.append(_to_float(row[-1]))
    return train_test_split(X, y, test_size=_TEST_SIZE, random_state=_SEED, shuffle=True)

def _get_accuracy(model, X, y):
    prediction = model.predict(X)
    accuracy = np.mean(prediction == y)
    return accuracy


def main(model_name, dataset_name):
    X_train, X_test, y_train, y_test = _get_dataset(dataset_name)

    print(f">>> {model_name}")
    model = run_grid_search(model_name, X_train, y_train)
    accuracy = _get_accuracy(model, X_test, y_test)

    print(f"valid score = {model.best_score_}")
    print(f"test score = {accuracy}")
    print(f"{model.best_params_}")
    return model


def _parse_args():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "-m",
        "--model-name",
        type=str,
        help=f"One of {list(_CLASSIFIERS.keys())}",
    )
    parser.add_argument(
        "-d",
        "--dataset-name",
        type=str,
        help=f"One of {_VALID_DATASET_NAMES}",
    )
    return vars(parser.parse_args())


if __name__ == "__main__":
    args = _parse_args()
    main(**args)
