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

_DATA_PATH = pathlib.Path("data/")
_VALID_DATASET_NAMES = ["heart-statlog", "cervical-cancer"]
_CLASSIFIERS = {
    "MLP": (
        MLPClassifier(hidden_layer_sizes=275, early_stopping=True, validation_fraction=0.005),
        {
            "hidden_layer_sizes": [225, 250, 275],
            # "learning_rate_init": [225, 250, 275],
        }
    ),
    "Random Forest": (
        RandomForestClassifier(random_state=42),
        {
            "n_estimators": [5, 10, 20],
            "max_depth": [5, 10, 20]
        }
    ),
    "Logistic Regression": (
        LogisticRegression(solver="saga", multi_class="liblinear"),
        {
            "penalty": ["l2", "l1", "elasticnet", "none"],
            "C": [1.0, 0.9, 0.8, 0.7]  # Smaller means more regularization
        }
    ),
}

def run_grid_search(clf, parameters, X_train, X_val, y_train, y_val):
    import pdb; pdb.set_trace()
    clf = GridSearchCV(clf, parameters, cv=5, n_jobs=-1)
    clf.fit(X_train, y_train)

    prediction = clf.predict(X_val)
    accuracy = np.mean(prediction == y_val)
    return accuracy, clf.best_params_


def _get_dataset(dataset_name):
    data_path = _DATA_PATH / dataset_name / "data" / f"{dataset_name}_csv.csv"
    data = []
    with open(data_path, "r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for i, row in enumerate(csv_reader):
            if i == 0:
                continue
            data.append((row[:-1], row[-1]))
    return zip(*data)


def main(X_train, X_val, y_train, y_val, clf_name):
    clf, params = _CLASSIFIERS[clf_name]
    print(f"Running GS for {clf_name}...")
    accuracy, best_params = run_grid_search(clf, params, X_train, X_val, y_train, y_val)
    print(f">>> {clf_name} score = {accuracy}")
    print(f"{best_params}")
    return accuracy, best_params


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
    parser.add_argument(
        "--test",
        action="store_true",
        help="Whether or not to run test predictions and to output to csv file",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    X, y = _get_dataset(args.dataset_name)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    main(X_train, X_val, y_train, y_val, clf_name=args.model_name)
